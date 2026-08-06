from __future__ import annotations

import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from filelore.audio import AudioRange
from filelore.embedding import AudioEmbedding, AudioInput, EmbeddingVector
from filelore.index import (
    FileIndexer,
    FileIndexRepository,
    FileMetadataQuery,
    file_metadata_filter,
    file_point_id,
    file_segment_point_id,
)
from filelore.metadata import AudioMetadata, AudioMetadataParser
from filelore.processors import AudioProcessor, PreparedFile, PreparedSegment
from filelore.storage import DistanceMetric, VectorConfig
from filelore.storage.qdrant import QdrantVectorDatabase


def create_wave(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 100
    frame_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


class RangeAudioDecoder:
    def decode(
        self,
        path: str | Path,
        audio_range: AudioRange,
        *,
        target_sampling_rate: int,
    ) -> AudioInput:
        return AudioInput(
            samples=(audio_range.start_seconds + 1.0, 0.0, 0.0),
            sampling_rate=target_sampling_rate,
        )


class RangeAudioEmbedding(AudioEmbedding):
    sampling_rate = 48_000
    max_length_seconds = 10.0
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(
            model_id="test-audio-model",
            vector_name="audio_test",
            dimensions=3,
        )

    def predict_batch(
        self, items: Sequence[AudioInput]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(
            (1.0, 0.0, 0.0)
            if item.samples[0] == 1.0
            else (0.0, 1.0, 0.0)
            for item in items
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)


def segment_config() -> dict[str, VectorConfig]:
    return {
        "audio_test": VectorConfig(3, distance=DistanceMetric.COSINE),
    }


@pytest.mark.parametrize(
    ("extension", "file_format"),
    (
        ("aif", "aif"),
        ("aif", "aiff"),
        ("aiff", "aif"),
        ("aiff", "aiff"),
        ("wav", "wav"),
        ("wav", "wave"),
        ("wave", "wav"),
        ("wave", "wave"),
    ),
)
def test_audio_format_filter_normalizes_extension_aliases(
    tmp_path: Path,
    extension: str,
    file_format: str,
) -> None:
    audio_path = tmp_path / f"effect.{extension}"
    audio_path.write_bytes(b"audio placeholder")
    metadata = AudioMetadata(
        path=audio_path,
        extension=f".{extension}",
        mime_type=None,
        size_bytes=audio_path.stat().st_size,
        modified_at=datetime.now(timezone.utc),
        duration_seconds=1.0,
        sample_rate_hz=48_000,
        channels=2,
        bitrate_bps=192_000,
        bits_per_sample=16,
        audio_format=extension.upper(),
    )

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(database)
        repository.store(metadata)

        results = repository.search_files(
            FileMetadataQuery(file_format=file_format)
        )

    assert [result.path for result in results] == [audio_path.resolve()]


def test_file_indexer_processes_audio_and_stores_timed_child_points(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "effects.wav"
    create_wave(audio_path, duration_seconds=12.0)
    embedding = RangeAudioEmbedding()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(
            database,
            segment_vector_configs=segment_config(),
        )
        indexer = FileIndexer(
            repository,
            AudioProcessor(
                embedding=embedding,
                decoder=RangeAudioDecoder(),
            ),
        )

        result = indexer.index_batch(tuple(indexer.discover(tmp_path)))

        assert result.failures == ()
        assert [entry.path for entry in result.entries] == [audio_path.resolve()]
        assert repository.count() == 1
        assert database.count(repository.segment_collection_name) == 2

        parent = database.retrieve(
            repository.collection_name,
            (file_point_id(audio_path),),
        )[0]
        segments = database.retrieve(
            repository.segment_collection_name,
            (
                file_segment_point_id(audio_path, 0),
                file_segment_point_id(audio_path, 1),
            ),
            with_vectors=True,
        )

        assert parent.payload["record_type"] == "file"
        assert parent.payload["segment_count"] == 2
        assert [item.payload["record_type"] for item in segments] == [
            "segment",
            "segment",
        ]
        assert [item.payload["parent_id"] for item in segments] == [
            file_point_id(audio_path),
            file_point_id(audio_path),
        ]
        assert [
            (
                item.payload["segment_index"],
                item.payload["segment_start_seconds"],
                item.payload["segment_end_seconds"],
            )
            for item in segments
        ] == [(0, 0.0, 10.0), (1, 5.0, 12.0)]
        assert [item.vectors["audio_test"] for item in segments] == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        results = repository.semantic_segment_search(
            [1.0, 0.0, 0.0],
            vector_name="audio_test",
            metadata_filter=file_metadata_filter(
                FileMetadataQuery(
                    file_format="wav",
                    sample_rate_hz=100,
                    bitrate_bps=1_600,
                    duration_longer_than=10.0,
                    duration_shorter_than=20.0,
                )
            ),
        )

        assert [result.file.path for result in results] == [
            audio_path.resolve(),
            audio_path.resolve(),
        ]
        assert [result.file.id for result in results] == [
            file_point_id(audio_path),
            file_point_id(audio_path),
        ]
        assert [result.segment.index for result in results if result.segment] == [
            0,
            1,
        ]
        assert results[0].score > results[1].score


def test_reindex_replaces_stale_segments_and_remove_cascades(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "effects.wav"
    create_wave(audio_path, duration_seconds=3.0)
    metadata = AudioMetadataParser().parse(audio_path)
    first = PreparedFile(
        metadata=metadata,
        vectors={},
        segments=tuple(
            PreparedSegment(
                index=index,
                start_seconds=float(index),
                end_seconds=float(index + 1),
                vectors={"audio_test": (1.0, 0.0, 0.0)},
            )
            for index in range(3)
        ),
    )
    replacement = PreparedFile(
        metadata=metadata,
        vectors={},
        segments=(
            PreparedSegment(
                index=0,
                start_seconds=0.0,
                end_seconds=3.0,
                vectors={"audio_test": (0.0, 1.0, 0.0)},
            ),
        ),
    )

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(
            database,
            segment_vector_configs=segment_config(),
        )
        repository.store_prepared(first)
        assert database.count(repository.segment_collection_name) == 3

        repository.store_prepared(replacement)

        assert repository.count() == 1
        assert database.count(repository.segment_collection_name) == 1
        assert database.retrieve(
            repository.segment_collection_name,
            (file_segment_point_id(audio_path, 1),),
        ) == ()

        repository.remove((audio_path,))

        assert repository.count() == 0
        assert database.count(repository.segment_collection_name) == 0


def test_segment_storage_requires_an_explicit_vector_configuration(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "effects.wav"
    create_wave(audio_path, duration_seconds=1.0)
    prepared = PreparedFile(
        metadata=AudioMetadataParser().parse(audio_path),
        vectors={},
        segments=(
            PreparedSegment(
                index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                vectors={"audio_test": (1.0, 0.0, 0.0)},
            ),
        ),
    )

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(database)

        with pytest.raises(ValueError, match="segment_vector_configs"):
            repository.store_prepared(prepared)

        assert repository.count() == 0


def test_file_indexer_preserves_processor_failures(tmp_path: Path) -> None:
    audio_path = tmp_path / "valid.wav"
    invalid_path = tmp_path / "invalid.wav"
    create_wave(audio_path, duration_seconds=1.0)
    invalid_path.write_text("not audio", encoding="utf-8")

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(database)
        indexer = FileIndexer(repository, AudioProcessor())

        result = indexer.index_batch((audio_path, invalid_path))

        assert [entry.path for entry in result.entries] == [audio_path.resolve()]
        assert len(result.failures) == 1
        assert result.failures[0].path == invalid_path
        assert repository.count() == 1


def test_segment_ids_are_stable_and_distinct_from_parent_ids(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "effects.wav"

    first = file_segment_point_id(audio_path, 0)

    assert first == file_segment_point_id(audio_path, 0)
    assert first != file_segment_point_id(audio_path, 1)
    assert first != file_point_id(audio_path)
