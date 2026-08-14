from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pytest

from filelore.audio import AudioRange
from filelore.embedding import (
    AudioEmbedding,
    AudioInput,
    BaseEmbedding,
    EmbeddingVector,
    TextEmbedding,
)
from filelore.index import FileIndexEntry, FileSearchResult, FileSegmentMatch
from filelore.processors import AudioProcessor
from filelore.search import (
    AudioFileQueryVectorizer,
    SearchSource,
    embed_search_source,
    search_vectors,
)


class RecordingTextEmbedding(TextEmbedding[str]):
    def __init__(self) -> None:
        super().__init__(model_id="text-test", vector_name="text_test", dimensions=2)
        self.texts: list[str] = []

    def predict_batch(
        self, items: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0) for _ in items)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        self.texts.extend(texts)
        return tuple((0.0, 1.0) for _ in texts)


class RecordingFileVectorizer:
    supported_extensions = frozenset({".example"})

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        assert embedding.vector_name == "text_test"
        self.paths.append(path)
        return ((1.0, 0.0), (0.0, 1.0))


class RecordingAudioEmbedding(AudioEmbedding):
    sampling_rate = 48_000
    max_length_seconds = 0.1
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(
            model_id="audio-test",
            vector_name="audio_test",
            dimensions=2,
        )
        self.audio_inputs: list[AudioInput] = []
        self.batches: list[tuple[AudioInput, ...]] = []

    def predict_batch(
        self,
        items: Sequence[AudioInput],
    ) -> tuple[EmbeddingVector, ...]:
        self.audio_inputs.extend(items)
        self.batches.append(tuple(items))
        return tuple((1.0, 0.0) for _ in items)

    def predict_text_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((0.0, 1.0) for _ in texts)


class RecordingAudioRangeDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, AudioRange, int]] = []

    def decode(
        self,
        path: str | Path,
        audio_range: AudioRange,
        *,
        target_sampling_rate: int,
    ) -> AudioInput:
        self.calls.append((Path(path), audio_range, target_sampling_rate))
        return AudioInput(
            samples=(audio_range.start_seconds, audio_range.end_seconds),
            sampling_rate=target_sampling_rate,
        )


class RecordingRepository:
    def __init__(
        self,
        results: Sequence[Sequence[FileSearchResult]],
    ) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def semantic_search(self, vector: Sequence[float], **kwargs: Any):
        self.calls.append({"scope": "file", "vector": tuple(vector), **kwargs})
        return tuple(self.results.pop(0))

    def semantic_segment_search(self, vector: Sequence[float], **kwargs: Any):
        self.calls.append({"scope": "segment", "vector": tuple(vector), **kwargs})
        return tuple(self.results.pop(0))


def result(
    entry_id: str,
    score: float,
    *,
    segment_index: int | None = None,
) -> FileSearchResult:
    entry = FileIndexEntry(
        id=entry_id,
        path=Path(f"{entry_id}.bin"),
        content_hash=f"hash-{entry_id}",
        file_type="example",
        metadata={},
        indexed_at=datetime.now().astimezone(),
    )
    segment = (
        FileSegmentMatch(segment_index, 0.0, 1.0)
        if segment_index is not None
        else None
    )
    return FileSearchResult(file=entry, score=score, segment=segment)


def test_search_source_requires_exactly_one_non_empty_input() -> None:
    assert SearchSource.from_text("  example query ").text == "example query"
    assert SearchSource.from_file("query.example").file == Path("query.example")

    with pytest.raises(ValueError, match="exactly one"):
        SearchSource()
    with pytest.raises(ValueError, match="exactly one"):
        SearchSource(text="query", file=Path("query.example"))
    with pytest.raises(ValueError, match="must not be empty"):
        SearchSource.from_text(" ")


def test_embed_search_source_supports_text_and_file_vectors() -> None:
    embedding = RecordingTextEmbedding()
    vectorizer = RecordingFileVectorizer()

    assert embed_search_source(
        SearchSource.from_text("orange cat"), embedding
    ) == ((0.0, 1.0),)
    assert embed_search_source(
        SearchSource.from_file("query.example"),
        embedding,
        file_vectorizer=vectorizer,
    ) == ((1.0, 0.0), (0.0, 1.0))
    assert embedding.texts == ["orange cat"]
    assert vectorizer.paths == [Path("query.example")]


def test_audio_file_query_matches_index_chunk_preparation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query.wav"
    sample_rate = 8_000
    frame_count = round(0.25 * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)
    index_embedding = RecordingAudioEmbedding()
    index_decoder = RecordingAudioRangeDecoder()
    indexed = AudioProcessor(
        embedding=index_embedding,
        decoder=index_decoder,
    ).process_batch((path,))
    query_embedding = RecordingAudioEmbedding()
    query_decoder = RecordingAudioRangeDecoder()

    vectors = AudioFileQueryVectorizer(decoder=query_decoder).predict_file(
        path,
        query_embedding,
    )

    indexed_vectors = tuple(
        segment.vectors[index_embedding.vector_name]
        for segment in indexed.files[0].segments
    )
    assert vectors == indexed_vectors == ((1.0, 0.0),) * 4
    assert index_decoder.calls == query_decoder.calls
    assert [len(batch) for batch in index_embedding.batches] == [2, 2]
    assert [len(batch) for batch in query_embedding.batches] == [2, 2]
    assert index_embedding.audio_inputs == query_embedding.audio_inputs
    assert all(
        item.sampling_rate == query_embedding.sampling_rate
        for item in query_embedding.audio_inputs
    )


def test_search_vectors_merges_multi_vector_results_by_best_score() -> None:
    repository = RecordingRepository(
        (
            (result("one", 0.7, segment_index=0), result("two", 0.8, segment_index=0)),
            (result("one", 0.9, segment_index=0), result("one", 0.6, segment_index=1)),
        )
    )

    results = search_vectors(
        repository,
        ((1.0, 0.0), (0.0, 1.0)),
        vector_name="example_vector",
        vector_scope="segment",
        limit=2,
    )

    assert [(item.file.id, item.segment.index, item.score) for item in results if item.segment] == [
        ("one", 0, 0.9),
        ("two", 0, 0.8),
    ]
    assert [call["scope"] for call in repository.calls] == ["segment", "segment"]
    assert all(call["limit"] == 2 for call in repository.calls)


@pytest.mark.parametrize(
    ("vectors", "vector_scope", "limit", "message"),
    (
        ((), "file", 10, "at least one"),
        (((1.0,),), "other", 10, "file or segment"),
        (((1.0,),), "file", 0, "positive"),
    ),
)
def test_search_vectors_rejects_invalid_requests(
    vectors: Sequence[Sequence[float]],
    vector_scope: Any,
    limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        search_vectors(
            RecordingRepository(()),
            vectors,
            vector_name="example",
            vector_scope=vector_scope,
            limit=limit,
        )
