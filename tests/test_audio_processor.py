from __future__ import annotations

import wave
from pathlib import Path
from typing import Sequence

import pytest

from filelore.audio import (
    AudioRange,
    SlidingWindowChunker,
    SoundFileAudioDecoder,
    WholeFileSegmenter,
)
from filelore.embedding import AudioEmbedding, AudioInput, EmbeddingVector
from filelore.metadata import AudioMetadataParser
from filelore.processors import AudioProcessor


def create_wave(
    path: Path,
    *,
    duration_seconds: float,
    sample_rate: int = 100,
    channels: int = 1,
) -> None:
    frame_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count * channels)


class RecordingAudioEmbedding(AudioEmbedding):
    sampling_rate = 48_000
    max_length_seconds = 10.0
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(
            model_id="test-audio-model",
            vector_name="audio_test",
            dimensions=3,
        )
        self.batches: list[tuple[AudioInput, ...]] = []

    def predict_batch(
        self, items: Sequence[AudioInput]
    ) -> tuple[EmbeddingVector, ...]:
        self.batches.append(tuple(items))
        return tuple(
            (float(item.samples[0]), 0.0, 0.0) for item in items
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)


class RecordingAudioDecoder:
    def __init__(
        self,
        *,
        fail_path: Path | None = None,
        fail_from_seconds: float = 0.0,
    ) -> None:
        self.fail_path = fail_path
        self.fail_from_seconds = fail_from_seconds
        self.calls: list[tuple[Path, AudioRange, int]] = []

    def decode(
        self,
        path: str | Path,
        audio_range: AudioRange,
        *,
        target_sampling_rate: int,
    ) -> AudioInput:
        prepared_path = Path(path)
        self.calls.append((prepared_path, audio_range, target_sampling_rate))
        if (
            self.fail_path is not None
            and prepared_path == self.fail_path.resolve()
            and audio_range.start_seconds >= self.fail_from_seconds
        ):
            raise ValueError("test decode failure")
        return AudioInput(
            samples=(audio_range.start_seconds + 1.0, 0.0, 0.0),
            sampling_rate=target_sampling_rate,
        )


def test_audio_segmenters_support_chunked_and_whole_file_inputs(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "long.wav"
    create_wave(audio_path, duration_seconds=23.0)
    metadata = AudioMetadataParser().parse(audio_path)

    chunked = SlidingWindowChunker(
        window_seconds=10.0,
        hop_seconds=5.0,
    ).segments(metadata)
    unchunked = WholeFileSegmenter().segments(metadata)

    assert [(item.start_seconds, item.end_seconds) for item in chunked] == [
        (0.0, 10.0),
        (5.0, 15.0),
        (10.0, 20.0),
        (15.0, 23.0),
    ]
    assert unchunked == (AudioRange(0.0, 23.0),)


@pytest.mark.parametrize(
    ("window_seconds", "hop_seconds"),
    ((0.0, 1.0), (10.0, 0.0), (5.0, 6.0)),
)
def test_audio_chunker_rejects_invalid_configuration(
    window_seconds: float,
    hop_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SlidingWindowChunker(
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )


def test_soundfile_decoder_reads_range_downmixes_and_resamples(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "stereo.wav"
    create_wave(
        audio_path,
        duration_seconds=1.0,
        sample_rate=8_000,
        channels=2,
    )

    decoded = SoundFileAudioDecoder().decode(
        audio_path,
        AudioRange(0.25, 0.75),
        target_sampling_rate=16_000,
    )

    assert decoded.sampling_rate == 16_000
    assert getattr(decoded.samples, "ndim") == 1
    assert getattr(decoded.samples, "dtype").name == "float32"
    assert len(decoded.samples) == 8_000


def test_audio_processor_chunks_batches_and_regroups_segments(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "long.wav"
    create_wave(audio_path, duration_seconds=23.0)
    embedding = RecordingAudioEmbedding()
    decoder = RecordingAudioDecoder()

    batch = AudioProcessor(
        embedding=embedding,
        decoder=decoder,
    ).process_batch((audio_path,))

    assert batch.failures == ()
    assert len(batch.files) == 1
    assert batch.files[0].vectors == {}
    assert [
        (segment.index, segment.start_seconds, segment.end_seconds)
        for segment in batch.files[0].segments
    ] == [
        (0, 0.0, 10.0),
        (1, 5.0, 15.0),
        (2, 10.0, 20.0),
        (3, 15.0, 23.0),
    ]
    assert [segment.vectors for segment in batch.files[0].segments] == [
        {"audio_test": (1.0, 0.0, 0.0)},
        {"audio_test": (6.0, 0.0, 0.0)},
        {"audio_test": (11.0, 0.0, 0.0)},
        {"audio_test": (16.0, 0.0, 0.0)},
    ]
    assert [len(items) for items in embedding.batches] == [2, 2]
    assert all(call[2] == 48_000 for call in decoder.calls)


def test_audio_processor_accepts_an_unchunked_strategy(tmp_path: Path) -> None:
    audio_path = tmp_path / "long.wav"
    create_wave(audio_path, duration_seconds=23.0)

    batch = AudioProcessor(
        embedding=RecordingAudioEmbedding(),
        segmenter=WholeFileSegmenter(),
        decoder=RecordingAudioDecoder(),
    ).process_batch((audio_path,))

    assert batch.failures == ()
    assert [
        (segment.start_seconds, segment.end_seconds)
        for segment in batch.files[0].segments
    ] == [(0.0, 23.0)]


def test_audio_processor_can_prepare_metadata_without_decoding(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "short.wav"
    create_wave(audio_path, duration_seconds=1.0)
    decoder = RecordingAudioDecoder()

    batch = AudioProcessor(decoder=decoder).process_batch((audio_path,))

    assert len(batch.files) == 1
    assert batch.files[0].vectors == {}
    assert batch.files[0].segments == ()
    assert batch.failures == ()
    assert decoder.calls == []


def test_audio_processor_discards_partial_chunks_after_a_decode_failure(
    tmp_path: Path,
) -> None:
    failing_path = tmp_path / "failing.wav"
    good_path = tmp_path / "good.wav"
    create_wave(failing_path, duration_seconds=12.0)
    create_wave(good_path, duration_seconds=1.0)
    decoder = RecordingAudioDecoder(
        fail_path=failing_path,
        fail_from_seconds=5.0,
    )

    batch = AudioProcessor(
        embedding=RecordingAudioEmbedding(),
        decoder=decoder,
        segment_batch_size=1,
    ).process_batch((failing_path, good_path))

    assert [item.metadata.path for item in batch.files] == [good_path.resolve()]
    assert len(batch.files[0].segments) == 1
    assert len(batch.failures) == 1
    assert batch.failures[0].path == failing_path


def test_audio_processor_does_not_decode_or_embed_an_empty_batch() -> None:
    embedding = RecordingAudioEmbedding()
    decoder = RecordingAudioDecoder()

    batch = AudioProcessor(
        embedding=embedding,
        decoder=decoder,
    ).process_batch(())

    assert batch.files == ()
    assert batch.failures == ()
    assert embedding.batches == []
    assert decoder.calls == []
