"""Shared model-ready audio chunk preparation and vectorization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from filelore.audio.decoding import AudioDecoder, SoundFileAudioDecoder
from filelore.audio.models import AudioRange
from filelore.audio.segmenting import AudioSegmenter, SlidingWindowChunker
from filelore.embedding import AudioEmbedding, AudioInput, EmbeddingVector
from filelore.metadata import AudioMetadata


@dataclass(frozen=True, slots=True)
class AudioVectorizationSource:
    """Parsed audio file supplied to the shared vectorization pipeline."""

    source_path: Path
    metadata: AudioMetadata


@dataclass(frozen=True, slots=True)
class AudioVectorizedSegment:
    """One timed audio chunk and its active-model vector."""

    index: int
    audio_range: AudioRange
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class AudioVectorizedFile:
    """Successfully vectorized chunks for one audio source."""

    source: AudioVectorizationSource
    segments: tuple[AudioVectorizedSegment, ...]


@dataclass(frozen=True, slots=True)
class AudioVectorizationFailure:
    """An audio source that could not produce a complete chunk set."""

    source: AudioVectorizationSource
    error: Exception


@dataclass(frozen=True, slots=True)
class AudioVectorizationBatch:
    """Successful and failed files from shared audio vectorization."""

    files: tuple[AudioVectorizedFile, ...]
    failures: tuple[AudioVectorizationFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class _PlannedSegment:
    file_index: int
    index: int
    audio_range: AudioRange


class AudioChunkVectorizer:
    """Plan, decode, resample, batch, and embed audio chunks consistently."""

    def __init__(
        self,
        embedding: AudioEmbedding,
        *,
        segmenter: AudioSegmenter | None = None,
        decoder: AudioDecoder | None = None,
        segment_batch_size: int | None = None,
    ) -> None:
        if segment_batch_size is not None and segment_batch_size < 1:
            raise ValueError("Audio segment batch size must be positive")
        self.embedding = embedding
        self.segmenter = segmenter or self._default_segmenter(embedding)
        self.decoder = decoder
        self.segment_batch_size = segment_batch_size or self._model_batch_size(
            embedding
        )

    def vectorize(
        self,
        sources: Sequence[AudioVectorizationSource],
    ) -> AudioVectorizationBatch:
        """Vectorize complete sources while discarding partial failed files."""
        if not sources:
            return AudioVectorizationBatch(files=())

        planned, failures, failed_files = self._plan_segments(sources)
        segments_by_file: dict[int, list[AudioVectorizedSegment]] = {}
        decoder = self.decoder or SoundFileAudioDecoder()

        for batch in _batches(planned, self.segment_batch_size):
            decoded: list[tuple[_PlannedSegment, AudioInput]] = []
            for segment in batch:
                if segment.file_index in failed_files:
                    continue
                source = sources[segment.file_index]
                try:
                    audio_input = decoder.decode(
                        source.metadata.path,
                        segment.audio_range,
                        target_sampling_rate=self.embedding.sampling_rate,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append(
                        AudioVectorizationFailure(source=source, error=error)
                    )
                    failed_files.add(segment.file_index)
                    segments_by_file.pop(segment.file_index, None)
                    continue
                decoded.append((segment, audio_input))

            decoded = [
                item for item in decoded if item[0].file_index not in failed_files
            ]
            if not decoded:
                continue

            vectors = self.embedding.predict_batch(
                tuple(audio_input for _, audio_input in decoded)
            )
            if len(vectors) != len(decoded):
                raise ValueError(
                    "Audio embedding count must match decoded segment count"
                )
            for (segment, _), vector in zip(decoded, vectors):
                segments_by_file.setdefault(segment.file_index, []).append(
                    AudioVectorizedSegment(
                        index=segment.index,
                        audio_range=segment.audio_range,
                        vector=vector,
                    )
                )

        files = tuple(
            AudioVectorizedFile(
                source=source,
                segments=tuple(segments_by_file.get(file_index, ())),
            )
            for file_index, source in enumerate(sources)
            if file_index not in failed_files
        )
        return AudioVectorizationBatch(files=files, failures=tuple(failures))

    def _plan_segments(
        self,
        sources: Sequence[AudioVectorizationSource],
    ) -> tuple[
        list[_PlannedSegment],
        list[AudioVectorizationFailure],
        set[int],
    ]:
        planned: list[_PlannedSegment] = []
        failures: list[AudioVectorizationFailure] = []
        failed_files: set[int] = set()
        for file_index, source in enumerate(sources):
            try:
                ranges = self.segmenter.segments(source.metadata)
                if not ranges:
                    raise ValueError("Audio file has no embeddable duration")
            except ValueError as error:
                failures.append(
                    AudioVectorizationFailure(source=source, error=error)
                )
                failed_files.add(file_index)
                continue
            planned.extend(
                _PlannedSegment(
                    file_index=file_index,
                    index=segment_index,
                    audio_range=audio_range,
                )
                for segment_index, audio_range in enumerate(ranges)
            )
        return planned, failures, failed_files

    @staticmethod
    def _default_segmenter(embedding: AudioEmbedding) -> SlidingWindowChunker:
        window_seconds = embedding.max_length_seconds
        return SlidingWindowChunker(
            window_seconds=window_seconds,
            hop_seconds=window_seconds / 2,
        )

    @staticmethod
    def _model_batch_size(embedding: AudioEmbedding) -> int:
        batch_size = getattr(embedding, "batch_size", 32)
        return batch_size if isinstance(batch_size, int) and batch_size > 0 else 32


def _batches(
    items: Sequence[_PlannedSegment],
    batch_size: int,
) -> Iterator[Sequence[_PlannedSegment]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
