"""Audio metadata, segmentation, decoding, and embedding processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from filelore.audio import (
    AudioDecoder,
    AudioRange,
    AudioSegmenter,
    SlidingWindowChunker,
    SoundFileAudioDecoder,
)
from filelore.embedding import AudioEmbedding, AudioInput
from filelore.metadata import AudioMetadata, AudioMetadataParser
from filelore.processors.models import (
    PreparedFile,
    PreparedSegment,
    ProcessingBatch,
    ProcessingFailure,
)


@dataclass(frozen=True, slots=True)
class _ParsedAudio:
    source_path: Path
    metadata: AudioMetadata


@dataclass(frozen=True, slots=True)
class _PlannedSegment:
    file_index: int
    index: int
    audio_range: AudioRange


class AudioProcessor:
    """Prepare audio metadata and optional timed vectors for indexing."""

    def __init__(
        self,
        *,
        metadata_parser: AudioMetadataParser | None = None,
        embedding: AudioEmbedding | None = None,
        segmenter: AudioSegmenter | None = None,
        decoder: AudioDecoder | None = None,
        segment_batch_size: int | None = None,
    ) -> None:
        if segment_batch_size is not None and segment_batch_size < 1:
            raise ValueError("Audio segment batch size must be positive")

        self.metadata_parser = metadata_parser or AudioMetadataParser()
        self.embedding = embedding
        self.segmenter = segmenter or self._default_segmenter(embedding)
        self.decoder = decoder
        self.segment_batch_size = segment_batch_size or self._model_batch_size(
            embedding
        )

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]:
        """Yield audio files supported by the metadata parser."""
        return self.metadata_parser.discover(directory, recursive=recursive)

    def process_batch(
        self, paths: Sequence[str | Path]
    ) -> ProcessingBatch[AudioMetadata]:
        """Prepare files while decoding and embedding bounded segment batches."""
        parsed, failures = self._parse_files(paths)
        if not parsed:
            return ProcessingBatch(files=(), failures=tuple(failures))
        if self.embedding is None:
            return ProcessingBatch(
                files=tuple(
                    PreparedFile(metadata=item.metadata, vectors={})
                    for item in parsed
                ),
                failures=tuple(failures),
            )

        planned, failed_files = self._plan_segments(parsed, failures)
        segments_by_file: dict[int, list[PreparedSegment]] = {}
        decoder = self.decoder or SoundFileAudioDecoder()

        for batch in _batches(planned, self.segment_batch_size):
            decoded: list[tuple[_PlannedSegment, AudioInput]] = []
            for segment in batch:
                if segment.file_index in failed_files:
                    continue
                item = parsed[segment.file_index]
                try:
                    audio_input = decoder.decode(
                        item.metadata.path,
                        segment.audio_range,
                        target_sampling_rate=self.embedding.sampling_rate,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append(
                        ProcessingFailure(path=item.source_path, error=error)
                    )
                    failed_files.add(segment.file_index)
                    segments_by_file.pop(segment.file_index, None)
                    continue
                decoded.append((segment, audio_input))

            decoded = [
                item
                for item in decoded
                if item[0].file_index not in failed_files
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
                    PreparedSegment(
                        index=segment.index,
                        start_seconds=segment.audio_range.start_seconds,
                        end_seconds=segment.audio_range.end_seconds,
                        vectors={self.embedding.vector_name: vector},
                    )
                )

        files = tuple(
            PreparedFile(
                metadata=item.metadata,
                vectors={},
                segments=tuple(segments_by_file.get(file_index, ())),
            )
            for file_index, item in enumerate(parsed)
            if file_index not in failed_files
        )
        return ProcessingBatch(files=files, failures=tuple(failures))

    def _parse_files(
        self, paths: Sequence[str | Path]
    ) -> tuple[list[_ParsedAudio], list[ProcessingFailure]]:
        parsed: list[_ParsedAudio] = []
        failures: list[ProcessingFailure] = []
        for path in paths:
            source_path = Path(path).expanduser()
            try:
                metadata = self.metadata_parser.parse(source_path)
            except (OSError, ValueError) as error:
                failures.append(ProcessingFailure(path=source_path, error=error))
                continue
            parsed.append(_ParsedAudio(source_path=source_path, metadata=metadata))
        return parsed, failures

    def _plan_segments(
        self,
        parsed: Sequence[_ParsedAudio],
        failures: list[ProcessingFailure],
    ) -> tuple[list[_PlannedSegment], set[int]]:
        planned: list[_PlannedSegment] = []
        failed_files: set[int] = set()
        for file_index, item in enumerate(parsed):
            try:
                ranges = self.segmenter.segments(item.metadata)
                if not ranges:
                    raise ValueError("Audio file has no embeddable duration")
            except ValueError as error:
                failures.append(
                    ProcessingFailure(path=item.source_path, error=error)
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
        return planned, failed_files

    @staticmethod
    def _default_segmenter(
        embedding: AudioEmbedding | None,
    ) -> SlidingWindowChunker:
        window_seconds = embedding.max_length_seconds if embedding else 10.0
        return SlidingWindowChunker(
            window_seconds=window_seconds,
            hop_seconds=window_seconds / 2,
        )

    @staticmethod
    def _model_batch_size(embedding: AudioEmbedding | None) -> int:
        batch_size = getattr(embedding, "batch_size", 32)
        return batch_size if isinstance(batch_size, int) and batch_size > 0 else 32


def _batches(
    items: Sequence[_PlannedSegment], batch_size: int
) -> Iterator[Sequence[_PlannedSegment]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
