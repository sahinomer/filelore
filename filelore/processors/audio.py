"""Indexing adapter for parsed and vectorized audio files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from filelore.audio import (
    AudioChunkVectorizer,
    AudioDecoder,
    AudioSegmenter,
    AudioVectorizationSource,
)
from filelore.embedding import AudioEmbedding
from filelore.metadata import AudioMetadata, AudioMetadataParser
from filelore.processors.models import (
    PreparedFile,
    PreparedSegment,
    ProcessingBatch,
    ProcessingFailure,
)


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
        self.chunk_vectorizer = (
            AudioChunkVectorizer(
                embedding,
                segmenter=segmenter,
                decoder=decoder,
                segment_batch_size=segment_batch_size,
            )
            if embedding is not None
            else None
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

        assert self.chunk_vectorizer is not None
        vectorized = self.chunk_vectorizer.vectorize(parsed)
        failures.extend(
            ProcessingFailure(
                path=failure.source.source_path,
                error=failure.error,
            )
            for failure in vectorized.failures
        )
        files = tuple(
            PreparedFile(
                metadata=item.source.metadata,
                vectors={},
                segments=tuple(
                    PreparedSegment(
                        index=segment.index,
                        start_seconds=segment.audio_range.start_seconds,
                        end_seconds=segment.audio_range.end_seconds,
                        vectors={self.embedding.vector_name: segment.vector},
                    )
                    for segment in item.segments
                ),
            )
            for item in vectorized.files
        )
        return ProcessingBatch(files=files, failures=tuple(failures))

    def _parse_files(
        self, paths: Sequence[str | Path]
    ) -> tuple[list[AudioVectorizationSource], list[ProcessingFailure]]:
        parsed: list[AudioVectorizationSource] = []
        failures: list[ProcessingFailure] = []
        for path in paths:
            source_path = Path(path).expanduser()
            try:
                metadata = self.metadata_parser.parse(source_path)
            except (OSError, ValueError) as error:
                failures.append(ProcessingFailure(path=source_path, error=error))
                continue
            parsed.append(
                AudioVectorizationSource(
                    source_path=source_path,
                    metadata=metadata,
                )
            )
        return parsed, failures
