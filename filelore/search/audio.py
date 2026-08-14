"""Audio file query adapter for chunked similarity search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from filelore.audio import (
    AudioChunkVectorizer,
    AudioDecoder,
    AudioSegmenter,
    AudioVectorizationSource,
)
from filelore.embedding import AudioEmbedding, BaseEmbedding, EmbeddingVector
from filelore.metadata import AudioMetadataParser
from filelore.search.execution import validate_query_file


class AudioFileQueryVectorizer:
    """Embed an audio file with the same chunking pipeline used for indexing."""

    supported_extensions = AudioMetadataParser.supported_extensions

    def __init__(
        self,
        *,
        metadata_parser: AudioMetadataParser | None = None,
        segmenter: AudioSegmenter | None = None,
        decoder: AudioDecoder | None = None,
        segment_batch_size: int | None = None,
    ) -> None:
        self.metadata_parser = metadata_parser or AudioMetadataParser()
        self.segmenter = segmenter
        self.decoder = decoder
        self.segment_batch_size = segment_batch_size

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        if not isinstance(embedding, AudioEmbedding):
            raise TypeError("Audio file search requires an audio embedding")
        prepared = validate_query_file(path, self.supported_extensions)
        try:
            metadata = self.metadata_parser.parse(prepared)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not prepare audio query {prepared}: {error}"
            ) from error
        source = AudioVectorizationSource(
            source_path=prepared,
            metadata=metadata,
        )
        batch = AudioChunkVectorizer(
            embedding,
            segmenter=self.segmenter,
            decoder=self.decoder,
            segment_batch_size=self.segment_batch_size,
        ).vectorize((source,))
        if batch.failures:
            failure = batch.failures[0]
            raise ValueError(
                f"Could not prepare audio query "
                f"{failure.source.source_path}: {failure.error}"
            ) from failure.error
        if len(batch.files) != 1:
            raise ValueError("Audio query produced no vectorized file")

        return tuple(segment.vector for segment in batch.files[0].segments)
