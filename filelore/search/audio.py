"""Audio file query adapter for chunked similarity search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from filelore.embedding import AudioEmbedding, BaseEmbedding, EmbeddingVector
from filelore.metadata import AudioMetadataParser
from filelore.processors import AudioProcessor
from filelore.search.core import validate_query_file


class AudioFileQueryVectorizer:
    """Embed an audio file with the same chunking pipeline used for indexing."""

    supported_extensions = AudioMetadataParser.supported_extensions

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        if not isinstance(embedding, AudioEmbedding):
            raise TypeError("Audio file search requires an audio embedding")
        prepared = validate_query_file(path, self.supported_extensions)
        batch = AudioProcessor(embedding=embedding).process_batch((prepared,))
        if batch.failures:
            failure = batch.failures[0]
            raise ValueError(
                f"Could not prepare audio query {failure.path}: {failure.error}"
            ) from failure.error
        if len(batch.files) != 1:
            raise ValueError("Audio query produced no processed file")

        vectors: list[EmbeddingVector] = []
        for segment in batch.files[0].segments:
            vector = segment.vectors.get(embedding.vector_name)
            if vector is None:
                raise ValueError(
                    "Audio query segment is missing the active model vector"
                )
            vectors.append(vector)
        return tuple(vectors)
