"""Image file query adapter for similarity search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from filelore.embedding import BaseEmbedding, EmbeddingVector, ImageEmbedding
from filelore.metadata import ImageMetadataParser
from filelore.search.core import validate_query_file


class ImageFileQueryVectorizer:
    """Embed one image file in the indexed image vector space."""

    supported_extensions = ImageMetadataParser.supported_extensions

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        if not isinstance(embedding, ImageEmbedding):
            raise TypeError("Image file search requires an image embedding")
        prepared = validate_query_file(path, self.supported_extensions)
        return (embedding.predict(prepared),)
