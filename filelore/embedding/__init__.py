"""Extensible embedding contracts and model implementations."""

from filelore.embedding.base import BaseEmbedding, EmbeddingVector
from filelore.embedding.image import (
    ClipImageEmbedding,
    ImageEmbedding,
    ImageInput,
)

__all__ = [
    "BaseEmbedding",
    "ClipImageEmbedding",
    "EmbeddingVector",
    "ImageEmbedding",
    "ImageInput",
]
