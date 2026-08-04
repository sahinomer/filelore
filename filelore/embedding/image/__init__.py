"""Image and multimodal embedding models."""

from filelore.embedding.image.base import ImageEmbedding, ImageInput
from filelore.embedding.image.clip import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_VECTOR_NAME,
    ClipImageEmbedding,
)

__all__ = [
    "ClipImageEmbedding",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_VECTOR_NAME",
    "ImageEmbedding",
    "ImageInput",
]
