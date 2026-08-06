"""Extensible embedding contracts and model implementations."""

from filelore.embedding.audio import (
    AudioEmbedding,
    AudioInput,
    ClapAudioEmbedding,
)
from filelore.embedding.base import BaseEmbedding, EmbeddingVector
from filelore.embedding.image import (
    ClipImageEmbedding,
    ImageEmbedding,
    ImageInput,
)
from filelore.embedding.text import TextEmbedding

__all__ = [
    "AudioEmbedding",
    "AudioInput",
    "BaseEmbedding",
    "ClapAudioEmbedding",
    "ClipImageEmbedding",
    "EmbeddingVector",
    "ImageEmbedding",
    "ImageInput",
    "TextEmbedding",
]
