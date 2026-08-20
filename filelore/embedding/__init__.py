"""Extensible embedding contracts and model implementations."""

from filelore.embedding.audio import (
    AudioEmbedding,
    AudioInput,
    ClapAudioEmbedding,
)
from filelore.embedding.base import BaseEmbedding, EmbeddingVector
from filelore.embedding.document import (
    DEFAULT_HARRIER_MODEL,
    DEFAULT_HARRIER_QUERY_PROMPT,
    DEFAULT_HARRIER_VECTOR_NAME,
    HarrierTextEmbedding,
    SentenceTransformerTextEmbedding,
)
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
    "DEFAULT_HARRIER_MODEL",
    "DEFAULT_HARRIER_QUERY_PROMPT",
    "DEFAULT_HARRIER_VECTOR_NAME",
    "EmbeddingVector",
    "HarrierTextEmbedding",
    "ImageEmbedding",
    "ImageInput",
    "SentenceTransformerTextEmbedding",
    "TextEmbedding",
]
