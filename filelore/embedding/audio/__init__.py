"""Audio and text embedding models."""

from filelore.embedding.audio.base import AudioEmbedding, AudioInput
from filelore.embedding.audio.clap import (
    DEFAULT_CLAP_MODEL,
    DEFAULT_CLAP_VECTOR_NAME,
    ClapAudioEmbedding,
)

__all__ = [
    "AudioEmbedding",
    "AudioInput",
    "ClapAudioEmbedding",
    "DEFAULT_CLAP_MODEL",
    "DEFAULT_CLAP_VECTOR_NAME",
]
