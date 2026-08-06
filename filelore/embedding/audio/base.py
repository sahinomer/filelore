"""Contracts shared by audio-text embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from filelore.embedding.text import TextEmbedding


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Decoded mono samples and their sampling rate."""

    samples: Sequence[float]
    sampling_rate: int


class AudioEmbedding(TextEmbedding[AudioInput]):
    """Embed decoded audio and text into the same comparable vector space."""

    sampling_rate: int
    max_length_seconds: float
