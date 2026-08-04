"""Contracts shared by image-text embedding models."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TypeAlias, Sequence

from PIL import Image

from filelore.embedding.base import BaseEmbedding, EmbeddingVector


ImageInput: TypeAlias = str | Path | Image.Image


class ImageEmbedding(BaseEmbedding[ImageInput]):
    """Embed images and text into the same comparable vector space."""

    def predict_text(self, text: str) -> EmbeddingVector:
        """Embed one semantic text query."""
        vectors = self.predict_text_batch((text,))
        if len(vectors) != 1:
            raise ValueError("A single text prediction must produce exactly one vector")
        return vectors[0]

    @abstractmethod
    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        """Embed text queries while preserving input order."""
        raise NotImplementedError
