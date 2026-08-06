"""Contracts shared by embeddings with a comparable text space."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Sequence, TypeVar

from filelore.embedding.base import BaseEmbedding, EmbeddingVector


EmbeddingInput = TypeVar("EmbeddingInput")


class TextEmbedding(BaseEmbedding[EmbeddingInput], Generic[EmbeddingInput]):
    """Embed one input modality and text into a shared vector space."""

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
