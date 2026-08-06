"""Model-independent embedding contracts and vector validation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar


EmbeddingInput = TypeVar("EmbeddingInput")
EmbeddingVector = tuple[float, ...]


class BaseEmbedding(ABC, Generic[EmbeddingInput]):
    """Base contract for models that map one input type into a vector space."""

    def __init__(self, *, model_id: str, vector_name: str, dimensions: int) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if not vector_name.strip():
            raise ValueError("vector_name must not be empty")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")

        self.model_id = model_id
        self.vector_name = vector_name
        self.dimensions = dimensions

    def close(self) -> None:
        """Release resources held by this embedding implementation."""

    def __enter__(self) -> BaseEmbedding[EmbeddingInput]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _clear_device_cache(self, torch_module: Any, device: str) -> None:
        """Best-effort release of the cache for the accelerator actually used."""
        device_type = device.partition(":")[0].casefold()
        if device_type == "cuda":
            backend = getattr(torch_module, "cuda", None)
            cache = getattr(backend, "empty_cache", None)
        elif device_type == "mps":
            backend = getattr(
                getattr(torch_module, "backends", None), "mps", None
            )
            cache = getattr(
                getattr(torch_module, "mps", None), "empty_cache", None
            )
        else:
            return

        is_available = getattr(backend, "is_available", None)
        if (
            not callable(is_available)
            or not is_available()
            or not callable(cache)
        ):
            return
        try:
            cache()
        except RuntimeError:
            # Cleanup must not turn completed indexing into a reported failure.
            return

    def predict(self, item: EmbeddingInput) -> EmbeddingVector:
        """Embed one item using the implementation's batch inference path."""
        vectors = self.predict_batch((item,))
        if len(vectors) != 1:
            raise ValueError("A single prediction must produce exactly one vector")
        return vectors[0]

    @abstractmethod
    def predict_batch(
        self, items: Sequence[EmbeddingInput]
    ) -> tuple[EmbeddingVector, ...]:
        """Embed a batch while preserving input order."""
        raise NotImplementedError

    def _prepare_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        expected_count: int,
        normalize: bool = False,
    ) -> tuple[EmbeddingVector, ...]:
        """Validate model output and convert it to storage-friendly vectors."""
        if len(vectors) != expected_count:
            raise ValueError(
                f"Model returned {len(vectors)} vectors for {expected_count} inputs"
            )

        prepared: list[EmbeddingVector] = []
        for vector in vectors:
            values = tuple(float(value) for value in vector)
            if len(values) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, got {len(values)}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Embedding vectors must contain only finite values")
            if normalize:
                magnitude = math.sqrt(sum(value * value for value in values))
                if magnitude == 0:
                    raise ValueError("Cannot normalize a zero-length embedding vector")
                values = tuple(value / magnitude for value in values)
            prepared.append(values)
        return tuple(prepared)
