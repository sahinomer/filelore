"""Generic Sentence Transformers implementation for text retrieval."""

from __future__ import annotations

import gc
import re
from collections.abc import Mapping
from typing import Any, Sequence

from filelore.embedding.base import EmbeddingVector
from filelore.embedding.document.base import DocumentEmbedding


def _load_sentence_transformer_backend(
    model_id: str,
    *,
    device: str,
    model_kwargs: Mapping[str, Any],
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """Load optional ML dependencies only when a text embedder is created."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "SentenceTransformerTextEmbedding requires the embedding "
            "dependencies; run 'uv sync --extra embedding' from the "
            "repository root"
        ) from error

    model = SentenceTransformer(
        model_id,
        device=None if device == "auto" else device,
        model_kwargs=dict(model_kwargs),
        trust_remote_code=trust_remote_code,
    )
    return torch, model


def _model_vector_name(model_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", model_id.casefold()).strip("_")
    return f"text_sentence_transformer_{normalized}"


class SentenceTransformerTextEmbedding(DocumentEmbedding):
    """Embed documents and queries with a configurable Sentence Transformer."""

    def __init__(
        self,
        *,
        model_id: str,
        vector_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if not device.strip():
            raise ValueError("device must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if query_prompt_name is not None and not query_prompt_name.strip():
            raise ValueError("query_prompt_name must not be empty")
        if document_prompt_name is not None and not document_prompt_name.strip():
            raise ValueError("document_prompt_name must not be empty")

        configured_model_kwargs = dict(model_kwargs or {})
        torch_module, model = _load_sentence_transformer_backend(
            model_id,
            device=device,
            model_kwargs=configured_model_kwargs,
            trust_remote_code=trust_remote_code,
        )
        dimensions = model.get_sentence_embedding_dimension()
        if not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError(
                f"Sentence Transformer model {model_id!r} has no valid "
                "embedding dimension"
            )

        self._torch = torch_module
        self._model = model
        self.device = str(model.device)
        self.batch_size = batch_size
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.model_kwargs = configured_model_kwargs
        self.trust_remote_code = trust_remote_code
        self._model.eval()

        super().__init__(
            model_id=model_id,
            vector_name=vector_name or _model_vector_name(model_id),
            dimensions=dimensions,
        )

    def predict_batch(
        self, items: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        """Embed document passages without applying the query instruction."""
        return self._encode(
            items,
            prompt_name=self.document_prompt_name,
            input_kind="Document texts",
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        """Embed retrieval queries with the model's configured query prompt."""
        return self._encode(
            texts,
            prompt_name=self.query_prompt_name,
            input_kind="Text queries",
        )

    def close(self) -> None:
        """Release the model and cached accelerator memory."""
        if self._model is None:
            return
        self._model = None
        gc.collect()
        self._clear_device_cache(self._torch, self.device)

    def _encode(
        self,
        texts: Sequence[str],
        *,
        prompt_name: str | None,
        input_kind: str,
    ) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, str):
            raise TypeError(f"{input_kind} must be provided as a sequence")
        if not texts:
            return ()
        if any(not isinstance(text, str) for text in texts):
            raise TypeError(f"Every item in {input_kind.casefold()} must be a string")
        if any(not text.strip() for text in texts):
            raise ValueError(f"{input_kind} must not be empty")
        if self._model is None:
            raise RuntimeError("The text embedding model has been closed")

        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name
        rows = self._model.encode(list(texts), **encode_kwargs)
        to_list = getattr(rows, "tolist", None)
        if callable(to_list):
            rows = to_list()
        return self._prepare_vectors(rows, expected_count=len(texts))
