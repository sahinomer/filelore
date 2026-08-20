"""Microsoft Harrier defaults for Sentence Transformers text retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from filelore.embedding.document.sentence_transformer import (
    SentenceTransformerTextEmbedding,
)


DEFAULT_HARRIER_MODEL = "microsoft/harrier-oss-v1-270m"
DEFAULT_HARRIER_VECTOR_NAME = "text_harrier_microsoft_oss_v1_270m"
DEFAULT_HARRIER_QUERY_PROMPT = "web_search_query"


class HarrierTextEmbedding(SentenceTransformerTextEmbedding):
    """Sentence Transformer text embeddings configured for Harrier retrieval."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_HARRIER_MODEL,
        vector_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if vector_name is None and model_id == DEFAULT_HARRIER_MODEL:
            vector_name = DEFAULT_HARRIER_VECTOR_NAME
        configured_model_kwargs = (
            {"dtype": "auto"} if model_kwargs is None else model_kwargs
        )
        super().__init__(
            model_id=model_id,
            vector_name=vector_name,
            device=device,
            batch_size=batch_size,
            query_prompt_name=DEFAULT_HARRIER_QUERY_PROMPT,
            model_kwargs=configured_model_kwargs,
        )
