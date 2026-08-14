"""Extension contracts consumed by the search package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Protocol, Sequence

from filelore.embedding import BaseEmbedding, EmbeddingVector
from filelore.index import FileSearchResult
from filelore.storage import MetadataFilter


class FileQueryVectorizer(Protocol):
    """Convert one supported query file into one or more search vectors."""

    supported_extensions: Collection[str]

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]: ...


class SearchRepository(Protocol):
    """Repository operations required by the shared search functions."""

    def semantic_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int,
        metadata_filter: MetadataFilter | None,
    ) -> tuple[FileSearchResult, ...]: ...

    def semantic_segment_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int,
        metadata_filter: MetadataFilter | None,
    ) -> tuple[FileSearchResult, ...]: ...
