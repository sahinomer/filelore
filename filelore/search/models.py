"""Public request, response, and result models for semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from filelore.embedding import BaseEmbedding
from filelore.index import FileMetadataQuery, FileSearchResult


VectorScope = Literal["file", "segment"]
EmbeddingFactory = Callable[[], BaseEmbedding[Any]]


@dataclass(frozen=True, slots=True)
class SearchTarget:
    """Model factory and vector scope needed to search one file type."""

    embedding_factory: EmbeddingFactory
    vector_scope: VectorScope

    def __post_init__(self) -> None:
        if not callable(self.embedding_factory):
            raise TypeError("Search target embedding factory must be callable")
        if self.vector_scope not in {"file", "segment"}:
            raise ValueError("Search target vector scope must be file or segment")


@dataclass(frozen=True, slots=True)
class SearchSource:
    """Exactly one semantic input used to create comparable query vectors."""

    text: str | None = None
    file: Path | None = None

    def __post_init__(self) -> None:
        has_text = self.text is not None
        has_file = self.file is not None
        if has_text == has_file:
            raise ValueError("Search source requires exactly one of text or file")
        if self.text is not None and not self.text.strip():
            raise ValueError("Search text must not be empty")

    @classmethod
    def from_text(cls, value: str) -> SearchSource:
        return cls(text=value.strip())

    @classmethod
    def from_file(cls, value: str | Path) -> SearchSource:
        return cls(file=Path(value).expanduser())

    @property
    def is_file(self) -> bool:
        return self.file is not None

    @property
    def display_value(self) -> str:
        if self.text is not None:
            return self.text
        assert self.file is not None
        return self.file.name


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Validated semantic source, target, and result metadata constraints."""

    source: SearchSource
    target: str
    metadata_query: FileMetadataQuery = field(default_factory=FileMetadataQuery)
    filters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Search target must not be empty")


@dataclass(frozen=True, slots=True)
class SearchResultGroup:
    """One visible result and any child segment matches grouped beneath it."""

    result: FileSearchResult
    matches: tuple[FileSearchResult, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchTimings:
    initialization_ms: float
    embedding_ms: float
    fetch_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class SearchResponse:
    request: SearchRequest
    results: tuple[SearchResultGroup, ...]
    limit: int
    query_vector_count: int
    timings: SearchTimings

    @property
    def grouped_match_count(self) -> int:
        return sum(len(item.matches) for item in self.results if item.matches)

    @property
    def grouped_file_count(self) -> int:
        return sum(bool(item.matches) for item in self.results)
