"""Low-level query embedding, vector search, and result grouping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

from filelore.embedding import BaseEmbedding, EmbeddingVector, TextEmbedding
from filelore.index import FileSearchResult
from filelore.search.models import SearchResultGroup, SearchSource, VectorScope
from filelore.search.protocols import FileQueryVectorizer, SearchRepository
from filelore.storage import MetadataFilter


def validate_query_file(
    path: str | Path,
    supported_extensions: Collection[str],
) -> Path:
    """Resolve a query file after validating its existence and extension."""
    prepared = Path(path).expanduser()
    if not prepared.is_file():
        raise ValueError(f"Query file does not exist: {prepared}")
    normalized_extensions = {
        extension.casefold()
        if extension.startswith(".")
        else f".{extension.casefold()}"
        for extension in supported_extensions
    }
    if prepared.suffix.casefold() not in normalized_extensions:
        extension = prepared.suffix or "<none>"
        raise ValueError(f"Unsupported query file extension: {extension}")
    return prepared.resolve()


def embed_search_source(
    source: SearchSource,
    embedding: BaseEmbedding[Any],
    *,
    file_vectorizer: FileQueryVectorizer | None = None,
) -> tuple[EmbeddingVector, ...]:
    """Embed text or delegate a file query to its modality adapter."""
    if source.text is not None:
        if not isinstance(embedding, TextEmbedding):
            raise TypeError("Text search requires a text embedding")
        return (embedding.predict_text(source.text),)

    if file_vectorizer is None:
        raise ValueError("File similarity search is not enabled for this target")
    assert source.file is not None
    vectors = file_vectorizer.predict_file(source.file, embedding)
    if not vectors:
        raise ValueError("Query file produced no embedding vectors")
    return vectors


def search_vectors(
    repository: SearchRepository,
    vectors: Sequence[Sequence[float]],
    *,
    vector_name: str,
    vector_scope: VectorScope,
    limit: int,
    metadata_filter: MetadataFilter | None = None,
) -> tuple[FileSearchResult, ...]:
    """Search one or many vectors, retaining each result's best score."""
    if not vectors:
        raise ValueError("Similarity search requires at least one query vector")
    if not vector_name.strip():
        raise ValueError("Similarity search requires a vector name")
    if vector_scope not in {"file", "segment"}:
        raise ValueError("Vector scope must be file or segment")
    if limit < 1:
        raise ValueError("Search limit must be positive")

    search = {
        "file": repository.semantic_search,
        "segment": repository.semantic_segment_search,
    }[vector_scope]
    merged: dict[tuple[str, int | None], FileSearchResult] = {}
    for vector in vectors:
        results = search(
            vector,
            vector_name=vector_name,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        for result in results:
            key = (
                result.file.id,
                result.segment.index if result.segment is not None else None,
            )
            previous = merged.get(key)
            if previous is None or result.score > previous.score:
                merged[key] = result

    return tuple(
        sorted(merged.values(), key=lambda result: result.score, reverse=True)[
            :limit
        ]
    )


def group_segment_results(
    results: Sequence[FileSearchResult],
    *,
    limit: int,
) -> tuple[SearchResultGroup, ...]:
    """Group segment matches by parent file and retain best-first matches."""
    groups: dict[str, list[FileSearchResult]] = {}
    for result in results:
        groups.setdefault(result.file.id, []).append(result)

    entities: list[SearchResultGroup] = []
    for matches in groups.values():
        ranked = tuple(sorted(matches, key=lambda item: item.score, reverse=True))
        entities.append(SearchResultGroup(result=ranked[0], matches=ranked))
    entities.sort(key=lambda item: item.result.score, reverse=True)
    return tuple(entities[:limit])
