"""High-level search orchestration shared by the CLI and TUI."""

from __future__ import annotations

from threading import RLock
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from filelore.embedding import BaseEmbedding
from filelore.index import IndexHandler, file_metadata_filter
from filelore.search.execution import (
    embed_search_source,
    group_segment_results,
    search_vectors,
    validate_query_file,
)
from filelore.search.models import (
    SearchRequest,
    SearchResponse,
    SearchResultGroup,
    SearchTimings,
)
from filelore.search.protocols import FileQueryVectorizer, SearchRepository
from filelore.search.query_parser import validate_search_metadata


SEGMENT_GROUP_OVERFETCH_FACTOR = 5
StageCallback = Callable[[str], None]


class SearchService:
    """Validate, embed, execute, group, and time semantic searches."""

    def __init__(
        self,
        repository: SearchRepository,
        handlers: Mapping[str, IndexHandler],
        allowed_targets: Sequence[str] | None = None,
        *,
        file_query_vectorizers: Mapping[str, FileQueryVectorizer] | None = None,
    ) -> None:
        selected_targets = tuple(
            dict.fromkeys(allowed_targets if allowed_targets is not None else handlers)
        )
        if not selected_targets:
            raise ValueError("Search requires at least one target")
        unknown = set(selected_targets).difference(handlers)
        if unknown:
            raise ValueError(f"Unsupported search target: {sorted(unknown)[0]}")
        self.repository = repository
        self.handlers = {target: handlers[target] for target in selected_targets}
        self.file_query_vectorizers = {
            target: vectorizer
            for target, vectorizer in (file_query_vectorizers or {}).items()
            if target in self.handlers
        }
        self.targets = selected_targets
        self.default_target = (
            "image" if "image" in self.handlers else selected_targets[0]
        )
        self._active_target: str | None = None
        self._embedding: BaseEmbedding[Any] | None = None
        self._lock = RLock()

    @property
    def active_target(self) -> str | None:
        return self._active_target

    def search(
        self,
        request: SearchRequest,
        limit: int,
        *,
        group_segments: bool = False,
        on_stage: StageCallback | None = None,
    ) -> SearchResponse:
        """Execute one complete search using the active target model."""
        self._validate_request(request, limit)
        with self._lock:
            total_started = perf_counter()
            initialization_started = perf_counter()
            embedding = self._activate(request.target, on_stage=on_stage)
            initialization_ms = (perf_counter() - initialization_started) * 1000

            if on_stage is not None:
                on_stage(f"Searching {request.target} files…")
            embedding_started = perf_counter()
            query_vectors = embed_search_source(
                request.source,
                embedding,
                file_vectorizer=self.file_query_vectorizers.get(request.target),
            )
            embedding_ms = (perf_counter() - embedding_started) * 1000

            handler = self.handlers[request.target]
            fetch_limit = (
                limit * SEGMENT_GROUP_OVERFETCH_FACTOR
                if group_segments and handler.vector_scope == "segment"
                else limit
            )
            fetch_started = perf_counter()
            raw_results = search_vectors(
                self.repository,
                query_vectors,
                vector_name=embedding.vector_name,
                vector_scope=handler.vector_scope,
                limit=fetch_limit,
                metadata_filter=file_metadata_filter(request.metadata_query),
            )
            fetch_ms = (perf_counter() - fetch_started) * 1000

            if group_segments and handler.vector_scope == "segment":
                results = group_segment_results(raw_results, limit=limit)
            else:
                results = tuple(
                    SearchResultGroup(result) for result in raw_results[:limit]
                )
            return SearchResponse(
                request=request,
                results=results,
                limit=limit,
                query_vector_count=len(query_vectors),
                timings=SearchTimings(
                    initialization_ms=initialization_ms,
                    embedding_ms=embedding_ms,
                    fetch_ms=fetch_ms,
                    total_ms=(perf_counter() - total_started) * 1000,
                ),
            )

    def close(self) -> None:
        """Release the currently active model, if any."""
        with self._lock:
            if self._embedding is not None:
                self._embedding.close()
            self._embedding = None
            self._active_target = None

    def __enter__(self) -> SearchService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_request(self, request: SearchRequest, limit: int) -> None:
        if request.target not in self.handlers:
            raise ValueError(f"Search target is not enabled: {request.target}")
        if limit < 1:
            raise ValueError("Search limit must be positive")
        validate_search_metadata(request.metadata_query, request.target)
        if request.source.file is None:
            return
        vectorizer = self.file_query_vectorizers.get(request.target)
        if vectorizer is None:
            raise ValueError(
                f"File similarity search is not enabled for {request.target}"
            )
        validate_query_file(request.source.file, vectorizer.supported_extensions)

    def _activate(
        self,
        target: str,
        *,
        on_stage: StageCallback | None,
    ) -> BaseEmbedding[Any]:
        if self._embedding is not None and self._active_target == target:
            return self._embedding
        if self._embedding is not None:
            self._embedding.close()
            self._embedding = None
            self._active_target = None
        if on_stage is not None:
            on_stage(f"Initializing {target} model…")
        embedding = self.handlers[target].embedding_factory()
        self._embedding = embedding
        self._active_target = target
        return embedding
