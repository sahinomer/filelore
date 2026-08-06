"""Orchestration between file processors and the persistent index."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)

from filelore.embedding import BaseEmbedding
from filelore.index.models import FileIndexEntry
from filelore.index.repository import FileIndexRepository
from filelore.metadata import BaseMetadata
from filelore.processors.models import ProcessingBatch, ProcessingFailure
from filelore.storage import DistanceMetric, VectorConfig, VectorDatabase


MetadataType = TypeVar("MetadataType", bound=BaseMetadata)


class FileProcessor(Protocol[MetadataType]):
    """Processor operations needed by the indexing pipeline."""

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]: ...

    def process_batch(
        self, paths: Sequence[str | Path]
    ) -> ProcessingBatch[MetadataType]: ...


@dataclass(frozen=True, slots=True)
class IndexingBatch:
    """Entries stored and files rejected during one indexing batch."""

    entries: tuple[FileIndexEntry, ...]
    failures: tuple[ProcessingFailure, ...] = ()


class FileIndexer(Generic[MetadataType]):
    """Run a format-specific processor and persist its successful output."""

    def __init__(
        self,
        repository: FileIndexRepository,
        processor: FileProcessor[MetadataType],
    ) -> None:
        self.repository = repository
        self.processor = processor

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]:
        return self.processor.discover(directory, recursive=recursive)

    def index_batch(self, paths: Sequence[str | Path]) -> IndexingBatch:
        processed = self.processor.process_batch(paths)
        entries = self.repository.store_prepared_many(processed.files)
        return IndexingBatch(entries=entries, failures=processed.failures)


VectorScope = Literal["file", "segment"]


@dataclass(frozen=True, slots=True)
class IndexHandler:
    """Lazy model and processor registration for one supported file type."""

    file_type: str
    extensions: frozenset[str]
    embedding_factory: Callable[[], BaseEmbedding[Any]]
    processor_factory: Callable[[BaseEmbedding[Any]], FileProcessor[Any]]
    vector_scope: VectorScope

    def __post_init__(self) -> None:
        file_type = self.file_type.strip().casefold()
        if not file_type:
            raise ValueError("Index handler file type must not be empty")
        if self.vector_scope not in {"file", "segment"}:
            raise ValueError("Index handler vector scope must be file or segment")
        normalized_extensions = frozenset(
            extension.casefold()
            if extension.startswith(".")
            else f".{extension.casefold()}"
            for extension in self.extensions
        )
        if not normalized_extensions:
            raise ValueError("Index handler must support at least one extension")
        object.__setattr__(self, "file_type", file_type)
        object.__setattr__(self, "extensions", normalized_extensions)

    @contextmanager
    def open_indexer(
        self, database: VectorDatabase
    ) -> Iterator[FileIndexer[Any]]:
        """Load one model and guarantee it is released after its queue."""
        embedding = self.embedding_factory()
        try:
            vector_configs = {
                embedding.vector_name: VectorConfig(
                    embedding.dimensions,
                    distance=DistanceMetric.COSINE,
                )
            }
            repository_arguments = (
                {"vector_configs": vector_configs}
                if self.vector_scope == "file"
                else {"segment_vector_configs": vector_configs}
            )
            repository = FileIndexRepository(database, **repository_arguments)
            yield FileIndexer(repository, self.processor_factory(embedding))
        finally:
            embedding.close()


@dataclass(frozen=True, slots=True)
class IndexQueue:
    """Paths of one type that can share a loaded model."""

    handler: IndexHandler
    paths: tuple[Path, ...]

    @property
    def file_type(self) -> str:
        return self.handler.file_type


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """Model-free result of classifying a directory once."""

    queues: tuple[IndexQueue, ...]

    @property
    def total_files(self) -> int:
        return sum(len(queue.paths) for queue in self.queues)


class IndexCoordinator:
    """Discover supported files and arrange homogeneous model queues."""

    def __init__(self, handlers: Sequence[IndexHandler]) -> None:
        if not handlers:
            raise ValueError("At least one index handler is required")
        self.handlers = tuple(handlers)
        self._handlers_by_type: dict[str, IndexHandler] = {}
        self._handlers_by_extension: dict[str, IndexHandler] = {}
        for handler in self.handlers:
            if handler.file_type in self._handlers_by_type:
                raise ValueError(
                    f"Duplicate index handler type: {handler.file_type}"
                )
            self._handlers_by_type[handler.file_type] = handler
            for extension in handler.extensions:
                existing = self._handlers_by_extension.get(extension)
                if existing is not None:
                    raise ValueError(
                        f"Extension {extension} is registered for both "
                        f"{existing.file_type} and {handler.file_type}"
                    )
                self._handlers_by_extension[extension] = handler

    @property
    def file_types(self) -> tuple[str, ...]:
        return tuple(handler.file_type for handler in self.handlers)

    def discover(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        allowed_types: Collection[str] | None = None,
    ) -> IndexPlan:
        """Classify supported paths without loading parsers or models."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {root}")

        allowed = (
            set(self._handlers_by_type)
            if allowed_types is None
            else {file_type.casefold() for file_type in allowed_types}
        )
        unknown = allowed.difference(self._handlers_by_type)
        if unknown:
            raise ValueError(
                f"Unsupported index file type: {sorted(unknown)[0]}"
            )

        paths_by_type: dict[str, list[Path]] = {
            file_type: [] for file_type in allowed
        }
        pattern = "**/*" if recursive else "*"
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            handler = self._handlers_by_extension.get(path.suffix.casefold())
            if handler is not None and handler.file_type in allowed:
                paths_by_type[handler.file_type].append(path)

        return IndexPlan(
            queues=tuple(
                IndexQueue(
                    handler=handler,
                    paths=tuple(paths_by_type.get(handler.file_type, ())),
                )
                for handler in self.handlers
                if paths_by_type.get(handler.file_type)
            )
        )

    @staticmethod
    def batches(queue: IndexQueue, batch_size: int) -> Iterator[tuple[Path, ...]]:
        if batch_size < 1:
            raise ValueError("Index batch size must be positive")
        for start in range(0, len(queue.paths), batch_size):
            yield queue.paths[start : start + batch_size]
