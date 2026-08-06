"""Orchestration between file processors and the persistent index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, Protocol, Sequence, TypeVar

from filelore.index.models import FileIndexEntry
from filelore.index.repository import FileIndexRepository
from filelore.metadata import BaseMetadata
from filelore.processors.models import ProcessingBatch, ProcessingFailure


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
