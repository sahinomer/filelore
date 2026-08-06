"""Format-independent results produced by file processors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from filelore.embedding import EmbeddingVector
from filelore.metadata import BaseMetadata


MetadataType = TypeVar("MetadataType", bound=BaseMetadata)


@dataclass(frozen=True, slots=True)
class PreparedSegment:
    """Timed vectors prepared from one segment of a parent file."""

    index: int
    start_seconds: float
    end_seconds: float
    vectors: dict[str, EmbeddingVector]


@dataclass(frozen=True, slots=True)
class PreparedFile(Generic[MetadataType]):
    """Metadata and optional vectors prepared for one index record."""

    metadata: MetadataType
    vectors: dict[str, EmbeddingVector]
    segments: tuple[PreparedSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessingFailure:
    """A file that could not be prepared for indexing."""

    path: Path
    error: Exception


@dataclass(frozen=True, slots=True)
class ProcessingBatch(Generic[MetadataType]):
    """Successful and failed results from one processor batch."""

    files: tuple[PreparedFile[MetadataType], ...]
    failures: tuple[ProcessingFailure, ...] = ()
