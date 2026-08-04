"""Data models returned and accepted by the file index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FileIndexEntry:
    id: str
    path: Path
    content_hash: str
    file_type: str
    metadata: dict[str, Any]
    indexed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "content_hash": self.content_hash,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "indexed_at": self.indexed_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    content_hash: str
    files: tuple[FileIndexEntry, ...]


@dataclass(frozen=True, slots=True)
class FileSearchResult:
    file: FileIndexEntry
    score: float


@dataclass(frozen=True, slots=True)
class FileMetadataQuery:
    """Optional metadata fields used to constrain file search."""

    name_contains: str | None = None
    file_format: str | None = None
    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None
