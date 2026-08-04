"""Stable file identity and content hashing helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid5


FILE_ID_NAMESPACE = UUID("daea7578-6e32-4a3f-9b3a-c0cc443646fb")


def normalized_path(path: str | Path) -> str:
    """Return the canonical path key used to identify a file across updates."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def file_point_id(path: str | Path) -> str:
    """Create a stable Qdrant-compatible UUID from an absolute file path."""
    return str(uuid5(FILE_ID_NAMESPACE, normalized_path(path)))


def calculate_file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 hash without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
