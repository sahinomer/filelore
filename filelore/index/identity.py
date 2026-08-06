"""Stable file identity and content hashing helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid5


FILE_ID_NAMESPACE = UUID("daea7578-6e32-4a3f-9b3a-c0cc443646fb")
FILE_SEGMENT_ID_NAMESPACE = UUID("93425ee8-579d-4962-b46a-37a81443f89f")


def normalized_path(path: str | Path) -> str:
    """Return the canonical path key used to identify a file across updates."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def file_point_id(path: str | Path) -> str:
    """Create a stable Qdrant-compatible UUID from an absolute file path."""
    return str(uuid5(FILE_ID_NAMESPACE, normalized_path(path)))


def file_segment_point_id(path: str | Path, segment_index: int) -> str:
    """Create a stable UUID for one numbered segment of a file."""
    if segment_index < 0:
        raise ValueError("Segment index must be non-negative")
    identity = f"{normalized_path(path)}\0{segment_index}"
    return str(uuid5(FILE_SEGMENT_ID_NAMESPACE, identity))


def calculate_file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 hash without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
