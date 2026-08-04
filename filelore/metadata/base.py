"""Format-independent metadata types and parser contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, Iterator, TypeVar


def _serialize(value: Any) -> Any:
    """Convert metadata values into JSON-compatible Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class BaseMetadata:
    """Metadata shared by every indexed file format."""

    file_type: ClassVar[str] = "file"

    path: Path
    extension: str
    mime_type: str | None
    size_bytes: int
    modified_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a representation suitable for JSON output and indexing."""
        return _serialize(asdict(self))


MetadataType = TypeVar("MetadataType", bound=BaseMetadata)


class MetadataParser(ABC, Generic[MetadataType]):
    """Base contract for a parser dedicated to one family of file formats."""

    supported_extensions: frozenset[str] = frozenset()

    def supports(self, path: str | Path) -> bool:
        """Return whether this parser recognizes the file extension."""
        return Path(path).suffix.lower() in self.supported_extensions

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]:
        """Yield supported files beneath a directory in stable path order."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {root}")

        pattern = "**/*" if recursive else "*"
        for path in sorted(root.glob(pattern)):
            if path.is_file() and self.supports(path):
                yield path

    @abstractmethod
    def parse(self, path: str | Path) -> MetadataType:
        """Extract metadata from one supported file."""
        raise NotImplementedError
