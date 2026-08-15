"""Base contract for format-specific document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from filelore.documents.models import ParsedDocument


class DocumentParser(ABC):
    """Parse one family of document formats into the common representation."""

    supported_extensions: ClassVar[frozenset[str]] = frozenset()

    def supports(self, path: str | Path) -> bool:
        """Return whether this parser recognizes the file extension."""
        return Path(path).suffix.casefold() in self.supported_extensions

    def prepare_path(self, path: str | Path) -> Path:
        """Validate and resolve a parser input path."""
        document_path = Path(path).expanduser()
        if not self.supports(document_path):
            extension = document_path.suffix or "<none>"
            raise ValueError(f"Unsupported document extension: {extension}")
        if not document_path.is_file():
            raise FileNotFoundError(document_path)
        return document_path.resolve()

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedDocument:
        """Extract text blocks and normalized metadata from one document."""
        raise NotImplementedError
