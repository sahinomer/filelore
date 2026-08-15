"""Contract for interchangeable document chunking strategies."""

from __future__ import annotations

from typing import Protocol

from filelore.documents.models import ParsedDocument, TextChunk


class DocumentChunker(Protocol):
    """Split a parsed document without depending on its original format."""

    def chunks(self, document: ParsedDocument) -> tuple[TextChunk, ...]: ...
