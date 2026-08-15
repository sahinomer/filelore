"""Format-neutral document parsing and chunking abstractions."""

from filelore.documents.chunking import DocumentChunker
from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
    TextChunk,
)
from filelore.documents.parsers import DocumentParser, MarkdownDocumentParser
from filelore.documents.registry import DocumentParserRegistry

__all__ = [
    "DocumentChunker",
    "DocumentParser",
    "DocumentParserRegistry",
    "MarkdownDocumentParser",
    "ParsedDocument",
    "SourceLocation",
    "TextBlock",
    "TextBlockType",
    "TextChunk",
]
