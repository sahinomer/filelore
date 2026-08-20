"""Format-neutral document parsing and chunking abstractions."""

from filelore.documents.chunking import (
    DocumentChunker,
    ParagraphChunker,
    SentenceSplitter,
    UnicodeSentenceSplitter,
)
from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
    TextChunk,
)
from filelore.documents.parsers import (
    DocxDocumentParser,
    DocumentParser,
    HtmlDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    PptxDocumentParser,
)
from filelore.documents.registry import DocumentParserRegistry


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset().union(
    PdfDocumentParser.supported_extensions,
    HtmlDocumentParser.supported_extensions,
    MarkdownDocumentParser.supported_extensions,
    DocxDocumentParser.supported_extensions,
    PptxDocumentParser.supported_extensions,
)

__all__ = [
    "DocumentChunker",
    "DocxDocumentParser",
    "DocumentParser",
    "DocumentParserRegistry",
    "HtmlDocumentParser",
    "MarkdownDocumentParser",
    "ParagraphChunker",
    "PdfDocumentParser",
    "PptxDocumentParser",
    "ParsedDocument",
    "SentenceSplitter",
    "SourceLocation",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "TextBlock",
    "TextBlockType",
    "TextChunk",
    "UnicodeSentenceSplitter",
]
