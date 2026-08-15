"""Format-specific document parser contracts."""

from filelore.documents.parsers.base import DocumentParser
from filelore.documents.parsers.html import HtmlDocumentParser
from filelore.documents.parsers.markdown import MarkdownDocumentParser
from filelore.documents.parsers.pdf import PdfDocumentParser

__all__ = [
    "DocumentParser",
    "HtmlDocumentParser",
    "MarkdownDocumentParser",
    "PdfDocumentParser",
]
