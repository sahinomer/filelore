"""Format-specific document parser contracts."""

from filelore.documents.parsers.base import DocumentParser
from filelore.documents.parsers.docx import DocxDocumentParser
from filelore.documents.parsers.html import HtmlDocumentParser
from filelore.documents.parsers.markdown import MarkdownDocumentParser
from filelore.documents.parsers.pdf import PdfDocumentParser
from filelore.documents.parsers.pptx import PptxDocumentParser

__all__ = [
    "DocumentParser",
    "DocxDocumentParser",
    "HtmlDocumentParser",
    "MarkdownDocumentParser",
    "PdfDocumentParser",
    "PptxDocumentParser",
]
