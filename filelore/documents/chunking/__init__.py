"""Interchangeable document chunking strategies."""

from filelore.documents.chunking.base import DocumentChunker, SentenceSplitter
from filelore.documents.chunking.paragraph import ParagraphChunker
from filelore.documents.chunking.sentence import UnicodeSentenceSplitter

__all__ = [
    "DocumentChunker",
    "ParagraphChunker",
    "SentenceSplitter",
    "UnicodeSentenceSplitter",
]
