"""Modality-independent query preparation and similarity search."""

from filelore.search.core import (
    FileQueryVectorizer,
    SearchRepository,
    SearchSource,
    embed_search_source,
    search_vectors,
    validate_query_file,
)
from filelore.search.image import ImageFileQueryVectorizer

__all__ = [
    "FileQueryVectorizer",
    "ImageFileQueryVectorizer",
    "SearchRepository",
    "SearchSource",
    "embed_search_source",
    "search_vectors",
    "validate_query_file",
]
