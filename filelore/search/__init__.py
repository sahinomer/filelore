"""Modality-independent query preparation and similarity search."""

from filelore.search.core import (
    FileQueryVectorizer,
    SearchRepository,
    SearchSource,
    embed_search_source,
    search_vectors,
    validate_query_file,
)

__all__ = [
    "FileQueryVectorizer",
    "SearchRepository",
    "SearchSource",
    "embed_search_source",
    "search_vectors",
    "validate_query_file",
]
