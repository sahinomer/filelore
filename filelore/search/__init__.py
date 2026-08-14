"""Public request parsing and semantic-search API."""

from filelore.search.audio import AudioFileQueryVectorizer
from filelore.search.execution import (
    embed_search_source,
    group_segment_results,
    search_vectors,
    validate_query_file,
)
from filelore.search.image import ImageFileQueryVectorizer
from filelore.search.models import (
    SearchRequest,
    SearchResponse,
    SearchResultGroup,
    SearchSource,
    SearchTarget,
    SearchTimings,
)
from filelore.search.query_parser import (
    ParsedSearchFilters,
    ParsedSearchQuery,
    parse_search_filters,
    parse_search_query,
    target_for_format,
    validate_search_metadata,
    validate_search_target,
)
from filelore.search.request_builder import (
    build_interactive_search_request,
    build_structured_search_request,
    resolve_search_target,
)
from filelore.search.protocols import FileQueryVectorizer, SearchRepository
from filelore.search.service import (
    SEGMENT_GROUP_OVERFETCH_FACTOR,
    SearchService,
)
from filelore.search.targets import default_file_query_vectorizers

__all__ = [
    "AudioFileQueryVectorizer",
    "FileQueryVectorizer",
    "ImageFileQueryVectorizer",
    "ParsedSearchFilters",
    "ParsedSearchQuery",
    "SEGMENT_GROUP_OVERFETCH_FACTOR",
    "SearchRequest",
    "SearchRepository",
    "SearchResponse",
    "SearchResultGroup",
    "SearchService",
    "SearchSource",
    "SearchTimings",
    "SearchTarget",
    "build_interactive_search_request",
    "build_structured_search_request",
    "default_file_query_vectorizers",
    "embed_search_source",
    "group_segment_results",
    "parse_search_filters",
    "parse_search_query",
    "resolve_search_target",
    "search_vectors",
    "target_for_format",
    "validate_search_metadata",
    "validate_query_file",
    "validate_search_target",
]
