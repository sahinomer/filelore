"""Public models, helpers, and repository for the persistent file index."""

from filelore.index.filters import (
    file_metadata_filter,
    file_type_filter,
    normalize_file_format,
)
from filelore.index.identity import (
    FILE_ID_NAMESPACE,
    FILE_SEGMENT_ID_NAMESPACE,
    calculate_file_hash,
    file_point_id,
    file_segment_point_id,
    normalized_path,
)
from filelore.index.models import (
    DuplicateGroup,
    FileIndexEntry,
    FileMetadataQuery,
    FileSegmentMatch,
    FileSearchResult,
)
from filelore.index.pipeline import (
    FileIndexer,
    FileProcessor,
    IndexCoordinator,
    IndexHandler,
    IndexingBatch,
    IndexPlan,
    IndexQueue,
)
from filelore.index.repository import FileIndexRepository

__all__ = [
    "FILE_ID_NAMESPACE",
    "FILE_SEGMENT_ID_NAMESPACE",
    "DuplicateGroup",
    "FileIndexEntry",
    "FileIndexRepository",
    "FileIndexer",
    "FileMetadataQuery",
    "FileSegmentMatch",
    "FileProcessor",
    "FileSearchResult",
    "IndexCoordinator",
    "IndexHandler",
    "IndexingBatch",
    "IndexPlan",
    "IndexQueue",
    "calculate_file_hash",
    "file_metadata_filter",
    "file_point_id",
    "file_segment_point_id",
    "file_type_filter",
    "normalize_file_format",
    "normalized_path",
]
