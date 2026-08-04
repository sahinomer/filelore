"""Provider-neutral vector storage interfaces and implementations."""

from filelore.storage.base import (
    CollectionConfig,
    ConditionOperator,
    DistanceMetric,
    MetadataCondition,
    MetadataFilter,
    MetadataIndex,
    MetadataIndexType,
    RecordPage,
    StoredRecord,
    VectorConfig,
    VectorDatabase,
    VectorRecord,
    VectorSearchResult,
)
from filelore.storage.qdrant import QdrantVectorDatabase

__all__ = [
    "CollectionConfig",
    "ConditionOperator",
    "DistanceMetric",
    "MetadataCondition",
    "MetadataFilter",
    "MetadataIndex",
    "MetadataIndexType",
    "RecordPage",
    "QdrantVectorDatabase",
    "StoredRecord",
    "VectorConfig",
    "VectorDatabase",
    "VectorRecord",
    "VectorSearchResult",
]
