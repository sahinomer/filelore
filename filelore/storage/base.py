"""Provider-independent contracts for vector and metadata storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ConditionOperator(str, Enum):
    EQUAL = "equal"
    IN = "in"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    TEXT_CONTAINS = "text_contains"


class MetadataIndexType(str, Enum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "bool"
    DATETIME = "datetime"
    TEXT = "text"
    UUID = "uuid"


@dataclass(frozen=True, slots=True)
class VectorConfig:
    dimensions: int
    distance: DistanceMetric = DistanceMetric.COSINE

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("Vector dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MetadataIndex:
    field: str
    field_type: MetadataIndexType


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    name: str
    vectors: Mapping[str, VectorConfig] = field(default_factory=dict)
    metadata_indexes: tuple[MetadataIndex, ...] = ()


@dataclass(frozen=True, slots=True)
class MetadataCondition:
    field: str
    value: Any
    operator: ConditionOperator = ConditionOperator.EQUAL


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Boolean metadata conditions supported by every storage adapter."""

    all_of: tuple[MetadataCondition, ...] = ()
    any_of: tuple[MetadataCondition, ...] = ()
    none_of: tuple[MetadataCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    vectors: Mapping[str, Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredRecord:
    id: str
    payload: dict[str, Any]
    vectors: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    record: StoredRecord
    score: float


@dataclass(frozen=True, slots=True)
class RecordPage:
    records: tuple[StoredRecord, ...]
    next_offset: str | None = None


class VectorDatabase(ABC):
    """Operations FileLore requires from any vector database provider."""

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> None:
        raise NotImplementedError

    def ensure_collection(self, config: CollectionConfig) -> None:
        if not self.collection_exists(config.name):
            self.create_collection(config)

    @abstractmethod
    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self, collection: str, ids: Sequence[str], *, with_vectors: bool = False
    ) -> tuple[StoredRecord, ...]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int = 10,
        metadata_filter: MetadataFilter | None = None,
        with_vectors: bool = False,
    ) -> tuple[VectorSearchResult, ...]:
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        collection: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> RecordPage:
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection: str, ids: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(
        self, collection: str, metadata_filter: MetadataFilter | None = None
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> VectorDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
