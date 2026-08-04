"""Qdrant implementation of the provider-neutral vector database contract."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from filelore.storage.base import (
    CollectionConfig,
    ConditionOperator,
    DistanceMetric,
    MetadataCondition,
    MetadataFilter,
    MetadataIndexType,
    RecordPage,
    StoredRecord,
    VectorDatabase,
    VectorRecord,
    VectorSearchResult,
)


_DISTANCES = {
    DistanceMetric.COSINE: models.Distance.COSINE,
    DistanceMetric.DOT: models.Distance.DOT,
    DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
    DistanceMetric.MANHATTAN: models.Distance.MANHATTAN,
}

_INDEX_TYPES = {
    MetadataIndexType.KEYWORD: models.PayloadSchemaType.KEYWORD,
    MetadataIndexType.INTEGER: models.PayloadSchemaType.INTEGER,
    MetadataIndexType.FLOAT: models.PayloadSchemaType.FLOAT,
    MetadataIndexType.BOOLEAN: models.PayloadSchemaType.BOOL,
    MetadataIndexType.DATETIME: models.PayloadSchemaType.DATETIME,
    MetadataIndexType.TEXT: models.PayloadSchemaType.TEXT,
    MetadataIndexType.UUID: models.PayloadSchemaType.UUID,
}


class QdrantVectorDatabase(VectorDatabase):
    """Persist vectors and JSON payloads using the standard Qdrant client API."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        url: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if client is not None:
            if path is not None or url is not None:
                raise ValueError(
                    "A preconfigured client cannot be combined with 'url' or 'path'"
                )
            self.path = None
            self.url = None
            self._client = client
        elif url is not None:
            self.path = None
            self.url = url
            self._client = QdrantClient(url=url)
        elif path is not None:
            self.path = Path(path).expanduser()
            self.url = None
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))
        else:
            raise ValueError("Provide at least one of 'url', 'path', or 'client'")

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def create_collection(self, config: CollectionConfig) -> None:
        vectors = {
            name: models.VectorParams(
                size=vector.dimensions,
                distance=_DISTANCES[vector.distance],
            )
            for name, vector in config.vectors.items()
        }
        self._client.create_collection(
            collection_name=config.name,
            vectors_config=vectors,
        )
        for metadata_index in config.metadata_indexes:
            self._client.create_payload_index(
                collection_name=config.name,
                field_name=metadata_index.field,
                field_schema=_INDEX_TYPES[metadata_index.field_type],
            )

    def ensure_collection(self, config: CollectionConfig) -> None:
        if not self.collection_exists(config.name):
            self.create_collection(config)
            return

        collection = self._client.get_collection(config.name)
        existing_vectors = collection.config.params.vectors
        if not isinstance(existing_vectors, dict):
            if config.vectors:
                raise ValueError(
                    f"Collection {config.name!r} uses an unnamed vector and cannot "
                    "accept named vectors"
                )
            return

        for name, vector in config.vectors.items():
            existing = existing_vectors.get(name)
            if existing is None:
                self._client.create_vector_name(
                    collection_name=config.name,
                    vector_name=name,
                    vector_name_config=models.DenseVectorNameConfig(
                        dense=models.DenseVectorConfig(
                            size=vector.dimensions,
                            distance=_DISTANCES[vector.distance],
                        )
                    ),
                )
                continue
            if (
                existing.size != vector.dimensions
                or existing.distance != _DISTANCES[vector.distance]
            ):
                raise ValueError(
                    f"Vector {name!r} in collection {config.name!r} has an "
                    "incompatible configuration"
                )

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        points = [
            models.PointStruct(
                id=record.id,
                payload=dict(record.payload),
                vector={
                    name: [float(value) for value in vector]
                    for name, vector in record.vectors.items()
                },
            )
            for record in records
        ]
        self._client.upsert(collection_name=collection, points=points, wait=True)

    def retrieve(
        self, collection: str, ids: Sequence[str], *, with_vectors: bool = False
    ) -> tuple[StoredRecord, ...]:
        if not ids:
            return ()
        records = self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            with_payload=True,
            with_vectors=with_vectors,
        )
        return tuple(self._to_stored_record(record) for record in records)

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
        if limit < 1:
            raise ValueError("Search limit must be positive")
        response = self._client.query_points(
            collection_name=collection,
            query=[float(value) for value in vector],
            using=vector_name,
            query_filter=self._to_qdrant_filter(metadata_filter),
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return tuple(
            VectorSearchResult(
                record=self._to_stored_record(point), score=float(point.score)
            )
            for point in response.points
        )

    def filter(
        self,
        collection: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> RecordPage:
        if limit < 1:
            raise ValueError("Filter limit must be positive")
        records, next_offset = self._client.scroll(
            collection_name=collection,
            scroll_filter=self._to_qdrant_filter(metadata_filter),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return RecordPage(
            records=tuple(self._to_stored_record(record) for record in records),
            next_offset=str(next_offset) if next_offset is not None else None,
        )

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(ids)),
            wait=True,
        )

    def count(
        self, collection: str, metadata_filter: MetadataFilter | None = None
    ) -> int:
        result = self._client.count(
            collection_name=collection,
            count_filter=self._to_qdrant_filter(metadata_filter),
            exact=True,
        )
        return result.count

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_stored_record(record: Any) -> StoredRecord:
        vectors: dict[str, list[float]] = {}
        if isinstance(record.vector, dict):
            vectors = {
                name: [float(value) for value in vector]
                for name, vector in record.vector.items()
                if isinstance(vector, list)
            }
        elif isinstance(record.vector, list):
            vectors[""] = [float(value) for value in record.vector]

        return StoredRecord(
            id=str(record.id),
            payload=dict(record.payload or {}),
            vectors=vectors,
        )

    @classmethod
    def _to_qdrant_filter(
        cls, metadata_filter: MetadataFilter | None
    ) -> models.Filter | None:
        if metadata_filter is None:
            return None
        return models.Filter(
            must=[cls._to_condition(item) for item in metadata_filter.all_of] or None,
            should=[cls._to_condition(item) for item in metadata_filter.any_of]
            or None,
            must_not=[cls._to_condition(item) for item in metadata_filter.none_of]
            or None,
        )

    @staticmethod
    def _to_condition(condition: MetadataCondition) -> models.FieldCondition:
        operator = condition.operator
        if operator in {ConditionOperator.EQUAL, ConditionOperator.CONTAINS}:
            return models.FieldCondition(
                key=condition.field,
                match=models.MatchValue(value=condition.value),
            )
        if operator is ConditionOperator.IN:
            if not isinstance(condition.value, (list, tuple, set, frozenset)):
                raise TypeError("The 'in' operator requires a sequence value")
            return models.FieldCondition(
                key=condition.field,
                match=models.MatchAny(any=list(condition.value)),
            )
        if operator is ConditionOperator.TEXT_CONTAINS:
            return models.FieldCondition(
                key=condition.field,
                match=models.MatchText(text=str(condition.value)),
            )

        range_arguments: dict[str, Any] = {}
        range_key = {
            ConditionOperator.GREATER_THAN: "gt",
            ConditionOperator.GREATER_THAN_OR_EQUAL: "gte",
            ConditionOperator.LESS_THAN: "lt",
            ConditionOperator.LESS_THAN_OR_EQUAL: "lte",
        }.get(operator)
        if range_key is None:
            raise ValueError(f"Unsupported condition operator: {operator}")
        if isinstance(condition.value, (date, datetime)):
            range_arguments[range_key] = condition.value
            return models.FieldCondition(
                key=condition.field,
                range=models.DatetimeRange(**range_arguments),
            )
        range_arguments[range_key] = float(condition.value)
        return models.FieldCondition(
            key=condition.field,
            range=models.Range(**range_arguments),
        )
