"""Persistence and search operations for the global file index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from filelore.index.filters import (
    file_metadata_filter,
    normalize_file_format,
)
from filelore.index.identity import (
    calculate_file_hash,
    file_point_id,
    normalized_path,
)
from filelore.index.models import (
    DuplicateGroup,
    FileIndexEntry,
    FileMetadataQuery,
    FileSearchResult,
)
from filelore.metadata import BaseMetadata
from filelore.storage import (
    CollectionConfig,
    MetadataCondition,
    MetadataFilter,
    StoredRecord,
    VectorConfig,
    VectorDatabase,
    VectorRecord,
)


class FileIndexRepository:
    """Provider-independent repository for indexed file records."""

    def __init__(
        self,
        database: VectorDatabase,
        *,
        collection_name: str = "files",
        vector_configs: Mapping[str, VectorConfig] | None = None,
    ) -> None:
        self.database = database
        self.collection_name = collection_name
        self.database.ensure_collection(
            CollectionConfig(
                name=collection_name,
                vectors=dict(vector_configs or {}),
            )
        )

    def store(
        self,
        metadata: BaseMetadata,
        *,
        vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> FileIndexEntry:
        return self.store_many([metadata], vector_sets=[vectors])[0]

    def store_many(
        self,
        metadata_items: Sequence[BaseMetadata],
        *,
        vector_sets: Sequence[Mapping[str, Sequence[float]] | None] | None = None,
    ) -> tuple[FileIndexEntry, ...]:
        """Hash and persist a batch, minimizing provider write overhead."""
        if vector_sets is None:
            vector_sets = [None] * len(metadata_items)
        if len(vector_sets) != len(metadata_items):
            raise ValueError("vector_sets must match the number of metadata items")

        entries: list[FileIndexEntry] = []
        records: list[VectorRecord] = []
        for metadata, vectors in zip(metadata_items, vector_sets):
            entry, record = self._prepare_record(metadata, vectors=vectors)
            entries.append(entry)
            records.append(record)
        self.database.upsert(self.collection_name, records)
        return tuple(entries)

    @staticmethod
    def _prepare_record(
        metadata: BaseMetadata,
        *,
        vectors: Mapping[str, Sequence[float]] | None,
    ) -> tuple[FileIndexEntry, VectorRecord]:
        path = metadata.path.resolve()
        content_hash = calculate_file_hash(path)
        indexed_at = datetime.now(timezone.utc)
        point_id = file_point_id(path)
        metadata_dict = metadata.to_dict()
        detected_format = metadata_dict.get("image_format") or metadata.extension
        payload = {
            "schema_version": 1,
            "absolute_path": str(path),
            "path_key": normalized_path(path),
            "file_name": path.name,
            "file_name_search": path.name.casefold(),
            "content_hash": content_hash,
            "hash_algorithm": "sha256",
            "file_type": metadata.file_type,
            "extension": metadata.extension,
            "format_key": normalize_file_format(str(detected_format)),
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
            "modified_at": metadata.modified_at.isoformat(),
            "indexed_at": indexed_at.isoformat(),
            "metadata": metadata_dict,
        }
        entry = FileIndexEntry(
            id=point_id,
            path=path,
            content_hash=content_hash,
            file_type=metadata.file_type,
            metadata=metadata_dict,
            indexed_at=indexed_at,
        )
        record = VectorRecord(
            id=point_id,
            payload=payload,
            vectors=dict(vectors or {}),
        )
        return entry, record

    def get_by_path(self, path: str | Path) -> FileIndexEntry | None:
        records = self.database.retrieve(
            self.collection_name, [file_point_id(path)], with_vectors=False
        )
        return self._to_entry(records[0]) if records else None

    def find_by_hash(
        self, content_hash: str, *, limit: int = 100
    ) -> tuple[FileIndexEntry, ...]:
        return self.search_metadata(
            MetadataFilter(
                all_of=(MetadataCondition("content_hash", content_hash),)
            ),
            limit=limit,
        )

    def search_metadata(
        self,
        metadata_filter: MetadataFilter | None = None,
        *,
        limit: int = 100,
    ) -> tuple[FileIndexEntry, ...]:
        page = self.database.filter(
            self.collection_name,
            metadata_filter=metadata_filter,
            limit=limit,
        )
        return tuple(self._to_entry(record) for record in page.records)

    def search_files(
        self, query: FileMetadataQuery, *, limit: int = 50
    ) -> tuple[FileIndexEntry, ...]:
        """Search common file and image fields, ignoring unspecified values."""
        return self.search_metadata(file_metadata_filter(query), limit=limit)

    def semantic_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[FileSearchResult, ...]:
        results = self.database.search(
            self.collection_name,
            vector,
            vector_name=vector_name,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        return tuple(
            FileSearchResult(file=self._to_entry(result.record), score=result.score)
            for result in results
        )

    def iter_all(self, *, page_size: int = 100) -> Iterator[FileIndexEntry]:
        offset: str | None = None
        while True:
            page = self.database.filter(
                self.collection_name, limit=page_size, offset=offset
            )
            yield from (self._to_entry(record) for record in page.records)
            if page.next_offset is None:
                break
            offset = page.next_offset

    def find_duplicate_groups(self) -> tuple[DuplicateGroup, ...]:
        groups: dict[str, list[FileIndexEntry]] = {}
        for entry in self.iter_all():
            groups.setdefault(entry.content_hash, []).append(entry)
        return tuple(
            DuplicateGroup(content_hash=digest, files=tuple(files))
            for digest, files in sorted(groups.items())
            if len(files) > 1
        )

    def remove(self, paths: Sequence[str | Path]) -> None:
        self.database.delete(
            self.collection_name, [file_point_id(path) for path in paths]
        )

    def count(self, metadata_filter: MetadataFilter | None = None) -> int:
        return self.database.count(self.collection_name, metadata_filter)

    @staticmethod
    def _to_entry(record: StoredRecord) -> FileIndexEntry:
        metadata = record.payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        indexed_at = record.payload.get("indexed_at")
        if not isinstance(indexed_at, str):
            raise ValueError(f"Indexed record {record.id} has no indexed_at timestamp")
        return FileIndexEntry(
            id=record.id,
            path=Path(str(record.payload["absolute_path"])),
            content_hash=str(record.payload["content_hash"]),
            file_type=str(record.payload["file_type"]),
            metadata=dict(metadata),
            indexed_at=datetime.fromisoformat(indexed_at),
        )
