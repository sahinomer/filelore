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
from filelore.metadata import BaseMetadata
from filelore.processors.models import PreparedFile, PreparedSegment
from filelore.storage import (
    CollectionConfig,
    ConditionOperator,
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
        segment_collection_name: str | None = None,
        segment_vector_configs: Mapping[str, VectorConfig] | None = None,
    ) -> None:
        self.database = database
        self.collection_name = collection_name
        self.segment_collection_name = (
            segment_collection_name or f"{collection_name}_segments"
        )
        self._segments_configured = segment_vector_configs is not None
        self.database.ensure_collection(
            CollectionConfig(
                name=collection_name,
                vectors=dict(vector_configs or {}),
            )
        )
        if segment_vector_configs is not None:
            self.database.ensure_collection(
                CollectionConfig(
                    name=self.segment_collection_name,
                    vectors=dict(segment_vector_configs),
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

    def store_prepared(self, prepared_file: PreparedFile) -> FileIndexEntry:
        """Persist one processor result and replace its timed segments."""
        return self.store_prepared_many((prepared_file,))[0]

    def store_prepared_many(
        self,
        prepared_files: Sequence[PreparedFile],
        *,
        content_hashes: Sequence[str] | None = None,
    ) -> tuple[FileIndexEntry, ...]:
        """Persist processor results, including child segment vectors."""
        if not prepared_files:
            return ()
        if content_hashes is None:
            prepared_hashes: Sequence[str | None] = [None] * len(
                prepared_files
            )
        else:
            if len(content_hashes) != len(prepared_files):
                raise ValueError(
                    "content_hashes must match the number of prepared files"
                )
            prepared_hashes = content_hashes
        if (
            any(item.segments for item in prepared_files)
            and not self._segments_configured
        ):
            raise ValueError(
                "segment_vector_configs are required to store file segments"
            )

        entries: list[FileIndexEntry] = []
        parent_records: list[VectorRecord] = []
        segment_records: list[VectorRecord] = []
        for prepared_file, content_hash in zip(
            prepared_files, prepared_hashes
        ):
            self._validate_segments(prepared_file.segments)
            entry, parent_record = self._prepare_record(
                prepared_file.metadata,
                vectors=prepared_file.vectors,
                segment_count=len(prepared_file.segments),
                content_hash=content_hash,
            )
            entries.append(entry)
            parent_records.append(parent_record)
            segment_records.extend(
                self._prepare_segment_record(parent_record, segment)
                for segment in prepared_file.segments
            )

        parent_ids = tuple(entry.id for entry in entries)
        if self.database.collection_exists(self.segment_collection_name):
            self.database.delete_by_filter(
                self.segment_collection_name,
                MetadataFilter(
                    all_of=(
                        MetadataCondition(
                            "parent_id",
                            parent_ids,
                            operator=ConditionOperator.IN,
                        ),
                    )
                ),
            )
            self.database.upsert(self.segment_collection_name, segment_records)
        self.database.upsert(self.collection_name, parent_records)
        return tuple(entries)

    @staticmethod
    def _prepare_record(
        metadata: BaseMetadata,
        *,
        vectors: Mapping[str, Sequence[float]] | None,
        segment_count: int = 0,
        content_hash: str | None = None,
    ) -> tuple[FileIndexEntry, VectorRecord]:
        path = metadata.path.resolve()
        resolved_content_hash = content_hash or calculate_file_hash(path)
        indexed_at = datetime.now(timezone.utc)
        point_id = file_point_id(path)
        metadata_dict = metadata.to_dict()
        payload = {
            "schema_version": 1,
            "record_type": "file",
            "absolute_path": str(path),
            "path_key": normalized_path(path),
            "file_name": path.name,
            "file_name_search": path.name.casefold(),
            "content_hash": resolved_content_hash,
            "hash_algorithm": "sha256",
            "file_type": metadata.file_type,
            "extension": metadata.extension,
            "format_key": normalize_file_format(metadata.extension),
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
            "modified_at": metadata.modified_at.isoformat(),
            "indexed_at": indexed_at.isoformat(),
            "segment_count": segment_count,
            "metadata": metadata_dict,
        }
        entry = FileIndexEntry(
            id=point_id,
            path=path,
            content_hash=resolved_content_hash,
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

    @staticmethod
    def _prepare_segment_record(
        parent_record: VectorRecord,
        segment: PreparedSegment,
    ) -> VectorRecord:
        path = Path(str(parent_record.payload["absolute_path"]))
        payload = dict(parent_record.payload)
        payload.update(
            {
                "record_type": "segment",
                "parent_id": parent_record.id,
                "segment_index": segment.index,
            }
        )
        if segment.start_seconds is not None:
            payload["segment_start_seconds"] = segment.start_seconds
            payload["segment_end_seconds"] = segment.end_seconds
        conflicting_fields = payload.keys() & segment.payload.keys()
        if conflicting_fields:
            raise ValueError(
                "Segment payload must not replace inherited fields: "
                + ", ".join(sorted(conflicting_fields))
            )
        payload.update(segment.payload)
        return VectorRecord(
            id=file_segment_point_id(path, segment.index),
            payload=payload,
            vectors=dict(segment.vectors),
        )

    @staticmethod
    def _validate_segments(segments: Sequence[PreparedSegment]) -> None:
        indexes: set[int] = set()
        for segment in segments:
            if segment.index in indexes:
                raise ValueError("Segment indexes must be unique within a file")
            has_start = segment.start_seconds is not None
            has_end = segment.end_seconds is not None
            if has_start != has_end:
                raise ValueError("Segment timestamps must be provided together")
            if has_start and has_end:
                assert segment.start_seconds is not None
                assert segment.end_seconds is not None
                if (
                    segment.start_seconds < 0
                    or segment.end_seconds <= segment.start_seconds
                ):
                    raise ValueError(
                        "Segment timestamps must define a positive range"
                    )
            if not segment.vectors:
                raise ValueError("Stored file segments must contain vectors")
            indexes.add(segment.index)

    def get_by_path(self, path: str | Path) -> FileIndexEntry | None:
        records = self.database.retrieve(
            self.collection_name, [file_point_id(path)], with_vectors=False
        )
        return self._to_entry(records[0]) if records else None

    def get_by_paths(
        self,
        paths: Sequence[str | Path],
        *,
        batch_size: int = 1_000,
    ) -> tuple[FileIndexEntry | None, ...]:
        """Retrieve path records in input order with no per-file round trips."""
        if batch_size < 1:
            raise ValueError("Path lookup batch size must be positive")
        point_ids = tuple(file_point_id(path) for path in paths)
        entries_by_id: dict[str, FileIndexEntry] = {}
        for start in range(0, len(point_ids), batch_size):
            records = self.database.retrieve(
                self.collection_name,
                point_ids[start : start + batch_size],
                with_vectors=False,
            )
            entries_by_id.update(
                (record.id, self._to_entry(record)) for record in records
            )
        return tuple(entries_by_id.get(point_id) for point_id in point_ids)

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
        """Search common file and media fields, ignoring unspecified values."""
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

    def semantic_segment_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[FileSearchResult, ...]:
        """Return raw timed child matches without grouping parent files."""
        if not self.database.collection_exists(self.segment_collection_name):
            return ()
        results = self.database.search(
            self.segment_collection_name,
            vector,
            vector_name=vector_name,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        return tuple(
            FileSearchResult(
                file=self._to_entry(
                    result.record,
                    entry_id=self._segment_parent_id(result.record),
                ),
                score=result.score,
                segment=self._to_segment(result.record),
            )
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
        point_ids = tuple(file_point_id(path) for path in paths)
        if point_ids and self.database.collection_exists(
            self.segment_collection_name
        ):
            self.database.delete_by_filter(
                self.segment_collection_name,
                MetadataFilter(
                    all_of=(
                        MetadataCondition(
                            "parent_id",
                            point_ids,
                            operator=ConditionOperator.IN,
                        ),
                    )
                ),
            )
        self.database.delete(
            self.collection_name, point_ids
        )

    def count(self, metadata_filter: MetadataFilter | None = None) -> int:
        return self.database.count(self.collection_name, metadata_filter)

    @staticmethod
    def _to_entry(
        record: StoredRecord,
        *,
        entry_id: str | None = None,
    ) -> FileIndexEntry:
        metadata = record.payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        indexed_at = record.payload.get("indexed_at")
        if not isinstance(indexed_at, str):
            raise ValueError(f"Indexed record {record.id} has no indexed_at timestamp")
        return FileIndexEntry(
            id=entry_id or record.id,
            path=Path(str(record.payload["absolute_path"])),
            content_hash=str(record.payload["content_hash"]),
            file_type=str(record.payload["file_type"]),
            metadata=dict(metadata),
            indexed_at=datetime.fromisoformat(indexed_at),
        )

    @staticmethod
    def _segment_parent_id(record: StoredRecord) -> str:
        parent_id = record.payload.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(
                f"Segment record {record.id} has no valid parent_id"
            )
        return parent_id

    @staticmethod
    def _to_segment(record: StoredRecord) -> FileSegmentMatch:
        try:
            return FileSegmentMatch(
                index=int(record.payload["segment_index"]),
                start_seconds=float(record.payload["segment_start_seconds"]),
                end_seconds=float(record.payload["segment_end_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Segment record {record.id} has invalid timestamps"
            ) from error
