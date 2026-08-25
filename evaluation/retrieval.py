"""Evaluate text-to-media or text-to-document retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence, TypeVar

from filelore.cli import DEFAULT_INDEX_PATH
from filelore.cli_display import CliDisplay
from filelore.embedding import (
    ClapAudioEmbedding,
    ClipImageEmbedding,
    EmbeddingVector,
    HarrierTextEmbedding,
    TextEmbedding,
)
from filelore.index import (
    FileIndexEntry,
    FileIndexRepository,
    FileSearchResult,
    calculate_file_hash,
    normalize_file_format,
)
from filelore.storage import (
    MetadataCondition,
    MetadataFilter,
    QdrantVectorDatabase,
)


DEFAULT_CUTOFFS = (1, 5, 10)
DEFAULT_BATCH_SIZE = 32
DEFAULT_LATENCY_SAMPLES = 1_000
DEFAULT_LATENCY_WARMUP = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
DEFAULT_DOCUMENT_FORMATS = ("docx", "html", "md", "pdf", "pptx")
DEFAULT_DOCUMENT_VIEW_WORKERS = 6
DOCUMENT_QUERY_TYPES = (
    "Analytical",
    "Boolean",
    "Comparative",
    "Descriptive",
    "Open-Ended",
    "Procedural",
    "Temporal",
)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetrievalAnnotation:
    query: str
    relevant_file: str


@dataclass(frozen=True, slots=True)
class DocumentRetrievalAnnotation:
    enterprise: str
    query_type: str
    query: str
    relevant_files: tuple[str, ...]
    source_rows: int


@dataclass(frozen=True, slots=True)
class IndexedDocumentVariant:
    file_id: str
    file_format: str
    path: str


@dataclass(frozen=True, slots=True)
class RelevantDocument:
    logical_id: str
    annotation_file: str
    variants: tuple[IndexedDocumentVariant, ...]


@dataclass(frozen=True, slots=True)
class EligibleDocumentQuery:
    enterprise: str
    query_type: str
    query: str
    relevant_documents: tuple[RelevantDocument, ...]


@dataclass(frozen=True, slots=True)
class DocumentViewQuery:
    enterprise: str
    query_type: str
    query: str
    relevant_logical_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentAnnotationCoverage:
    annotation_rows: int
    annotation_queries: int
    annotation_files: int
    indexed_target_files: int
    indexed_logical_files: int
    indexed_formats: Mapping[str, int]
    eligible_queries: tuple[EligibleDocumentQuery, ...]
    fully_covered_queries: int
    partially_covered_queries: int
    skipped_queries: int
    missing_files: tuple[str, ...]
    missing_format_variants: Mapping[str, tuple[str, ...]]
    ambiguous_files: tuple[str, ...]
    logical_id_by_file_id: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_rows": self.annotation_rows,
            "annotation_queries": self.annotation_queries,
            "annotation_files": self.annotation_files,
            "indexed_target_files": self.indexed_target_files,
            "indexed_logical_files": self.indexed_logical_files,
            "indexed_formats": dict(self.indexed_formats),
            "eligible_queries": len(self.eligible_queries),
            "fully_covered_queries": self.fully_covered_queries,
            "partially_covered_queries": self.partially_covered_queries,
            "skipped_queries": self.skipped_queries,
            "missing_files": list(self.missing_files),
            "missing_format_variants": {
                filename: list(formats)
                for filename, formats in self.missing_format_variants.items()
            },
            "ambiguous_files": list(self.ambiguous_files),
        }


@dataclass(frozen=True, slots=True)
class EligibleQuery:
    query: str
    relevant_file: str
    relevant_id: str


@dataclass(frozen=True, slots=True)
class AnnotationCoverage:
    annotation_queries: int
    annotation_files: int
    indexed_target_files: int
    eligible_queries: tuple[EligibleQuery, ...]
    eligible_files: int
    missing_files: tuple[str, ...]
    ambiguous_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_queries": self.annotation_queries,
            "annotation_files": self.annotation_files,
            "indexed_target_files": self.indexed_target_files,
            "eligible_queries": len(self.eligible_queries),
            "eligible_files": self.eligible_files,
            "missing_files": list(self.missing_files),
            "ambiguous_files": list(self.ambiguous_files),
        }


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    cutoff: int
    mrr: float
    recall: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class DocumentRetrievalMetrics:
    cutoff: int
    hit: float
    mrr: float
    recall: float
    ndcg: float
    map: float
    complete: float


@dataclass(frozen=True, slots=True)
class DocumentQueryOutcome:
    enterprise: str
    query_type: str
    query: str
    relevant_count: int
    relevant_logical_ids: tuple[str, ...]
    relevant_ranks: tuple[int, ...]
    returned_physical_files: int


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    samples: int
    warmup_queries: int
    average_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(frozen=True, slots=True)
class EvaluationThroughput:
    batch_size: int
    total_ms: float
    embedding_ms: float
    retrieval_ms: float
    queries_per_second: float
    amortized_ms_per_query: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    ranks: tuple[int | None, ...]
    metrics: tuple[RetrievalMetrics, ...]
    throughput: EvaluationThroughput


@dataclass(frozen=True, slots=True)
class DocumentEvaluationResult:
    outcomes: tuple[DocumentQueryOutcome, ...]
    metrics: tuple[DocumentRetrievalMetrics, ...]
    metrics_by_query_type: Mapping[
        str, tuple[DocumentRetrievalMetrics, ...]
    ]
    macro_query_type_metrics: tuple[DocumentRetrievalMetrics, ...]
    multi_document_metrics: tuple[DocumentRetrievalMetrics, ...] | None
    throughput: EvaluationThroughput


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingCache:
    vectors_by_query: Mapping[str, EmbeddingVector]
    embedding_ms: float


@dataclass(frozen=True, slots=True)
class DocumentMetricPassTiming:
    workers: int
    query_count: int
    view_evaluations: int
    embedding_ms: float
    parallel_retrieval_ms: float
    total_ms: float
    queries_per_second: float
    view_evaluations_per_second: float


def parse_clotho_annotations(path: Path) -> tuple[RetrievalAnnotation, ...]:
    """Expand every Clotho caption column into a separate retrieval query."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if "file_name" not in fields:
            raise ValueError("Clotho annotations require a 'file_name' column")
        caption_fields = sorted(
            (field for field in fields if field.startswith("caption_")),
            key=_caption_field_order,
        )
        if not caption_fields:
            raise ValueError("Clotho annotations require caption columns")

        annotations: list[RetrievalAnnotation] = []
        for row_number, row in enumerate(reader, start=2):
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                raise ValueError(
                    f"Clotho row {row_number} has no file_name"
                )
            for field in caption_fields:
                caption = (row.get(field) or "").strip()
                if not caption:
                    raise ValueError(
                        f"Clotho row {row_number} has no value for {field}"
                    )
                annotations.append(
                    RetrievalAnnotation(caption, file_name)
                )
    return tuple(annotations)


def parse_coco_annotations(path: Path) -> tuple[RetrievalAnnotation, ...]:
    """Join COCO captions to image filenames through their image IDs."""
    with path.open("r", encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("COCO annotations must contain a JSON object")
    images = document.get("images")
    captions = document.get("annotations")
    if not isinstance(images, list) or not isinstance(captions, list):
        raise ValueError("COCO annotations require 'images' and 'annotations' lists")

    filenames_by_id: dict[object, str] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Every COCO image entry must be an object")
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("Every COCO image requires an id and file_name")
        if image_id in filenames_by_id:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        filenames_by_id[image_id] = file_name.strip()

    annotations: list[RetrievalAnnotation] = []
    for annotation in captions:
        if not isinstance(annotation, dict):
            raise ValueError("Every COCO annotation entry must be an object")
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if image_id not in filenames_by_id:
            raise ValueError(
                f"COCO annotation references unknown image id: {image_id}"
            )
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Every COCO annotation requires a caption")
        annotations.append(
            RetrievalAnnotation(caption.strip(), filenames_by_id[image_id])
        )
    return tuple(annotations)


def parse_document_annotations(
    path: Path,
) -> tuple[DocumentRetrievalAnnotation, ...]:
    """Parse and consolidate RAG-Multi-Corpus supporting-fact rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_fields = {
            "Enterprise Name",
            "Query Type",
            "Query",
            "Supporting Facts",
        }
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                "Document annotations are missing columns: "
                + ", ".join(sorted(missing_fields))
            )

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            enterprise = (row.get("Enterprise Name") or "").strip()
            query = (row.get("Query") or "").strip()
            query_type = _document_query_type(
                (row.get("Query Type") or "").strip()
            )
            if not enterprise or not query:
                raise ValueError(
                    f"Document annotation row {row_number} requires an "
                    "enterprise and query"
                )
            raw_facts = row.get("Supporting Facts") or ""
            try:
                facts = json.loads(raw_facts)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Supporting Facts JSON on row {row_number}"
                ) from error
            if not isinstance(facts, list) or not facts:
                raise ValueError(
                    f"Document annotation row {row_number} requires "
                    "supporting facts"
                )
            filenames: list[str] = []
            for fact in facts:
                if not isinstance(fact, dict):
                    raise ValueError(
                        f"Supporting fact on row {row_number} must be an object"
                    )
                filename = fact.get("filename")
                if not isinstance(filename, str) or not filename.strip():
                    raise ValueError(
                        f"Supporting fact on row {row_number} requires a filename"
                    )
                filenames.append(filename.strip())

            key = (_identity_key(enterprise), _query_key(query))
            group = grouped.setdefault(
                key,
                {
                    "enterprise": enterprise,
                    "query_type": query_type,
                    "query": query,
                    "relevant_files": {},
                    "source_rows": 0,
                },
            )
            if group["query_type"] != query_type:
                raise ValueError(
                    "Duplicate document query has conflicting query types: "
                    f"{query}"
                )
            for filename in filenames:
                group["relevant_files"].setdefault(
                    _document_filename_key(filename), filename
                )
            group["source_rows"] += 1

    return tuple(
        DocumentRetrievalAnnotation(
            enterprise=group["enterprise"],
            query_type=group["query_type"],
            query=group["query"],
            relevant_files=tuple(group["relevant_files"].values()),
            source_rows=group["source_rows"],
        )
        for group in grouped.values()
    )


def load_annotations(
    path: Path, target: str
) -> tuple[RetrievalAnnotation, ...]:
    if target == "image":
        return parse_coco_annotations(path)
    if target == "audio":
        return parse_clotho_annotations(path)
    raise ValueError(f"Unsupported retrieval target: {target}")


def filter_indexed_annotations(
    annotations: Sequence[RetrievalAnnotation],
    indexed_entries: Sequence[FileIndexEntry],
    *,
    target: str,
) -> AnnotationCoverage:
    """Resolve annotations to unique indexed parent records by filename."""
    target_entries = tuple(
        entry for entry in indexed_entries if entry.file_type == target
    )
    entries_by_name: dict[str, list[FileIndexEntry]] = {}
    for entry in target_entries:
        entries_by_name.setdefault(
            _filename_key(entry.path.name), []
        ).append(entry)

    original_names: dict[str, str] = {}
    for annotation in annotations:
        key = _filename_key(annotation.relevant_file)
        original_names.setdefault(key, annotation.relevant_file)

    missing_keys = {
        key for key in original_names if key not in entries_by_name
    }
    ambiguous_keys = {
        key
        for key in original_names
        if len(entries_by_name.get(key, ())) > 1
    }
    eligible: list[EligibleQuery] = []
    eligible_file_keys: set[str] = set()
    for annotation in annotations:
        key = _filename_key(annotation.relevant_file)
        matches = entries_by_name.get(key, ())
        if len(matches) != 1:
            continue
        eligible_file_keys.add(key)
        eligible.append(
            EligibleQuery(
                query=annotation.query,
                relevant_file=annotation.relevant_file,
                relevant_id=matches[0].id,
            )
        )

    return AnnotationCoverage(
        annotation_queries=len(annotations),
        annotation_files=len(original_names),
        indexed_target_files=len(target_entries),
        eligible_queries=tuple(eligible),
        eligible_files=len(eligible_file_keys),
        missing_files=tuple(
            sorted(original_names[key] for key in missing_keys)
        ),
        ambiguous_files=tuple(
            sorted(original_names[key] for key in ambiguous_keys)
        ),
    )


def filter_indexed_document_annotations(
    annotations: Sequence[DocumentRetrievalAnnotation],
    indexed_entries: Sequence[FileIndexEntry],
) -> DocumentAnnotationCoverage:
    """Resolve logical document citations to indexed format variants."""
    target_entries = tuple(
        entry for entry in indexed_entries if entry.file_type == "text"
    )
    logical_id_by_file_id = {
        entry.id: _logical_document_id(entry) for entry in target_entries
    }
    indexed_formats: dict[str, int] = defaultdict(int)
    entries_by_key: dict[tuple[str, str], list[FileIndexEntry]] = defaultdict(list)
    for entry in target_entries:
        file_format = normalize_file_format(entry.path.suffix)
        indexed_formats[file_format] += 1
        entries_by_key[
            (
                _identity_key(_document_enterprise(entry)),
                _document_filename_key(entry.path.name),
            )
        ].append(entry)

    annotation_file_keys = {
        (_identity_key(annotation.enterprise), _document_filename_key(filename))
        for annotation in annotations
        for filename in annotation.relevant_files
    }
    missing_files: set[str] = set()
    missing_format_variants: dict[str, tuple[str, ...]] = {}
    ambiguous_files: set[str] = set()
    eligible: list[EligibleDocumentQuery] = []
    fully_covered = 0
    partially_covered = 0
    skipped = 0

    for annotation in annotations:
        relevant_documents: list[RelevantDocument] = []
        unresolved = 0
        for annotation_file in annotation.relevant_files:
            display_name = f"{annotation.enterprise}/{annotation_file}"
            key = (
                _identity_key(annotation.enterprise),
                _document_filename_key(annotation_file),
            )
            matches = entries_by_key.get(key, ())
            if not matches:
                missing_files.add(display_name)
                unresolved += 1
                continue
            logical_ids = {
                logical_id_by_file_id[entry.id] for entry in matches
            }
            formats = [normalize_file_format(entry.path.suffix) for entry in matches]
            if len(logical_ids) != 1 or len(formats) != len(set(formats)):
                ambiguous_files.add(display_name)
                unresolved += 1
                continue
            variants = tuple(
                sorted(
                    (
                        IndexedDocumentVariant(
                            file_id=entry.id,
                            file_format=normalize_file_format(entry.path.suffix),
                            path=str(entry.path),
                        )
                        for entry in matches
                    ),
                    key=lambda item: _document_format_order(item.file_format),
                )
            )
            available_formats = {variant.file_format for variant in variants}
            missing_formats = tuple(
                file_format
                for file_format in DEFAULT_DOCUMENT_FORMATS
                if file_format not in available_formats
            )
            if missing_formats:
                missing_format_variants[display_name] = missing_formats
            relevant_documents.append(
                RelevantDocument(
                    logical_id=next(iter(logical_ids)),
                    annotation_file=annotation_file,
                    variants=variants,
                )
            )

        if not relevant_documents:
            skipped += 1
            continue
        if unresolved:
            partially_covered += 1
        else:
            fully_covered += 1
        eligible.append(
            EligibleDocumentQuery(
                enterprise=annotation.enterprise,
                query_type=annotation.query_type,
                query=annotation.query,
                relevant_documents=tuple(relevant_documents),
            )
        )

    return DocumentAnnotationCoverage(
        annotation_rows=sum(item.source_rows for item in annotations),
        annotation_queries=len(annotations),
        annotation_files=len(annotation_file_keys),
        indexed_target_files=len(target_entries),
        indexed_logical_files=len(set(logical_id_by_file_id.values())),
        indexed_formats=dict(sorted(indexed_formats.items())),
        eligible_queries=tuple(eligible),
        fully_covered_queries=fully_covered,
        partially_covered_queries=partially_covered,
        skipped_queries=skipped,
        missing_files=tuple(sorted(missing_files)),
        missing_format_variants=dict(sorted(missing_format_variants.items())),
        ambiguous_files=tuple(sorted(ambiguous_files)),
        logical_id_by_file_id=logical_id_by_file_id,
    )


def document_queries_for_view(
    queries: Sequence[EligibleDocumentQuery],
    *,
    file_format: str | None,
) -> tuple[DocumentViewQuery, ...]:
    """Select available logical relevance judgments for one search view."""
    prepared_format = (
        normalize_file_format(file_format) if file_format is not None else None
    )
    selected: list[DocumentViewQuery] = []
    for query in queries:
        relevant_ids = tuple(
            document.logical_id
            for document in query.relevant_documents
            if prepared_format is None
            or any(
                variant.file_format == prepared_format
                for variant in document.variants
            )
        )
        if not relevant_ids:
            continue
        selected.append(
            DocumentViewQuery(
                enterprise=query.enterprise,
                query_type=query.query_type,
                query=query.query,
                relevant_logical_ids=relevant_ids,
            )
        )
    return tuple(selected)


def evaluate_queries(
    file_index: FileIndexRepository,
    embedding: TextEmbedding[Any],
    queries: Sequence[EligibleQuery],
    *,
    target: str,
    cutoffs: Sequence[int],
    candidate_limit: int,
    batch_size: int,
    clock: Callable[[], float] = perf_counter,
    on_progress: Callable[[int], None] | None = None,
) -> EvaluationResult:
    """Evaluate all ranks with batch inference and sequential retrieval."""
    prepared_cutoffs = _validate_cutoffs(cutoffs)
    if candidate_limit < max(prepared_cutoffs):
        raise ValueError("Candidate limit must be at least the largest cutoff")
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if not queries:
        raise ValueError("No eligible annotation queries were found in the index")

    search = _search_for_target(file_index, target)

    ranks: list[int | None] = []
    embedding_ms = 0.0
    retrieval_ms = 0.0
    maximum_cutoff = max(prepared_cutoffs)
    total_started = clock()
    for batch in _batches(queries, batch_size):
        embedding_started = clock()
        vectors = embedding.predict_text_batch(
            tuple(query.query for query in batch)
        )
        embedding_ms += (clock() - embedding_started) * 1000
        if len(vectors) != len(batch):
            raise ValueError(
                "Batch inference result count must match the query count"
            )
        for query, vector in zip(batch, vectors):
            try:
                retrieval_started = clock()
                raw_results = search(
                    vector,
                    vector_name=embedding.vector_name,
                    limit=candidate_limit,
                )
                retrieval_ms += (clock() - retrieval_started) * 1000
                ranked_results = unique_file_results(raw_results)[:maximum_cutoff]
                ranks.append(relevant_rank(ranked_results, query.relevant_id))
            finally:
                if on_progress is not None:
                    on_progress(1)
    total_ms = (clock() - total_started) * 1000
    query_count = len(queries)

    return EvaluationResult(
        ranks=tuple(ranks),
        metrics=calculate_retrieval_metrics(ranks, prepared_cutoffs),
        throughput=EvaluationThroughput(
            batch_size=batch_size,
            total_ms=total_ms,
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            queries_per_second=query_count * 1000 / total_ms,
            amortized_ms_per_query=total_ms / query_count,
        ),
    )


def embed_document_queries(
    embedding: TextEmbedding[Any],
    queries: Sequence[DocumentViewQuery],
    *,
    batch_size: int,
    clock: Callable[[], float] = perf_counter,
    on_progress: Callable[[int], None] | None = None,
) -> DocumentEmbeddingCache:
    """Embed each consolidated document query once for all retrieval views."""
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if not queries:
        raise ValueError("No eligible document queries were found in the index")

    vectors_by_query: dict[str, EmbeddingVector] = {}
    started = clock()
    for batch in _batches(queries, batch_size):
        vectors = embedding.predict_text_batch(
            tuple(query.query for query in batch)
        )
        if len(vectors) != len(batch):
            raise ValueError(
                "Batch inference result count must match the query count"
            )
        for query, vector in zip(batch, vectors):
            query_id = _document_query_id(query.enterprise, query.query)
            if query_id in vectors_by_query:
                raise ValueError(
                    "Document evaluation queries must be unique by enterprise "
                    "and query text"
                )
            vectors_by_query[query_id] = vector
            if on_progress is not None:
                on_progress(1)
    return DocumentEmbeddingCache(
        vectors_by_query=vectors_by_query,
        embedding_ms=(clock() - started) * 1000,
    )


def evaluate_document_queries(
    file_index: FileIndexRepository,
    queries: Sequence[DocumentViewQuery],
    *,
    embedding_cache: DocumentEmbeddingCache,
    vector_name: str,
    logical_id_by_file_id: Mapping[str, str],
    file_format: str | None,
    cutoffs: Sequence[int],
    candidate_limit: int,
    batch_size: int,
    clock: Callable[[], float] = perf_counter,
    on_progress: Callable[[int], None] | None = None,
) -> DocumentEvaluationResult:
    """Evaluate chunk retrieval with physical-file and logical-file judgments."""
    prepared_cutoffs = _validate_cutoffs(cutoffs)
    if candidate_limit < max(prepared_cutoffs):
        raise ValueError("Candidate limit must be at least the largest cutoff")
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if not queries:
        raise ValueError("No eligible document queries were found in the index")

    conditions = [MetadataCondition("file_type", "text")]
    if file_format is not None:
        conditions.append(
            MetadataCondition(
                "format_key", normalize_file_format(file_format)
            )
        )
    metadata_filter = MetadataFilter(all_of=tuple(conditions))

    outcomes: list[DocumentQueryOutcome] = []
    retrieval_ms = 0.0
    maximum_cutoff = max(prepared_cutoffs)
    retrieval_pass_started = clock()
    for query in queries:
        try:
            query_id = _document_query_id(query.enterprise, query.query)
            vector = embedding_cache.vectors_by_query.get(query_id)
            if vector is None:
                raise ValueError(
                    f"No cached embedding exists for document query: {query.query}"
                )
            retrieval_started = clock()
            raw_results = file_index.semantic_segment_search(
                vector,
                vector_name=vector_name,
                limit=candidate_limit,
                metadata_filter=metadata_filter,
            )
            retrieval_ms += (clock() - retrieval_started) * 1000
            physical_results = unique_file_results(raw_results)[
                :maximum_cutoff
            ]
            outcomes.append(
                DocumentQueryOutcome(
                    enterprise=query.enterprise,
                    query_type=query.query_type,
                    query=query.query,
                    relevant_count=len(query.relevant_logical_ids),
                    relevant_logical_ids=query.relevant_logical_ids,
                    relevant_ranks=document_relevant_ranks(
                        physical_results,
                        query.relevant_logical_ids,
                        logical_id_by_file_id,
                    ),
                    returned_physical_files=len(physical_results),
                )
            )
        finally:
            if on_progress is not None:
                on_progress(1)
    retrieval_pass_ms = (clock() - retrieval_pass_started) * 1000
    total_ms = embedding_cache.embedding_ms + retrieval_pass_ms

    grouped_outcomes: dict[str, list[DocumentQueryOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped_outcomes[outcome.query_type].append(outcome)
    metrics_by_query_type = {
        query_type: calculate_document_retrieval_metrics(
            query_outcomes, prepared_cutoffs
        )
        for query_type, query_outcomes in sorted(grouped_outcomes.items())
    }
    official_type_metrics = tuple(
        metrics_by_query_type[query_type]
        for query_type in DOCUMENT_QUERY_TYPES
        if query_type in metrics_by_query_type
    )
    macro_type_metrics = official_type_metrics or tuple(
        metrics_by_query_type.values()
    )
    multi_document_outcomes = tuple(
        outcome for outcome in outcomes if outcome.relevant_count > 1
    )
    query_count = len(queries)
    return DocumentEvaluationResult(
        outcomes=tuple(outcomes),
        metrics=calculate_document_retrieval_metrics(
            outcomes, prepared_cutoffs
        ),
        metrics_by_query_type=metrics_by_query_type,
        macro_query_type_metrics=_mean_document_metric_groups(
            macro_type_metrics
        ),
        multi_document_metrics=(
            calculate_document_retrieval_metrics(
                multi_document_outcomes, prepared_cutoffs
            )
            if multi_document_outcomes
            else None
        ),
        throughput=EvaluationThroughput(
            batch_size=batch_size,
            total_ms=total_ms,
            embedding_ms=embedding_cache.embedding_ms,
            retrieval_ms=retrieval_ms,
            queries_per_second=query_count * 1000 / total_ms,
            amortized_ms_per_query=total_ms / query_count,
        ),
    )


def document_relevant_ranks(
    results: Sequence[FileSearchResult],
    relevant_logical_ids: Sequence[str],
    logical_id_by_file_id: Mapping[str, str],
) -> tuple[int, ...]:
    """Return first physical result ranks for distinct relevant documents."""
    relevant = set(relevant_logical_ids)
    credited: set[str] = set()
    ranks: list[int] = []
    for rank, result in enumerate(results, start=1):
        logical_id = logical_id_by_file_id.get(result.file.id)
        if logical_id not in relevant or logical_id in credited:
            continue
        assert logical_id is not None
        credited.add(logical_id)
        ranks.append(rank)
    return tuple(ranks)


def measure_query_latency(
    file_index: FileIndexRepository,
    embedding: TextEmbedding[Any],
    queries: Sequence[EligibleQuery | DocumentViewQuery],
    *,
    target: str,
    candidate_limit: int,
    warmup_queries: int,
    clock: Callable[[], float] = perf_counter,
    on_progress: Callable[[int], None] | None = None,
) -> LatencyMetrics:
    """Measure the production-style one-query-at-a-time search path."""
    if not queries:
        raise ValueError("No latency queries were selected")
    if warmup_queries < 0:
        raise ValueError("Latency warmup count must not be negative")
    search = _search_for_target(file_index, target)

    for query in queries[:warmup_queries]:
        vector = embedding.predict_text(query.query)
        search(
            vector,
            vector_name=embedding.vector_name,
            limit=candidate_limit,
        )

    query_times_ms: list[float] = []
    for query in queries:
        try:
            started = clock()
            vector = embedding.predict_text(query.query)
            search(
                vector,
                vector_name=embedding.vector_name,
                limit=candidate_limit,
            )
            query_times_ms.append((clock() - started) * 1000)
        finally:
            if on_progress is not None:
                on_progress(1)
    return calculate_latency_metrics(
        query_times_ms,
        warmup_queries=min(warmup_queries, len(queries)),
    )


def select_latency_queries(
    queries: Sequence[T],
    *,
    sample_size: int,
    seed: int,
) -> tuple[T, ...]:
    """Select a reproducible latency sample without replacement."""
    if sample_size < 1:
        raise ValueError("Latency sample size must be positive")
    if sample_size >= len(queries):
        return tuple(queries)
    indexes = sorted(random.Random(seed).sample(range(len(queries)), sample_size))
    return tuple(queries[index] for index in indexes)


def unique_file_results(
    results: Sequence[FileSearchResult],
) -> tuple[FileSearchResult, ...]:
    """Keep the highest-ranked result for each parent file."""
    seen: set[str] = set()
    unique: list[FileSearchResult] = []
    for result in results:
        if result.file.id in seen:
            continue
        seen.add(result.file.id)
        unique.append(result)
    return tuple(unique)


def relevant_rank(
    results: Sequence[FileSearchResult], relevant_id: str
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.file.id == relevant_id:
            return rank
    return None


def calculate_retrieval_metrics(
    ranks: Sequence[int | None], cutoffs: Sequence[int]
) -> tuple[RetrievalMetrics, ...]:
    if not ranks:
        raise ValueError("Cannot calculate retrieval metrics without queries")
    prepared_cutoffs = _validate_cutoffs(cutoffs)
    query_count = len(ranks)
    metrics: list[RetrievalMetrics] = []
    for cutoff in prepared_cutoffs:
        reciprocal_ranks = [
            1.0 / rank if rank is not None and rank <= cutoff else 0.0
            for rank in ranks
        ]
        hits = [
            1.0 if rank is not None and rank <= cutoff else 0.0
            for rank in ranks
        ]
        discounted_gains = [
            1.0 / math.log2(rank + 1)
            if rank is not None and rank <= cutoff
            else 0.0
            for rank in ranks
        ]
        metrics.append(
            RetrievalMetrics(
                cutoff=cutoff,
                mrr=sum(reciprocal_ranks) / query_count,
                recall=sum(hits) / query_count,
                ndcg=sum(discounted_gains) / query_count,
            )
        )
    return tuple(metrics)


def calculate_document_retrieval_metrics(
    outcomes: Sequence[DocumentQueryOutcome],
    cutoffs: Sequence[int],
) -> tuple[DocumentRetrievalMetrics, ...]:
    """Calculate binary multi-relevance metrics at every cutoff."""
    if not outcomes:
        raise ValueError(
            "Cannot calculate document retrieval metrics without queries"
        )
    if any(outcome.relevant_count < 1 for outcome in outcomes):
        raise ValueError("Every document query requires a relevant document")
    prepared_cutoffs = _validate_cutoffs(cutoffs)
    metrics: list[DocumentRetrievalMetrics] = []
    for cutoff in prepared_cutoffs:
        hit_values: list[float] = []
        reciprocal_ranks: list[float] = []
        recall_values: list[float] = []
        ndcg_values: list[float] = []
        average_precision_values: list[float] = []
        complete_values: list[float] = []
        for outcome in outcomes:
            ranks = tuple(
                sorted(
                    rank
                    for rank in set(outcome.relevant_ranks)
                    if rank <= cutoff
                )
            )
            hit_values.append(1.0 if ranks else 0.0)
            reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
            recall_values.append(len(ranks) / outcome.relevant_count)
            discounted_gain = sum(
                1.0 / math.log2(rank + 1) for rank in ranks
            )
            ideal_gain = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(
                    1, min(outcome.relevant_count, cutoff) + 1
                )
            )
            ndcg_values.append(discounted_gain / ideal_gain)
            average_precision_values.append(
                sum(
                    relevant_seen / rank
                    for relevant_seen, rank in enumerate(ranks, start=1)
                )
                / min(outcome.relevant_count, cutoff)
            )
            complete_values.append(
                1.0 if len(ranks) == outcome.relevant_count else 0.0
            )
        metrics.append(
            DocumentRetrievalMetrics(
                cutoff=cutoff,
                hit=fmean(hit_values),
                mrr=fmean(reciprocal_ranks),
                recall=fmean(recall_values),
                ndcg=fmean(ndcg_values),
                map=fmean(average_precision_values),
                complete=fmean(complete_values),
            )
        )
    return tuple(metrics)


def _mean_document_metric_groups(
    groups: Sequence[Sequence[DocumentRetrievalMetrics]],
) -> tuple[DocumentRetrievalMetrics, ...]:
    """Calculate an equal-weight macro average over query-type metrics."""
    if not groups:
        raise ValueError("Cannot calculate a macro average without groups")
    metric_count = len(groups[0])
    if any(len(group) != metric_count for group in groups):
        raise ValueError("Metric groups must use the same cutoffs")
    return tuple(
        DocumentRetrievalMetrics(
            cutoff=groups[0][index].cutoff,
            hit=fmean(group[index].hit for group in groups),
            mrr=fmean(group[index].mrr for group in groups),
            recall=fmean(group[index].recall for group in groups),
            ndcg=fmean(group[index].ndcg for group in groups),
            map=fmean(group[index].map for group in groups),
            complete=fmean(group[index].complete for group in groups),
        )
        for index in range(metric_count)
    )


def calculate_latency_metrics(
    query_times_ms: Sequence[float],
    *,
    warmup_queries: int = 0,
) -> LatencyMetrics:
    if not query_times_ms:
        raise ValueError("Cannot calculate latency metrics without queries")
    return LatencyMetrics(
        samples=len(query_times_ms),
        warmup_queries=warmup_queries,
        average_ms=fmean(query_times_ms),
        p95_ms=percentile(query_times_ms, 0.95),
        p99_ms=percentile(query_times_ms, 0.99),
    )


def percentile(values: Sequence[float], probability: float) -> float:
    """Calculate a linearly interpolated percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    )


def _search_for_target(
    file_index: FileIndexRepository, target: str
) -> Callable[..., tuple[FileSearchResult, ...]]:
    search = {
        "image": file_index.semantic_search,
        "audio": file_index.semantic_segment_search,
        "document": file_index.semantic_segment_search,
    }.get(target)
    if search is None:
        raise ValueError(f"Unsupported retrieval target: {target}")
    return search


def _batches(
    items: Sequence[T], batch_size: int
) -> tuple[Sequence[T], ...]:
    return tuple(
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FileLore text-to-image, text-to-audio, or "
            "text-to-document retrieval."
        )
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="COCO JSON, Clotho CSV, or document supporting-facts CSV",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=("image", "audio", "document"),
        help="select the indexed target and its embedding model",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        metavar="N",
        help="metric cutoffs (default: 1 5 10)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        help=(
            "raw results requested per query; audio defaults to max(100, "
            "10*k) and documents to max(200, 20*k) for parent-file "
            "deduplication"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=(
            "text queries embedded together during the full metric pass "
            f"(default: {DEFAULT_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=DEFAULT_LATENCY_SAMPLES,
        metavar="N",
        help=(
            "single-query searches measured for latency "
            f"(default: {DEFAULT_LATENCY_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=DEFAULT_LATENCY_WARMUP,
        metavar="N",
        help=(
            "unmeasured sequential warm-up queries "
            f"(default: {DEFAULT_LATENCY_WARMUP})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "random seed for latency sampling "
            f"(default: {DEFAULT_RANDOM_SEED})"
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="evaluate only the first N eligible queries",
    )
    parser.add_argument(
        "--document-workers",
        type=int,
        default=DEFAULT_DOCUMENT_VIEW_WORKERS,
        metavar="N",
        help=(
            "concurrent mixed/per-format document retrieval views "
            f"(default: {DEFAULT_DOCUMENT_VIEW_WORKERS})"
        ),
    )
    parser.add_argument(
        "--model",
        help="embedding model id (default: the target's FileLore model)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("FILELORE_QDRANT_URL"),
        help="use a Qdrant service instead of Python Local Mode",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"local Qdrant index directory (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON result file "
            "(default: evaluation/results/<timestamp>-<target>.json)"
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.annotations = args.annotations.expanduser()
    if not args.annotations.is_file():
        raise ValueError(f"Annotation file does not exist: {args.annotations}")
    args.k = list(_validate_cutoffs(args.k))
    if args.batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if args.latency_samples < 1:
        raise ValueError("Latency sample size must be positive")
    if args.latency_warmup < 0:
        raise ValueError("Latency warmup count must not be negative")
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("Maximum query count must be positive")
    if args.document_workers < 1:
        raise ValueError("Document view worker count must be positive")
    if args.candidate_limit is not None and args.candidate_limit < max(args.k):
        raise ValueError("Candidate limit must be at least the largest cutoff")
    if args.qdrant_url is None:
        args.index_path = args.index_path.expanduser()
        if not args.index_path.is_dir():
            raise ValueError(f"Index directory does not exist: {args.index_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    display = CliDisplay()
    try:
        validate_args(args)
        output_path = args.output or default_result_path(args.target)
        if args.target == "document":
            return _run_document_evaluation(
                args=args,
                display=display,
                output_path=output_path,
            )
        with display.status("Reading annotations…"):
            annotations = load_annotations(args.annotations, args.target)
        annotation_file_count = len(
            {_filename_key(item.relevant_file) for item in annotations}
        )
        display.print_info(
            f"Annotations: {len(annotations):,} queries  •  "
            f"{annotation_file_count:,} files"
        )
        maximum_cutoff = max(args.k)
        candidate_limit = args.candidate_limit or (
            maximum_cutoff
            if args.target == "image"
            else max(100, maximum_cutoff * 10)
        )

        with QdrantVectorDatabase(
            path=args.index_path,
            url=args.qdrant_url,
        ) as database:
            if not database.collection_exists("files"):
                raise ValueError("The Qdrant 'files' collection does not exist")
            file_index = FileIndexRepository(database)
            with display.status("Checking indexed files…"):
                coverage = filter_indexed_annotations(
                    annotations,
                    tuple(file_index.iter_all()),
                    target=args.target,
                )
            queries = coverage.eligible_queries
            if args.max_queries is not None:
                queries = queries[: args.max_queries]
            if not queries:
                raise ValueError(
                    "None of the annotated files exist uniquely in the index"
                )
            display.print_info(
                f"Evaluation set: {len(queries):,} queries  •  "
                f"{coverage.eligible_files:,} files  •  "
                f"{len(coverage.missing_files):,} missing  •  "
                f"{len(coverage.ambiguous_files):,} ambiguous"
            )

            with display.status(f"Initializing {args.target} model…"):
                model_started = perf_counter()
                embedding = _create_embedding(
                    args.target,
                    model_id=args.model,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                model_load_ms = (perf_counter() - model_started) * 1000
            try:
                latency_queries = select_latency_queries(
                    queries,
                    sample_size=args.latency_samples,
                    seed=args.seed,
                )
                display.print_info(
                    f"Latency sample: {len(latency_queries):,} sequential "
                    f"queries  •  {min(args.latency_warmup, len(latency_queries)):,} "
                    "warm-up queries excluded"
                )
                with display.indexing(
                    len(latency_queries),
                    label="Measuring query latency",
                    rate_unit="queries",
                ) as progress:
                    latency = measure_query_latency(
                        file_index,
                        embedding,
                        latency_queries,
                        target=args.target,
                        candidate_limit=candidate_limit,
                        warmup_queries=args.latency_warmup,
                        on_progress=progress.advance,
                    )

                display.print_info(
                    f"Metric pass: batch inference ({args.batch_size})  •  "
                    "sequential retrieval"
                )
                with display.indexing(
                    len(queries),
                    label="Evaluating retrieval",
                    rate_unit="queries",
                ) as progress:
                    result = evaluate_queries(
                        file_index,
                        embedding,
                        queries,
                        target=args.target,
                        cutoffs=args.k,
                        candidate_limit=candidate_limit,
                        batch_size=args.batch_size,
                        on_progress=progress.advance,
                    )
            finally:
                embedding.close()

        summary = _summary(
            args=args,
            coverage=coverage,
            evaluated_queries=len(queries),
            candidate_limit=candidate_limit,
            model_load_ms=model_load_ms,
            embedding=embedding,
            result=result,
            latency=latency,
        )
        print_summary(summary)
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        display.print_info(f"Result saved: {output_path.resolve()}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


def _run_document_evaluation(
    *,
    args: argparse.Namespace,
    display: CliDisplay,
    output_path: Path,
) -> int:
    with display.status("Reading document annotations…"):
        annotations = parse_document_annotations(args.annotations)
    display.print_info(
        f"Annotations: {sum(item.source_rows for item in annotations):,} rows  •  "
        f"{len(annotations):,} consolidated queries"
    )
    maximum_cutoff = max(args.k)
    candidate_limit = args.candidate_limit or max(
        200, maximum_cutoff * 20
    )

    with QdrantVectorDatabase(
        path=args.index_path,
        url=args.qdrant_url,
    ) as database:
        if not database.collection_exists("files"):
            raise ValueError("The Qdrant 'files' collection does not exist")
        if not database.collection_exists("files_segments"):
            raise ValueError(
                "The Qdrant 'files_segments' collection does not exist"
            )
        file_index = FileIndexRepository(database)
        with display.status("Checking indexed documents…"):
            coverage = filter_indexed_document_annotations(
                annotations,
                tuple(file_index.iter_all()),
            )
        eligible_queries = coverage.eligible_queries
        if args.max_queries is not None:
            eligible_queries = eligible_queries[: args.max_queries]
        if not eligible_queries:
            raise ValueError(
                "None of the document queries have a relevant indexed file"
            )
        display.print_info(
            f"Evaluation set: {len(eligible_queries):,} queries  •  "
            f"{coverage.fully_covered_queries:,} fully covered  •  "
            f"{coverage.partially_covered_queries:,} partially covered  •  "
            f"{coverage.skipped_queries:,} skipped"
        )
        display.print_info(
            f"Indexed documents: {coverage.indexed_target_files:,} files  •  "
            f"{coverage.indexed_logical_files:,} logical documents  •  "
            f"{len(coverage.missing_files):,} missing citations  •  "
            f"{len(coverage.missing_format_variants):,} incomplete variants  •  "
            f"{len(coverage.ambiguous_files):,} ambiguous citations"
        )

        with display.status("Initializing document model…"):
            model_started = perf_counter()
            embedding = _create_embedding(
                "document",
                model_id=args.model,
                device=args.device,
                batch_size=args.batch_size,
            )
            model_load_ms = (perf_counter() - model_started) * 1000
        try:
            mixed_queries = document_queries_for_view(
                eligible_queries,
                file_format=None,
            )
            latency_queries = select_latency_queries(
                mixed_queries,
                sample_size=args.latency_samples,
                seed=args.seed,
            )
            display.print_info(
                f"Latency sample: {len(latency_queries):,} sequential "
                f"queries  •  {min(args.latency_warmup, len(latency_queries)):,} "
                "warm-up queries excluded"
            )
            with display.indexing(
                len(latency_queries),
                label="Measuring document query latency",
                rate_unit="queries",
            ) as progress:
                latency = measure_query_latency(
                    file_index,
                    embedding,
                    latency_queries,
                    target="document",
                    candidate_limit=candidate_limit,
                    warmup_queries=args.latency_warmup,
                    on_progress=progress.advance,
                )

            metric_pass_started = perf_counter()
            with display.indexing(
                len(mixed_queries),
                label="Embedding document queries",
                rate_unit="queries",
            ) as progress:
                embedding_cache = embed_document_queries(
                    embedding,
                    mixed_queries,
                    batch_size=args.batch_size,
                    on_progress=progress.advance,
                )

            views = (("mixed", None),) + tuple(
                (file_format, file_format)
                for file_format in DEFAULT_DOCUMENT_FORMATS
            )
            prepared_views: list[
                tuple[str, str | None, tuple[DocumentViewQuery, ...]]
            ] = []
            for view_name, file_format in views:
                view_queries = document_queries_for_view(
                    eligible_queries,
                    file_format=file_format,
                )
                if view_queries:
                    prepared_views.append(
                        (view_name, file_format, view_queries)
                    )

            worker_count = min(args.document_workers, len(prepared_views))
            display.print_info(
                f"Metric retrieval: {len(prepared_views)} views  •  "
                f"{worker_count} parallel workers  •  cached query embeddings"
            )
            completed_views: dict[str, DocumentEvaluationResult] = {}
            progress_lock = Lock()
            with display.indexing(
                sum(len(view_queries) for _, _, view_queries in prepared_views),
                label="Evaluating document retrieval views",
                rate_unit="query views",
            ) as progress:

                def advance_retrieval_progress(amount: int) -> None:
                    with progress_lock:
                        progress.advance(amount)

                parallel_retrieval_started = perf_counter()
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="document-evaluation",
                ) as executor:
                    futures = {
                        executor.submit(
                            evaluate_document_queries,
                            file_index,
                            view_queries,
                            embedding_cache=embedding_cache,
                            vector_name=embedding.vector_name,
                            logical_id_by_file_id=(
                                coverage.logical_id_by_file_id
                            ),
                            file_format=file_format,
                            cutoffs=args.k,
                            candidate_limit=candidate_limit,
                            batch_size=args.batch_size,
                            on_progress=advance_retrieval_progress,
                        ): view_name
                        for view_name, file_format, view_queries in prepared_views
                    }
                    for future in as_completed(futures):
                        view_name = futures[future]
                        completed_views[view_name] = future.result()
                parallel_retrieval_ms = (
                    perf_counter() - parallel_retrieval_started
                ) * 1000

            view_results = {
                view_name: completed_views[view_name]
                for view_name, _, _ in prepared_views
            }
            metric_total_ms = (perf_counter() - metric_pass_started) * 1000
            view_evaluations = sum(
                len(result.outcomes) for result in view_results.values()
            )
            metric_timing = DocumentMetricPassTiming(
                workers=worker_count,
                query_count=len(mixed_queries),
                view_evaluations=view_evaluations,
                embedding_ms=embedding_cache.embedding_ms,
                parallel_retrieval_ms=parallel_retrieval_ms,
                total_ms=metric_total_ms,
                queries_per_second=(
                    len(mixed_queries) * 1000 / metric_total_ms
                ),
                view_evaluations_per_second=(
                    view_evaluations * 1000 / metric_total_ms
                ),
            )
        finally:
            embedding.close()

    summary = _document_summary(
        args=args,
        coverage=coverage,
        evaluated_queries=len(eligible_queries),
        candidate_limit=candidate_limit,
        model_load_ms=model_load_ms,
        embedding=embedding,
        view_results=view_results,
        metric_timing=metric_timing,
        latency=latency,
    )
    print_document_summary(summary)
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    display.print_info(f"Result saved: {output_path.resolve()}")
    return 0


def default_result_path(target: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RESULTS_DIRECTORY / f"{timestamp}-{target}.json"


def print_summary(summary: Mapping[str, Any]) -> None:
    coverage = summary["coverage"]
    latency = summary["query_latency_ms"]
    throughput = summary["throughput"]
    print(f"Target: {summary['target']}")
    print(f"Model: {summary['model_id']}")
    print(f"Vector: {summary['vector_name']}")
    print(
        "Coverage: "
        f"{coverage['eligible_files']}/{coverage['annotation_files']} files, "
        f"{summary['evaluated_queries']}/{coverage['annotation_queries']} queries"
    )
    print(
        f"Excluded files: {len(coverage['missing_files'])} missing, "
        f"{len(coverage['ambiguous_files'])} ambiguous"
    )
    print(f"Raw candidate limit: {summary['candidate_limit']}")
    print("\nRetrieval")
    print("k\tMRR\tRecall\tNDCG")
    for metric in summary["metrics"]:
        print(
            f"{metric['cutoff']}\t{metric['mrr']:.6f}\t"
            f"{metric['recall']:.6f}\t{metric['ndcg']:.6f}"
        )
    print("\nFull metric-pass throughput")
    print(
        f"Mode: batch inference ({throughput['batch_size']}) + "
        "sequential retrieval"
    )
    print(f"Total: {throughput['total_ms'] / 1000:.3f} s")
    print(f"Embedding: {throughput['embedding_ms'] / 1000:.3f} s")
    print(f"Retrieval: {throughput['retrieval_ms'] / 1000:.3f} s")
    print(f"Throughput: {throughput['queries_per_second']:.3f} queries/s")
    print(
        "Amortized: "
        f"{throughput['amortized_ms_per_query']:.3f} ms/query"
    )
    print("\nSingle-query latency (sequential embedding + retrieval)")
    print(
        f"Samples: {latency['samples']}  •  "
        f"warm-up excluded: {latency['warmup_queries']}"
    )
    print(f"Average: {latency['average_ms']:.3f} ms")
    print(f"P95: {latency['p95_ms']:.3f} ms")
    print(f"P99: {latency['p99_ms']:.3f} ms")
    print(f"Model load: {summary['model_load_ms']:.3f} ms (excluded)")


def print_document_summary(summary: Mapping[str, Any]) -> None:
    coverage = summary["coverage"]
    print(f"Target: {summary['target']}")
    print(f"Model: {summary['model_id']}")
    print(f"Vector: {summary['vector_name']}")
    print(
        "Coverage: "
        f"{summary['evaluated_queries']}/{coverage['annotation_queries']} "
        "queries evaluated, "
        f"{coverage['partially_covered_queries']} partial, "
        f"{coverage['skipped_queries']} skipped"
    )
    print(
        f"Missing citations: {len(coverage['missing_files'])}; "
        "incomplete format variants: "
        f"{len(coverage['missing_format_variants'])}; "
        f"ambiguous citations: {len(coverage['ambiguous_files'])}"
    )
    print(f"Raw candidate limit: {summary['candidate_limit']}")

    for view_name, view in summary["views"].items():
        print(
            f"\nRetrieval: {view_name} "
            f"({view['evaluated_queries']} queries)"
        )
        print("k\tHit\tMRR\tRecall\tNDCG\tMAP\tComplete")
        for metric in view["metrics"]:
            print(
                f"{metric['cutoff']}\t{metric['hit']:.6f}\t"
                f"{metric['mrr']:.6f}\t{metric['recall']:.6f}\t"
                f"{metric['ndcg']:.6f}\t{metric['map']:.6f}\t"
                f"{metric['complete']:.6f}"
            )

    mixed = summary["views"].get("mixed")
    if mixed is not None:
        print("\nMixed retrieval by query type at maximum k")
        print("Type\tQueries\tHit\tMRR\tRecall\tNDCG\tMAP\tComplete")
        for query_type, metrics in mixed["metrics_by_query_type"].items():
            metric = metrics[-1]
            print(
                f"{query_type}\t{mixed['query_type_counts'][query_type]}\t"
                f"{metric['hit']:.6f}\t{metric['mrr']:.6f}\t"
                f"{metric['recall']:.6f}\t{metric['ndcg']:.6f}\t"
                f"{metric['map']:.6f}\t{metric['complete']:.6f}"
            )
        macro = mixed["macro_query_type_metrics"][-1]
        print(
            f"Official-type macro\t-\t{macro['hit']:.6f}\t"
            f"{macro['mrr']:.6f}\t{macro['recall']:.6f}\t"
            f"{macro['ndcg']:.6f}\t{macro['map']:.6f}\t"
            f"{macro['complete']:.6f}"
        )

    metric_pass = summary["metric_pass"]
    print("\nFull document metric pass")
    print(
        f"Mode: one shared batch-embedding pass + "
        f"{metric_pass['workers']} parallel retrieval views"
    )
    print(f"Total: {metric_pass['total_ms'] / 1000:.3f} s")
    print(f"Embedding: {metric_pass['embedding_ms'] / 1000:.3f} s")
    print(
        "Parallel retrieval wall time: "
        f"{metric_pass['parallel_retrieval_ms'] / 1000:.3f} s"
    )
    print(
        f"Completed: {metric_pass['queries_per_second']:.3f} "
        "full query evaluations/s"
    )
    print(
        f"View throughput: {metric_pass['view_evaluations_per_second']:.3f} "
        "query views/s"
    )

    latency = summary["query_latency_ms"]
    print("\nMixed single-query latency (embedding + retrieval)")
    print(
        f"Samples: {latency['samples']}  •  "
        f"warm-up excluded: {latency['warmup_queries']}"
    )
    print(f"Average: {latency['average_ms']:.3f} ms")
    print(f"P95: {latency['p95_ms']:.3f} ms")
    print(f"P99: {latency['p99_ms']:.3f} ms")
    print(f"Model load: {summary['model_load_ms']:.3f} ms (excluded)")


def _document_summary(
    *,
    args: argparse.Namespace,
    coverage: DocumentAnnotationCoverage,
    evaluated_queries: int,
    candidate_limit: int,
    model_load_ms: float,
    embedding: TextEmbedding[Any],
    view_results: Mapping[str, DocumentEvaluationResult],
    metric_timing: DocumentMetricPassTiming,
    latency: LatencyMetrics,
) -> dict[str, Any]:
    return {
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": calculate_file_hash(args.annotations),
        "target": "document",
        "model_id": embedding.model_id,
        "vector_name": embedding.vector_name,
        "cutoffs": list(args.k),
        "candidate_limit": candidate_limit,
        "evaluated_queries": evaluated_queries,
        "document_formats": list(DEFAULT_DOCUMENT_FORMATS),
        "coverage": coverage.to_dict(),
        "views": {
            view_name: _document_view_summary(result)
            for view_name, result in view_results.items()
        },
        "metric_pass": {
            "mode": "shared_batch_embedding_parallel_view_retrieval",
            **asdict(metric_timing),
        },
        "query_latency_ms": {
            "view": "mixed",
            "mode": "sequential_embedding_sequential_retrieval",
            "sample_seed": args.seed,
            **asdict(latency),
        },
        "model_load_ms": model_load_ms,
    }


def _document_view_summary(
    result: DocumentEvaluationResult,
) -> dict[str, Any]:
    query_type_counts: dict[str, int] = defaultdict(int)
    for outcome in result.outcomes:
        query_type_counts[outcome.query_type] += 1
    macro_query_types = [
        query_type
        for query_type in DOCUMENT_QUERY_TYPES
        if query_type in result.metrics_by_query_type
    ] or list(result.metrics_by_query_type)
    maximum_cutoff = max(metric.cutoff for metric in result.metrics)
    return {
        "evaluated_queries": len(result.outcomes),
        "query_type_counts": dict(sorted(query_type_counts.items())),
        "metrics": [asdict(metric) for metric in result.metrics],
        "aggregation": "query_weighted_micro_average",
        "metrics_by_query_type": {
            query_type: [asdict(metric) for metric in metrics]
            for query_type, metrics in result.metrics_by_query_type.items()
        },
        "macro_query_types": macro_query_types,
        "macro_query_type_metrics": [
            asdict(metric) for metric in result.macro_query_type_metrics
        ],
        "multi_document_query_count": sum(
            outcome.relevant_count > 1 for outcome in result.outcomes
        ),
        "underfilled_queries_at_max_cutoff": sum(
            outcome.returned_physical_files < maximum_cutoff
            for outcome in result.outcomes
        ),
        "multi_document_metrics": (
            [asdict(metric) for metric in result.multi_document_metrics]
            if result.multi_document_metrics is not None
            else None
        ),
        "throughput": {
            "mode": "shared_batch_embedding_parallel_view_retrieval",
            "embedding_reused_across_views": True,
            **asdict(result.throughput),
        },
        "query_outcomes": [asdict(outcome) for outcome in result.outcomes],
    }


def _summary(
    *,
    args: argparse.Namespace,
    coverage: AnnotationCoverage,
    evaluated_queries: int,
    candidate_limit: int,
    model_load_ms: float,
    embedding: TextEmbedding[Any],
    result: EvaluationResult,
    latency: LatencyMetrics,
) -> dict[str, Any]:
    return {
        "annotations": str(args.annotations.resolve()),
        "target": args.target,
        "model_id": embedding.model_id,
        "vector_name": embedding.vector_name,
        "cutoffs": list(args.k),
        "candidate_limit": candidate_limit,
        "evaluated_queries": evaluated_queries,
        "coverage": coverage.to_dict(),
        "metrics": [asdict(metric) for metric in result.metrics],
        "throughput": {
            "mode": "batch_inference_sequential_retrieval",
            **asdict(result.throughput),
        },
        "query_latency_ms": {
            "mode": "sequential_embedding_sequential_retrieval",
            "sample_seed": args.seed,
            **asdict(latency),
        },
        "model_load_ms": model_load_ms,
    }


def _create_embedding(
    target: str,
    *,
    model_id: str | None,
    device: str,
    batch_size: int,
) -> TextEmbedding[Any]:
    if target == "image":
        arguments: dict[str, Any] = {
            "device": device,
            "batch_size": batch_size,
        }
        if model_id is not None:
            arguments["model_id"] = model_id
        return ClipImageEmbedding(**arguments)
    if target == "audio":
        arguments = {"device": device, "batch_size": batch_size}
        if model_id is not None:
            arguments["model_id"] = model_id
        return ClapAudioEmbedding(**arguments)
    if target == "document":
        arguments = {"device": device, "batch_size": batch_size}
        if model_id is not None:
            arguments["model_id"] = model_id
        return HarrierTextEmbedding(**arguments)
    raise ValueError(f"Unsupported retrieval target: {target}")


def _caption_field_order(field: str) -> tuple[float, str]:
    suffix = field.removeprefix("caption_")
    return (int(suffix), field) if suffix.isdigit() else (math.inf, field)


def _filename_key(value: str) -> str:
    return value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()


def _identity_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _query_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _document_query_id(enterprise: str, query: str) -> str:
    return f"{_identity_key(enterprise)}:{_query_key(query)}"


def _document_filename_key(value: str) -> str:
    filename = value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return _identity_key(Path(filename).stem)


def _document_query_type(value: str) -> str:
    if not value:
        raise ValueError("Document annotation query type must not be empty")
    official_by_key = {
        _identity_key(query_type): query_type
        for query_type in DOCUMENT_QUERY_TYPES
    }
    return official_by_key.get(_identity_key(value), "Other")


def _document_enterprise(entry: FileIndexEntry) -> str:
    return entry.path.parent.parent.name


def _logical_document_id(entry: FileIndexEntry) -> str:
    corpus_path = entry.path.parent.parent.as_posix().casefold()
    return f"{corpus_path}/{_document_filename_key(entry.path.name)}"


def _document_format_order(file_format: str) -> tuple[int, str]:
    try:
        return (DEFAULT_DOCUMENT_FORMATS.index(file_format), file_format)
    except ValueError:
        return (len(DEFAULT_DOCUMENT_FORMATS), file_format)


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Metric cutoffs must contain positive integers")
    return tuple(sorted(set(cutoffs)))


if __name__ == "__main__":
    raise SystemExit(main())
