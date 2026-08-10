"""Evaluate text-to-media retrieval against COCO or Clotho annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from filelore.cli import DEFAULT_INDEX_PATH
from filelore.cli_display import CliDisplay
from filelore.embedding import (
    ClapAudioEmbedding,
    ClipImageEmbedding,
    TextEmbedding,
)
from filelore.index import FileIndexEntry, FileIndexRepository, FileSearchResult
from filelore.storage import QdrantVectorDatabase


DEFAULT_CUTOFFS = (1, 5, 10)
DEFAULT_BATCH_SIZE = 32
DEFAULT_LATENCY_SAMPLES = 1_000
DEFAULT_LATENCY_WARMUP = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True, slots=True)
class RetrievalAnnotation:
    query: str
    relevant_file: str


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


def measure_query_latency(
    file_index: FileIndexRepository,
    embedding: TextEmbedding[Any],
    queries: Sequence[EligibleQuery],
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
    queries: Sequence[EligibleQuery],
    *,
    sample_size: int,
    seed: int,
) -> tuple[EligibleQuery, ...]:
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
    }.get(target)
    if search is None:
        raise ValueError(f"Unsupported retrieval target: {target}")
    return search


def _batches(
    items: Sequence[EligibleQuery], batch_size: int
) -> tuple[Sequence[EligibleQuery], ...]:
    return tuple(
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FileLore text-to-image or text-to-audio retrieval against "
            "COCO or Clotho captions."
        )
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="COCO captions JSON or Clotho captions CSV",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=("image", "audio"),
        help="select the indexed media type and its embedding model",
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
            "raw results requested per query; audio defaults to max(100, 10*k) "
            "to allow parent-file deduplication"
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
    raise ValueError(f"Unsupported retrieval target: {target}")


def _caption_field_order(field: str) -> tuple[float, str]:
    suffix = field.removeprefix("caption_")
    return (int(suffix), field) if suffix.isdigit() else (math.inf, field)


def _filename_key(value: str) -> str:
    return value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Metric cutoffs must contain positive integers")
    return tuple(sorted(set(cutoffs)))


if __name__ == "__main__":
    raise SystemExit(main())
