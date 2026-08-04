"""Repeatable local or server benchmark for FileLore's Qdrant adapter."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import tempfile
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import date
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from time import perf_counter, sleep
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import QdrantClient, models

from filelore.storage import (
    CollectionConfig,
    MetadataCondition,
    MetadataFilter,
    MetadataIndex,
    MetadataIndexType,
    QdrantVectorDatabase,
    VectorConfig,
    VectorRecord,
)


BENCHMARK_INDEXES = (
    MetadataIndex("extension", MetadataIndexType.KEYWORD),
    MetadataIndex("size_bytes", MetadataIndexType.INTEGER),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qdrant ingest, payload filtering, and vector search."
    )
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--query-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--ready-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--url",
        help=(
            "Qdrant server URL, for example http://localhost:6333; "
            "omit to use Python Local Mode"
        ),
    )
    parser.add_argument(
        "--payload-indexes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="create extension and size_bytes indexes before ingest (server only)",
    )
    parser.add_argument(
        "--collection",
        help="collection name; defaults to an isolated generated name",
    )
    parser.add_argument(
        "--keep-collection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep a server collection after the benchmark for inspection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also save the printed JSON result to this path",
    )
    args = parser.parse_args()

    positive = {
        "records": args.records,
        "dimensions": args.dimensions,
        "batch size": args.batch_size,
        "query runs": args.query_runs,
        "ready timeout": args.ready_timeout_seconds,
    }
    for label, value in positive.items():
        if value < 1:
            parser.error(f"{label} must be positive")
    if args.warmup_runs < 0:
        parser.error("warmup runs must not be negative")
    if args.payload_indexes and args.url is None:
        parser.error(
            "--payload-indexes requires --url because Python Local Mode "
            "does not implement payload indexes"
        )
    if args.keep_collection and args.url is None:
        parser.error("--keep-collection only applies to server mode")
    return args


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def measure(
    operation: Callable[[], object],
    *,
    runs: int,
    warmup_runs: int,
) -> dict[str, float | int]:
    for _ in range(warmup_runs):
        operation()

    durations_ms: list[float] = []
    for _ in range(runs):
        started = perf_counter()
        operation()
        durations_ms.append((perf_counter() - started) * 1000)

    return {
        "runs": runs,
        "min_ms": round(min(durations_ms), 3),
        "mean_ms": round(mean(durations_ms), 3),
        "median_ms": round(median(durations_ms), 3),
        "p95_ms": round(percentile(durations_ms, 0.95), 3),
        "max_ms": round(max(durations_ms), 3),
    }


def metadata_filter(field: str, value: str | int) -> MetadataFilter:
    return MetadataFilter(all_of=(MetadataCondition(field, value),))


def build_record_batches(
    *,
    records: int,
    dimensions: int,
    batch_size: int,
    seed: int,
) -> tuple[tuple[VectorRecord, ...], ...]:
    """Build deterministic records outside the measured ingestion interval."""
    randomizer = random.Random(seed)
    batches: list[tuple[VectorRecord, ...]] = []
    for start in range(0, records, batch_size):
        batch: list[VectorRecord] = []
        for number in range(start, min(start + batch_size, records)):
            batch.append(
                VectorRecord(
                    id=str(uuid5(NAMESPACE_URL, f"benchmark:{number}")),
                    payload={
                        "extension": ".pdf" if number % 4 == 0 else ".txt",
                        "size_bytes": number * 100,
                    },
                    vectors={
                        "document": [
                            randomizer.random() for _ in range(dimensions)
                        ]
                    },
                )
            )
        batches.append(tuple(batch))
    return tuple(batches)


def payload_schema(client: QdrantClient, collection: str) -> dict[str, str]:
    schema = client.get_collection(collection).payload_schema or {}
    return {
        field: str(description.data_type).removeprefix("PayloadSchemaType.")
        for field, description in schema.items()
    }


def wait_for_collection_ready(
    client: QdrantClient,
    collection: str,
    *,
    timeout_seconds: float,
) -> float:
    """Wait until server-side indexing and optimization have settled."""
    started = perf_counter()
    deadline = started + timeout_seconds
    while True:
        information = client.get_collection(collection)
        if (
            information.status == models.CollectionStatus.GREEN
            and information.optimizer_status == models.OptimizersStatusOneOf.OK
        ):
            return perf_counter() - started
        optimizer_error = getattr(information.optimizer_status, "error", None)
        if optimizer_error is not None:
            raise RuntimeError(f"Qdrant optimizer failed: {optimizer_error}")
        if perf_counter() >= deadline:
            raise TimeoutError(
                f"Qdrant collection {collection!r} did not become ready "
                f"within {timeout_seconds:g} seconds"
            )
        sleep(0.1)


def run_benchmark(
    args: argparse.Namespace,
    database: QdrantVectorDatabase,
    *,
    client: QdrantClient | None,
    collection: str,
) -> dict[str, Any]:
    indexes = BENCHMARK_INDEXES if args.payload_indexes else ()
    database.create_collection(
        CollectionConfig(
            name=collection,
            vectors={"document": VectorConfig(args.dimensions)},
            metadata_indexes=indexes,
        )
    )

    workload_started = perf_counter()
    record_batches = build_record_batches(
        records=args.records,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    workload_build_seconds = perf_counter() - workload_started

    ingest_started = perf_counter()
    for records in record_batches:
        database.upsert(collection, records)
    ingest_seconds = perf_counter() - ingest_started
    ready_wait_seconds = (
        wait_for_collection_ready(
            client,
            collection,
            timeout_seconds=args.ready_timeout_seconds,
        )
        if client is not None
        else 0.0
    )

    extension_filter = metadata_filter("extension", ".pdf")
    target_number = args.records // 2
    size_filter = metadata_filter("size_bytes", target_number * 100)
    query_vector = [0.5] * args.dimensions

    extension_result = database.filter(
        collection,
        metadata_filter=extension_filter,
        limit=20,
    )
    size_result = database.filter(
        collection,
        metadata_filter=size_filter,
        limit=20,
    )
    vector_result = database.search(
        collection,
        query_vector,
        vector_name="document",
        limit=10,
    )
    filtered_vector_result = database.search(
        collection,
        query_vector,
        vector_name="document",
        metadata_filter=extension_filter,
        limit=10,
    )

    measure_args = {
        "runs": args.query_runs,
        "warmup_runs": args.warmup_runs,
    }
    extension_metrics = measure(
        lambda: database.filter(
            collection,
            metadata_filter=extension_filter,
            limit=20,
        ),
        **measure_args,
    )
    extension_metrics["results"] = len(extension_result.records)
    extension_metrics["selectivity"] = 0.25

    size_metrics = measure(
        lambda: database.filter(
            collection,
            metadata_filter=size_filter,
            limit=20,
        ),
        **measure_args,
    )
    size_metrics["results"] = len(size_result.records)
    size_metrics["selectivity"] = round(1 / args.records, 8)

    vector_metrics = measure(
        lambda: database.search(
            collection,
            query_vector,
            vector_name="document",
            limit=10,
        ),
        **measure_args,
    )
    vector_metrics["results"] = len(vector_result)

    filtered_vector_metrics = measure(
        lambda: database.search(
            collection,
            query_vector,
            vector_name="document",
            metadata_filter=extension_filter,
            limit=10,
        ),
        **measure_args,
    )
    filtered_vector_metrics["results"] = len(filtered_vector_result)
    filtered_vector_metrics["selectivity"] = 0.25

    schema = payload_schema(client, collection) if client is not None else {}
    return {
        "recorded_at": date.today().isoformat(),
        "mode": "server" if args.url is not None else "local",
        "collection": collection if args.keep_collection else None,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "qdrant_client": version("qdrant-client"),
        },
        "configuration": {
            "records": args.records,
            "dimensions": args.dimensions,
            "batch_size": args.batch_size,
            "query_runs": args.query_runs,
            "warmup_runs": args.warmup_runs,
            "ready_timeout_seconds": args.ready_timeout_seconds,
            "seed": args.seed,
            "payload_indexes": [
                {"field": index.field, "type": index.field_type.value}
                for index in indexes
            ],
        },
        "verified_payload_schema": schema,
        "workload_build_seconds": round(workload_build_seconds, 3),
        "ingest_seconds": round(ingest_seconds, 3),
        "ready_wait_seconds": round(ready_wait_seconds, 3),
        "metadata_queries": {
            "extension_keyword_25_percent": extension_metrics,
            "size_bytes_integer_single_record": size_metrics,
        },
        "vector_queries": {
            "unfiltered": vector_metrics,
            "extension_filtered_25_percent": filtered_vector_metrics,
        },
    }


def main() -> None:
    args = parse_args()
    collection = args.collection or f"filelore_benchmark_{uuid4().hex}"

    with ExitStack() as stack:
        if args.url is None:
            directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="filelore-benchmark-")
            )
            client = None
            database = stack.enter_context(
                QdrantVectorDatabase(Path(directory))
            )
        else:
            client = QdrantClient(url=args.url)
            database = stack.enter_context(QdrantVectorDatabase(client=client))
            if database.collection_exists(collection):
                raise ValueError(
                    f"Benchmark collection already exists: {collection!r}"
                )

        try:
            result = run_benchmark(
                args,
                database,
                client=client,
                collection=collection,
            )
        finally:
            if (
                client is not None
                and not args.keep_collection
                and client.collection_exists(collection)
            ):
                client.delete_collection(collection)

    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{serialized}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
