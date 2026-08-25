"""Repeatable end-to-end benchmark for FileLore Harrier embeddings."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections.abc import Callable, Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from filelore.documents import ParagraphChunker
from filelore.embedding import DEFAULT_HARRIER_MODEL, HarrierTextEmbedding


DOCUMENT_PASSAGES = (
    (
        "Public transport connects neighborhoods through rail, bus, ferry, and "
        "walking routes. Service frequency, reliable transfers, and accessible "
        "stations determine whether travelers can reach work and education."
    ),
    (
        "Kent içi ulaşım planlamasında yolculuk süresi, aktarma kolaylığı "
        "ve erişilebilirlik birlikte değerlendirilir. Düzenli seferler, farklı "
        "mahallelerde yaşayan insanların kamu hizmetlerine ulaşmasını "
        "kolaylaştırır."
    ),
    (
        "Climate observations combine temperature, precipitation, wind, and "
        "ocean measurements. Long-running records help researchers distinguish "
        "short-term variability from persistent changes in regional conditions."
    ),
    (
        "Bir araştırma raporu; yöntemi, veri kaynaklarını, sınırlılıkları ve "
        "sonuçları açıkça belirtmelidir. Bu yapı, okuyucunun bulguları yeniden "
        "değerlendirmesine ve kanıtları karşılaştırmasına yardım eder."
    ),
    (
        "日本の都市計画では、住宅、公共交通、公園、防災設備を総合的に配置する。"
        "地域の人口変化と生活パターンを分析し、"
        "長期的に利用できる基盤を整備する。"
    ),
    (
        "يعتمد التخطيط الحضري على بيانات السكان والنقل "
        "والخدمات العامة. ويساعد تحليل هذه البيانات على توزيع "
        "الموارد بشكل عادل وتحسين جودة الحياة."
    ),
    (
        "A software design document records constraints, interfaces, failure "
        "modes, and operational assumptions. Concrete examples make the design "
        "reviewable and reduce ambiguity when several components evolve together."
    ),
    (
        "The archive catalog describes creators, dates, subjects, formats, and "
        "relationships between records. Consistent metadata lets researchers find "
        "relevant material without knowing the collection's physical arrangement."
    ),
)

QUERY_PROMPTS = (
    "how public transport improves access to work",
    "kent içi ulaşımda aktarma kolaylığı",
    "evidence used to study regional climate change",
    "araştırma raporunda yöntem ve sınırlılıklar",
    "日本の都市計画と防災設備",
    "توزيع الموارد في التخطيط الحضري",
    "what belongs in a software design document",
    "metadata for discovering archival records",
)


def percentile(values: Sequence[float], probability: float) -> float:
    """Calculate an interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
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
    item_count: int,
    character_count: int,
    runs: int,
    warmup_runs: int,
    synchronize: Callable[[], None],
    reset_peak_memory: Callable[[], None],
    peak_memory_mb: Callable[[], float | None],
) -> dict[str, float | int | None]:
    """Measure a synchronous operation after unrecorded warm-up runs."""
    for _ in range(warmup_runs):
        operation()
        synchronize()

    reset_peak_memory()
    durations_ms: list[float] = []
    for _ in range(runs):
        synchronize()
        started = perf_counter()
        operation()
        synchronize()
        durations_ms.append((perf_counter() - started) * 1000)

    median_ms = median(durations_ms)
    return {
        "items": item_count,
        "input_characters": character_count,
        "runs": runs,
        "min_ms": round(min(durations_ms), 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(percentile(durations_ms, 0.95), 3),
        "max_ms": round(max(durations_ms), 3),
        "median_ms_per_item": round(median_ms / item_count, 3),
        "median_items_per_second": round(item_count * 1000 / median_ms, 3),
        "median_characters_per_second": round(
            character_count * 1000 / median_ms, 3
        ),
        "peak_gpu_memory_mb": peak_memory_mb(),
    }


def synthetic_documents(
    count: int, *, characters: int, seed: int
) -> list[str]:
    """Create deterministic multilingual chunk-sized inputs in memory."""
    randomizer = random.Random(seed)
    documents: list[str] = []
    for index in range(count):
        passages = list(DOCUMENT_PASSAGES)
        randomizer.shuffle(passages)
        heading = f"Benchmark document {index + 1}"
        parts = [heading]
        passage_index = 0
        while len("\n\n".join(parts)) < characters:
            parts.append(passages[passage_index % len(passages)])
            passage_index += 1
        documents.append("\n\n".join(parts)[:characters].rstrip())
    return documents


def repeated_queries(count: int) -> list[str]:
    """Return a deterministic multilingual short-query workload."""
    return [QUERY_PROMPTS[index % len(QUERY_PROMPTS)] for index in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Harrier document and query embedding inference."
    )
    parser.add_argument("--model", default=DEFAULT_HARRIER_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
        metavar="N",
    )
    parser.add_argument("--document-count", type=int, default=64)
    parser.add_argument(
        "--document-characters",
        type=int,
        default=ParagraphChunker.default_max_characters,
        help="characters per synthetic document chunk (default: chunker maximum)",
    )
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON result to this file",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "document count": args.document_count,
        "document characters": args.document_characters,
        "query count": args.query_count,
        "runs": args.runs,
    }
    for label, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if args.warmup_runs < 0:
        raise ValueError("warmup runs must not be negative")
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        raise ValueError("batch sizes must contain positive values")
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch

    torch.set_float32_matmul_precision(args.matmul_precision)
    model_started = perf_counter()
    embedding = HarrierTextEmbedding(
        model_id=args.model,
        device=args.device,
        batch_size=max(args.batch_sizes),
    )
    model_load_seconds = perf_counter() - model_started

    documents = synthetic_documents(
        args.document_count,
        characters=args.document_characters,
        seed=args.seed,
    )
    queries = repeated_queries(args.query_count)
    using_cuda = embedding.device.startswith("cuda")

    def synchronize() -> None:
        if using_cuda:
            torch.cuda.synchronize()

    def reset_peak_memory() -> None:
        if using_cuda:
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb() -> float | None:
        if not using_cuda:
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    measure_arguments: dict[str, Any] = {
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "synchronize": synchronize,
        "reset_peak_memory": reset_peak_memory,
        "peak_memory_mb": peak_memory_mb,
    }

    try:
        embedding.batch_size = 1
        document_single = measure(
            lambda: embedding.predict(documents[0]),
            item_count=1,
            character_count=len(documents[0]),
            **measure_arguments,
        )
        query_single = measure(
            lambda: embedding.predict_text(queries[0]),
            item_count=1,
            character_count=len(queries[0]),
            **measure_arguments,
        )

        document_batches: list[dict[str, Any]] = []
        query_batches: list[dict[str, Any]] = []
        for batch_size in args.batch_sizes:
            embedding.batch_size = batch_size
            document_result = measure(
                lambda: embedding.predict_batch(documents),
                item_count=len(documents),
                character_count=sum(map(len, documents)),
                **measure_arguments,
            )
            document_result["batch_size"] = batch_size
            document_batches.append(document_result)

            query_result = measure(
                lambda: embedding.predict_text_batch(queries),
                item_count=len(queries),
                character_count=sum(map(len, queries)),
                **measure_arguments,
            )
            query_result["batch_size"] = batch_size
            query_batches.append(query_result)

        best_document = max(
            document_batches,
            key=lambda result: float(result["median_items_per_second"]),
        )
        result = {
            "recorded_at": date.today().isoformat(),
            "baseline_score": {
                "name": "best_median_document_embeddings_per_second",
                "value": best_document["median_items_per_second"],
                "batch_size": best_document["batch_size"],
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "sentence_transformers": version("sentence-transformers"),
                "device": embedding.device,
                "device_name": (
                    torch.cuda.get_device_name(embedding.device)
                    if using_cuda
                    else platform.processor()
                ),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if using_cuda else None,
                "matmul_precision": torch.get_float32_matmul_precision(),
            },
            "configuration": {
                "model": embedding.model_id,
                "dimensions": embedding.dimensions,
                "query_prompt_name": embedding.query_prompt_name,
                "document_source": "synthetic_memory",
                "document_count": len(documents),
                "document_characters": args.document_characters,
                "query_count": len(queries),
                "batch_sizes": args.batch_sizes,
                "runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "seed": args.seed,
            },
            "model_load_seconds": round(model_load_seconds, 3),
            "document_single": document_single,
            "query_single": query_single,
            "document_batches": document_batches,
            "query_batches": query_batches,
        }
        serialized = json.dumps(result, indent=2)
        print(serialized)
        if args.output is not None:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{serialized}\n", encoding="utf-8")
    finally:
        embedding.close()


if __name__ == "__main__":
    main()
