"""Repeatable end-to-end benchmark for FileLore CLIP embeddings."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from PIL import Image

from filelore.embedding import ClipImageEmbedding, ImageInput
from filelore.metadata import ImageMetadataParser


TEXT_PROMPTS = (
    "a red sports car on a city street",
    "a dog running through green grass",
    "a snowy mountain under a blue sky",
    "a family standing on a sandy beach",
    "a close-up photograph of a flower",
    "a plate of food on a wooden table",
    "an old building at sunset",
    "a person riding a bicycle",
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
        "runs": runs,
        "min_ms": round(min(durations_ms), 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(percentile(durations_ms, 0.95), 3),
        "max_ms": round(max(durations_ms), 3),
        "median_ms_per_item": round(median_ms / item_count, 3),
        "median_items_per_second": round(item_count * 1000 / median_ms, 3),
        "peak_gpu_memory_mb": peak_memory_mb(),
    }


def synthetic_images(
    count: int, *, size: int, seed: int
) -> list[Image.Image]:
    """Create deterministic RGB inputs outside the measured interval."""
    randomizer = random.Random(seed)
    return [
        Image.new(
            "RGB",
            (size, size),
            color=(
                randomizer.randrange(256),
                randomizer.randrange(256),
                randomizer.randrange(256),
            ),
        )
        for _ in range(count)
    ]


def image_paths(directory: Path, *, count: int) -> list[Path]:
    paths = list(ImageMetadataParser().discover(directory))
    if not paths:
        raise ValueError(f"No supported images found under {directory}")
    return paths[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end CLIP image and text embedding inference."
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--use-fast-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the fast CLIP image processor (default: enabled)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
        metavar="N",
    )
    parser.add_argument("--image-count", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--image-directory",
        type=Path,
        help="use real image paths instead of generated in-memory images",
    )
    parser.add_argument("--text-count", type=int, default=64)
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
        "image count": args.image_count,
        "image size": args.image_size,
        "text count": args.text_count,
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
    if args.image_directory is not None:
        args.image_directory = args.image_directory.expanduser()
        if not args.image_directory.is_dir():
            raise ValueError(
                f"Image directory does not exist: {args.image_directory}"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    import transformers

    torch.set_float32_matmul_precision(args.matmul_precision)
    model_started = perf_counter()
    embedding = ClipImageEmbedding(
        model_id=args.model,
        device=args.device,
        batch_size=max(args.batch_sizes),
        use_fast_processor=args.use_fast_processor,
    )
    model_load_seconds = perf_counter() - model_started

    if args.image_directory is None:
        images: list[ImageInput] = synthetic_images(
            args.image_count,
            size=args.image_size,
            seed=args.seed,
        )
        image_source = "synthetic_memory"
    else:
        images = image_paths(args.image_directory, count=args.image_count)
        image_source = "filesystem"
    texts = [
        TEXT_PROMPTS[index % len(TEXT_PROMPTS)]
        for index in range(args.text_count)
    ]

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

    embedding.batch_size = 1
    image_single = measure(
        lambda: embedding.predict(images[0]),
        item_count=1,
        **measure_arguments,
    )
    text_single = measure(
        lambda: embedding.predict_text(texts[0]),
        item_count=1,
        **measure_arguments,
    )

    image_batches: list[dict[str, Any]] = []
    text_batches: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        embedding.batch_size = batch_size
        image_result = measure(
            lambda: embedding.predict_batch(images),
            item_count=len(images),
            **measure_arguments,
        )
        image_result["batch_size"] = batch_size
        image_batches.append(image_result)

        text_result = measure(
            lambda: embedding.predict_text_batch(texts),
            item_count=len(texts),
            **measure_arguments,
        )
        text_result["batch_size"] = batch_size
        text_batches.append(text_result)

    best_image = max(
        image_batches,
        key=lambda result: float(result["median_items_per_second"]),
    )
    result = {
        "recorded_at": date.today().isoformat(),
        "baseline_score": {
            "name": "best_median_image_embeddings_per_second",
            "value": best_image["median_items_per_second"],
            "batch_size": best_image["batch_size"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": embedding.device,
            "device_name": (
                torch.cuda.get_device_name(0) if using_cuda else platform.processor()
            ),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if using_cuda else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "configuration": {
            "model": embedding.model_id,
            "dimensions": embedding.dimensions,
            "processor_use_fast": args.use_fast_processor,
            "image_source": image_source,
            "image_count": len(images),
            "image_size": args.image_size if image_source == "synthetic_memory" else None,
            "text_count": len(texts),
            "batch_sizes": args.batch_sizes,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
        },
        "model_load_seconds": round(model_load_seconds, 3),
        "image_single": image_single,
        "text_single": text_single,
        "image_batches": image_batches,
        "text_batches": text_batches,
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{serialized}\n", encoding="utf-8")

    for image in images:
        if isinstance(image, Image.Image):
            image.close()


if __name__ == "__main__":
    main()
