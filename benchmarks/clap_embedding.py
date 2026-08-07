"""Repeatable end-to-end benchmark for FileLore CLAP embeddings."""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from filelore.embedding import AudioInput, ClapAudioEmbedding


TEXT_PROMPTS = (
    "a dog barking",
    "rain falling on a window",
    "a thunderstorm",
    "waves crashing on a beach",
    "a person playing an acoustic guitar",
    "birds singing in a forest",
    "traffic on a busy city street",
    "people applauding in a large room",
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


def synthetic_audio(
    count: int,
    *,
    sampling_rate: int,
    duration_seconds: float,
    seed: int,
) -> list[AudioInput]:
    """Create deterministic mono waveforms outside the measured interval."""
    sample_count = round(sampling_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sampling_rate
    randomizer = np.random.default_rng(seed)
    inputs: list[AudioInput] = []
    for index in range(count):
        fundamental_hz = 110.0 + index * 13.0
        samples = (
            0.45 * np.sin(2 * np.pi * fundamental_hz * time_axis)
            + 0.2 * np.sin(2 * np.pi * fundamental_hz * 2.0 * time_axis)
            + 0.02 * randomizer.standard_normal(sample_count)
        ).astype(np.float32)
        inputs.append(AudioInput(samples=samples, sampling_rate=sampling_rate))
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end CLAP audio and text embedding inference."
    )
    parser.add_argument("--model", default="laion/larger_clap_general")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        metavar="N",
    )
    parser.add_argument("--audio-count", type=int, default=64)
    parser.add_argument(
        "--audio-duration",
        type=float,
        help="waveform duration in seconds (default: model maximum)",
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
        "audio count": args.audio_count,
        "text count": args.text_count,
        "runs": args.runs,
    }
    for label, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if args.audio_duration is not None and (
        not math.isfinite(args.audio_duration) or args.audio_duration <= 0
    ):
        raise ValueError("audio duration must be a finite positive number")
    if args.warmup_runs < 0:
        raise ValueError("warmup runs must not be negative")
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        raise ValueError("batch sizes must contain positive values")
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    import transformers

    torch.set_float32_matmul_precision(args.matmul_precision)
    model_started = perf_counter()
    embedding = ClapAudioEmbedding(
        model_id=args.model,
        device=args.device,
        batch_size=max(args.batch_sizes),
    )
    model_load_seconds = perf_counter() - model_started

    audio_duration = (
        embedding.max_length_seconds
        if args.audio_duration is None
        else args.audio_duration
    )
    if audio_duration > embedding.max_length_seconds:
        embedding.close()
        raise ValueError(
            f"audio duration must not exceed the model maximum of "
            f"{embedding.max_length_seconds:g} seconds"
        )

    audio_inputs = synthetic_audio(
        args.audio_count,
        sampling_rate=embedding.sampling_rate,
        duration_seconds=audio_duration,
        seed=args.seed,
    )
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

    try:
        embedding.batch_size = 1
        audio_single = measure(
            lambda: embedding.predict(audio_inputs[0]),
            item_count=1,
            **measure_arguments,
        )
        text_single = measure(
            lambda: embedding.predict_text(texts[0]),
            item_count=1,
            **measure_arguments,
        )

        audio_batches: list[dict[str, Any]] = []
        text_batches: list[dict[str, Any]] = []
        for batch_size in args.batch_sizes:
            embedding.batch_size = batch_size
            audio_result = measure(
                lambda: embedding.predict_batch(audio_inputs),
                item_count=len(audio_inputs),
                **measure_arguments,
            )
            audio_result["batch_size"] = batch_size
            audio_batches.append(audio_result)

            text_result = measure(
                lambda: embedding.predict_text_batch(texts),
                item_count=len(texts),
                **measure_arguments,
            )
            text_result["batch_size"] = batch_size
            text_batches.append(text_result)

        best_audio = max(
            audio_batches,
            key=lambda result: float(result["median_items_per_second"]),
        )
        result = {
            "recorded_at": date.today().isoformat(),
            "baseline_score": {
                "name": "best_median_audio_embeddings_per_second",
                "value": best_audio["median_items_per_second"],
                "batch_size": best_audio["batch_size"],
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "device": embedding.device,
                "device_name": (
                    torch.cuda.get_device_name(0)
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
                "audio_source": "synthetic_memory",
                "audio_count": len(audio_inputs),
                "audio_duration_seconds": audio_duration,
                "sampling_rate_hz": embedding.sampling_rate,
                "text_count": len(texts),
                "batch_sizes": args.batch_sizes,
                "runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "seed": args.seed,
            },
            "model_load_seconds": round(model_load_seconds, 3),
            "audio_single": audio_single,
            "text_single": text_single,
            "audio_batches": audio_batches,
            "text_batches": text_batches,
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
