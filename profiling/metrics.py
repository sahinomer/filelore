"""Metric collection and report generation for indexing profiles."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import subprocess
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Iterator, Sequence


@dataclass(slots=True)
class StageEvent:
    event_id: int
    parent_id: int | None
    stage: str
    started_ms: float
    duration_ms: float
    cpu_ms: float
    items: int | None = None
    input_bytes: int | None = None
    cuda_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResourceSample:
    timestamp_ms: float
    process_cpu_percent: float
    system_cpu_percent: float
    rss_bytes: int
    read_bytes: int
    write_bytes: int
    gpu_utilization_percent: float | None = None
    gpu_memory_mb: float | None = None
    gpu_power_watts: float | None = None


@dataclass(slots=True)
class StageSpan:
    items: int | None = None
    input_bytes: int | None = None
    cuda_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class StageRecorder:
    """Collect nested, low-overhead wall and process CPU timing events."""

    def __init__(self) -> None:
        self.started = perf_counter()
        self.events: list[StageEvent] = []
        self._event_ids = 0
        self._lock = threading.Lock()
        self._parents: ContextVar[tuple[int, ...]] = ContextVar(
            "filelore_profile_parents", default=()
        )

    @contextmanager
    def span(
        self,
        stage: str,
        *,
        items: int | None = None,
        input_bytes: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> Iterator[StageSpan]:
        with self._lock:
            self._event_ids += 1
            event_id = self._event_ids
        parents = self._parents.get()
        parent_id = parents[-1] if parents else None
        token = self._parents.set((*parents, event_id))
        started = perf_counter()
        cpu_started = process_time()
        measured = StageSpan(
            items=items,
            input_bytes=input_bytes,
            details=dict(details or {}),
        )
        error: str | None = None
        try:
            yield measured
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            finished = perf_counter()
            event = StageEvent(
                event_id=event_id,
                parent_id=parent_id,
                stage=stage,
                started_ms=round((started - self.started) * 1000, 3),
                duration_ms=round((finished - started) * 1000, 3),
                cpu_ms=round((process_time() - cpu_started) * 1000, 3),
                items=measured.items,
                input_bytes=measured.input_bytes,
                cuda_ms=(
                    round(measured.cuda_ms, 3)
                    if measured.cuda_ms is not None
                    else None
                ),
                error=error,
                details=measured.details,
            )
            with self._lock:
                self.events.append(event)
            self._parents.reset(token)

    def require_stages(self, stages: Sequence[str]) -> None:
        recorded = {event.stage for event in self.events}
        missing = sorted(set(stages).difference(recorded))
        if missing:
            raise RuntimeError(
                "Profiler instrumentation did not observe required stages: "
                + ", ".join(missing)
            )


class _NvidiaMonitor:
    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[float, float, float] | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=creation_flags,
            )
        except OSError:
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                utilization, memory, power = (
                    float(value.strip()) for value in line.split(",")[:3]
                )
            except (TypeError, ValueError):
                continue
            with self._lock:
                self._latest = (utilization, memory, power)

    def latest(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)


class ResourceSampler:
    """Sample process, system, disk-I/O, and NVIDIA GPU utilization."""

    def __init__(self, recorder: StageRecorder, *, interval_ms: int = 200) -> None:
        if interval_ms < 50:
            raise ValueError("Resource sample interval must be at least 50 ms")
        self.recorder = recorder
        self.interval_ms = interval_ms
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia = _NvidiaMonitor(interval_ms)
        self._stopped = False

    def start(self) -> None:
        try:
            import psutil
        except ImportError as error:
            raise ImportError(
                "Index profiling requires psutil; run "
                "'uv sync --extra embedding --group profiling'"
            ) from error
        self._psutil = psutil
        self._process = psutil.Process()
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._nvidia.start()
        self._take_sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000):
            self._take_sample()

    def _take_sample(self) -> None:
        try:
            memory = self._process.memory_info()
            io = self._process.io_counters()
            gpu = self._nvidia.latest()
            self.samples.append(
                ResourceSample(
                    timestamp_ms=round(
                        (perf_counter() - self.recorder.started) * 1000, 3
                    ),
                    process_cpu_percent=float(self._process.cpu_percent(None)),
                    system_cpu_percent=float(self._psutil.cpu_percent(None)),
                    rss_bytes=int(memory.rss),
                    read_bytes=int(io.read_bytes),
                    write_bytes=int(io.write_bytes),
                    gpu_utilization_percent=gpu[0] if gpu else None,
                    gpu_memory_mb=gpu[1] if gpu else None,
                    gpu_power_watts=gpu[2] if gpu else None,
                )
            )
        except (OSError, self._psutil.Error):
            pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._take_sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._nvidia.stop()

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def percentile(values: Sequence[float], probability: float) -> float:
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


def aggregate_stages(events: Sequence[StageEvent]) -> list[dict[str, Any]]:
    run_ms = max(
        (event.duration_ms for event in events if event.stage == "overall.run"),
        default=max((event.duration_ms for event in events), default=0.0),
    )
    stages: dict[str, list[StageEvent]] = {}
    for event in events:
        stages.setdefault(event.stage, []).append(event)

    aggregates: list[dict[str, Any]] = []
    for stage, recorded in stages.items():
        durations = [event.duration_ms for event in recorded]
        total_ms = sum(durations)
        item_values = [event.items for event in recorded if event.items is not None]
        byte_values = [
            event.input_bytes
            for event in recorded
            if event.input_bytes is not None
        ]
        cuda_values = [
            event.cuda_ms for event in recorded if event.cuda_ms is not None
        ]
        detail_totals: dict[str, float] = {}
        for event in recorded:
            for name, value in event.details.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    detail_totals[name] = detail_totals.get(name, 0.0) + value
        items = sum(item_values) if item_values else None
        input_bytes = sum(byte_values) if byte_values else None
        aggregates.append(
            {
                "stage": stage,
                "calls": len(recorded),
                "items": items,
                "input_bytes": input_bytes,
                "total_ms": round(total_ms, 3),
                "inclusive_percent_of_run": (
                    round(total_ms * 100 / run_ms, 2) if run_ms else None
                ),
                "median_ms": round(statistics.median(durations), 3),
                "p95_ms": round(percentile(durations, 0.95), 3),
                "max_ms": round(max(durations), 3),
                "cpu_ms": round(sum(event.cpu_ms for event in recorded), 3),
                "cuda_ms": round(sum(cuda_values), 3) if cuda_values else None,
                "items_per_second": (
                    round(items * 1000 / total_ms, 3)
                    if items is not None and total_ms > 0
                    else None
                ),
                "input_mb_per_second": (
                    round(input_bytes / (1024 * 1024) * 1000 / total_ms, 3)
                    if input_bytes is not None and total_ms > 0
                    else None
                ),
                "errors": sum(event.error is not None for event in recorded),
                "detail_totals": {
                    name: round(value, 3)
                    for name, value in sorted(detail_totals.items())
                },
            }
        )
    return sorted(aggregates, key=lambda item: str(item["stage"]))


def summarize_resources(samples: Sequence[ResourceSample]) -> dict[str, Any]:
    if not samples:
        return {}
    process_cpu = [sample.process_cpu_percent for sample in samples]
    system_cpu = [sample.system_cpu_percent for sample in samples]
    rss_mb = [sample.rss_bytes / (1024 * 1024) for sample in samples]
    gpu_samples = [
        sample.gpu_utilization_percent
        for sample in samples
        if sample.gpu_utilization_percent is not None
    ]
    gpu_memory = [
        sample.gpu_memory_mb
        for sample in samples
        if sample.gpu_memory_mb is not None
    ]
    gpu_power = [
        sample.gpu_power_watts
        for sample in samples
        if sample.gpu_power_watts is not None
    ]
    return {
        "samples": len(samples),
        "process_cpu_average_percent": round(
            statistics.fmean(process_cpu), 2
        ),
        "process_cpu_p95_percent": round(
            percentile(process_cpu, 0.95), 2
        ),
        "process_cpu_max_percent": round(max(process_cpu), 2),
        "system_cpu_average_percent": round(
            statistics.fmean(system_cpu), 2
        ),
        "system_cpu_p95_percent": round(percentile(system_cpu, 0.95), 2),
        "system_cpu_max_percent": round(max(system_cpu), 2),
        "rss_average_mb": round(statistics.fmean(rss_mb), 3),
        "rss_p95_mb": round(percentile(rss_mb, 0.95), 3),
        "peak_rss_mb": round(max(rss_mb), 3),
        "process_read_mb": round(
            max(0, samples[-1].read_bytes - samples[0].read_bytes)
            / (1024 * 1024),
            3,
        ),
        "process_write_mb": round(
            max(0, samples[-1].write_bytes - samples[0].write_bytes)
            / (1024 * 1024),
            3,
        ),
        "gpu_utilization_average_percent": (
            round(statistics.fmean(gpu_samples), 2) if gpu_samples else None
        ),
        "gpu_utilization_p95_percent": (
            round(percentile(gpu_samples, 0.95), 2) if gpu_samples else None
        ),
        "gpu_active_sample_percent": (
            round(sum(value > 0 for value in gpu_samples) * 100 / len(gpu_samples), 2)
            if gpu_samples
            else None
        ),
        "gpu_memory_average_mb": (
            round(statistics.fmean(gpu_memory), 3) if gpu_memory else None
        ),
        "gpu_memory_p95_mb": (
            round(percentile(gpu_memory, 0.95), 3) if gpu_memory else None
        ),
        "peak_gpu_memory_mb": round(max(gpu_memory), 3) if gpu_memory else None,
        "gpu_power_average_watts": (
            round(statistics.fmean(gpu_power), 2) if gpu_power else None
        ),
        "gpu_power_p95_watts": (
            round(percentile(gpu_power, 0.95), 2) if gpu_power else None
        ),
        "gpu_power_max_watts": round(max(gpu_power), 2) if gpu_power else None,
    }


def environment_details() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "beautifulsoup4",
        "markdown-it-py",
        "numpy",
        "Pillow",
        "psutil",
        "pypdf",
        "python-docx",
        "python-pptx",
        "qdrant-client",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        import torch
    except ImportError:
        device_name = None
        cuda = None
    else:
        cuda = torch.version.cuda
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cuda": cuda,
        "cuda_device": device_name,
        "packages": packages,
    }


def write_profile_outputs(
    output_directory: Path,
    *,
    configuration: dict[str, Any],
    exit_codes: dict[str, int],
    events: Sequence[StageEvent],
    samples: Sequence[ResourceSample],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stages = aggregate_stages(events)
    resources = summarize_resources(samples)
    summary = {
        "schema_version": 1,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "environment": environment_details(),
        "configuration": configuration,
        "exit_codes": exit_codes,
        "stages": stages,
        "resources": resources,
    }
    (output_directory / "summary.json").write_text(
        f"{json.dumps(summary, indent=2)}\n", encoding="utf-8"
    )
    _write_events(output_directory / "events.csv", events)
    _write_resources(output_directory / "resources.csv", samples)
    (output_directory / "summary.md").write_text(
        _markdown_summary(summary), encoding="utf-8"
    )


def _write_events(path: Path, events: Sequence[StageEvent]) -> None:
    fields = tuple(asdict(StageEvent(0, None, "", 0, 0, 0)))
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for event in sorted(events, key=lambda item: item.event_id):
            row = asdict(event)
            row["details"] = json.dumps(row["details"], sort_keys=True)
            writer.writerow(row)


def _write_resources(path: Path, samples: Sequence[ResourceSample]) -> None:
    fields = tuple(asdict(ResourceSample(0, 0, 0, 0, 0, 0)))
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)


def _markdown_summary(summary: dict[str, Any]) -> str:
    configuration = summary["configuration"]
    environment = summary["environment"]
    discovered = configuration.get("discovered_files", {})
    discovered_extensions = configuration.get("discovered_extensions", {})
    displayed_extensions = ", ".join(
        f"{extension}={count}"
        for extension, count in sorted(discovered_extensions.items())
    )
    lines = [
        "# FileLore indexing profile",
        "",
        f"Recorded: `{summary['recorded_at']}`",
        "",
        "## Test setup",
        "",
        f"- Platform: `{environment['platform']}`",
        f"- CPU: `{environment['processor'] or '-'}`",
        f"- CUDA GPU: `{environment['cuda_device'] or '-'}`",
        f"- Python: `{environment['python']}`",
        f"- Image directory: `{configuration.get('image_directory') or '-'}`",
        f"- Audio directory: `{configuration.get('audio_directory') or '-'}`",
        f"- Document directory: `{configuration.get('document_directory') or '-'}`",
        f"- Index mode: `{configuration.get('index_mode') or '-'}`",
        f"- Local index path: `{configuration.get('index_path') or '-'}`",
        f"- Qdrant service URL: `{configuration.get('qdrant_url') or '-'}`",
        f"- Discovered images: `{discovered.get('image', 0)}`",
        f"- Discovered audio files: `{discovered.get('audio', 0)}`",
        f"- Discovered documents: `{discovered.get('text', 0)}`",
        f"- Discovered extensions: `{displayed_extensions or '-'}`",
        f"- Index batch size: `{configuration['batch_size']}`",
        f"- Image model: `{configuration['image_model']}`",
        f"- Audio model: `{configuration['audio_model']}`",
        f"- Document model: `{configuration['document_model']}`",
        "",
        "## Stage metrics",
        "",
        "Stage times are inclusive. Nested rows must not be added together.",
        "",
        "| Stage | Calls | Items | Total | % run | Median | p95 | Items/s | CUDA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in summary["stages"]:
        lines.append(
            "| {stage} | {calls} | {items} | {total} ms | {percent} | "
            "{median} ms | {p95} ms | {rate} | {cuda} |".format(
                stage=stage["stage"],
                calls=stage["calls"],
                items=stage["items"] if stage["items"] is not None else "-",
                total=stage["total_ms"],
                percent=(
                    f"{stage['inclusive_percent_of_run']}%"
                    if stage["inclusive_percent_of_run"] is not None
                    else "-"
                ),
                median=stage["median_ms"],
                p95=stage["p95_ms"],
                rate=(
                    stage["items_per_second"]
                    if stage["items_per_second"] is not None
                    else "-"
                ),
                cuda=(
                    f"{stage['cuda_ms']} ms"
                    if stage["cuda_ms"] is not None
                    else "-"
                ),
            )
        )
    lines.extend(["", "## Resource metrics", ""])
    resources = summary["resources"]
    if not resources:
        lines.append("Resource sampling was disabled.")
    else:
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                *(
                    "| {name} | {value} |".format(
                        name=name.replace("_", " "),
                        value=value if value is not None else "-",
                    )
                    for name, value in resources.items()
                ),
            ]
        )
    return "\n".join(lines) + "\n"
