"""Profile the real FileLore indexing pipeline without changing core code."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import patch

from filelore.cli import main as filelore_main
from filelore.embedding import (
    AudioEmbedding,
    ClapAudioEmbedding,
    ClipImageEmbedding,
    ImageEmbedding,
)
from filelore.storage import QdrantVectorDatabase
from profiling.instrumentation import ExternalInstrumentation
from profiling.metrics import (
    ResourceSampler,
    ResourceSample,
    StageEvent,
    StageRecorder,
    write_profile_outputs,
)


DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_AUDIO_MODEL = "laion/larger_clap_general"


@dataclass(frozen=True, slots=True)
class ProfileConfiguration:
    output_directory: Path
    image_directory: Path | None = None
    audio_directory: Path | None = None
    index_path: Path | None = None
    qdrant_url: str | None = None
    batch_size: int = 100
    image_model: str = DEFAULT_IMAGE_MODEL
    audio_model: str = DEFAULT_AUDIO_MODEL
    device: str = "auto"
    use_fast_image_processor: bool = True
    sample_interval_ms: int = 200
    resource_sampling: bool = True
    cprofile: bool = False


@dataclass(frozen=True, slots=True)
class ProfileResult:
    output_directory: Path
    exit_codes: dict[str, int]
    events: tuple[StageEvent, ...]
    resource_samples: tuple[ResourceSample, ...]

    @property
    def successful(self) -> bool:
        return all(code == 0 for code in self.exit_codes.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile FileLore's real indexing pipeline with external stage "
            "and resource instrumentation."
        )
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        help="COCO-Val-2017 image directory or another image dataset",
    )
    parser.add_argument(
        "--audio-directory",
        type=Path,
        help="clotho_audio_evaluation directory or another audio dataset",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="result directory (default: profiling/results/<timestamp>)",
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--qdrant-url",
        help=(
            "Qdrant service URL; the files and files_segments collections "
            "must be absent or empty"
        ),
    )
    storage.add_argument(
        "--index-path",
        type=Path,
        help=(
            "persistent local Qdrant path; must be absent or empty "
            "(default: temporary directory)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--audio-model", default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--use-fast-image-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample-interval-ms", type=int, default=200)
    parser.add_argument(
        "--resource-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample CPU, memory, process I/O, and NVIDIA GPU metrics",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="also write a cProfile call profile",
    )
    return parser.parse_args(argv)


def configuration_from_args(args: argparse.Namespace) -> ProfileConfiguration:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = args.output_directory or (
        Path(__file__).resolve().parent / "results" / timestamp
    )
    configuration = ProfileConfiguration(
        output_directory=output_directory.expanduser(),
        image_directory=(
            args.image_directory.expanduser()
            if args.image_directory is not None
            else None
        ),
        audio_directory=(
            args.audio_directory.expanduser()
            if args.audio_directory is not None
            else None
        ),
        index_path=(
            args.index_path.expanduser() if args.index_path is not None else None
        ),
        qdrant_url=(
            args.qdrant_url.strip() if args.qdrant_url is not None else None
        ),
        batch_size=args.batch_size,
        image_model=args.image_model,
        audio_model=args.audio_model,
        device=args.device,
        use_fast_image_processor=args.use_fast_image_processor,
        sample_interval_ms=args.sample_interval_ms,
        resource_sampling=args.resource_sampling,
        cprofile=args.cprofile,
    )
    validate_configuration(configuration)
    return configuration


def validate_configuration(configuration: ProfileConfiguration) -> None:
    if (
        configuration.image_directory is None
        and configuration.audio_directory is None
    ):
        raise ValueError("Provide at least one image or audio directory")
    for label, directory in (
        ("Image", configuration.image_directory),
        ("Audio", configuration.audio_directory),
    ):
        if directory is not None and not directory.is_dir():
            raise ValueError(f"{label} directory does not exist: {directory}")
    if configuration.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if configuration.sample_interval_ms < 50:
        raise ValueError("Resource sample interval must be at least 50 ms")
    if (
        configuration.qdrant_url is not None
        and not configuration.qdrant_url.strip()
    ):
        raise ValueError("Qdrant URL must not be empty")
    if (
        configuration.qdrant_url is not None
        and configuration.index_path is not None
    ):
        raise ValueError("Qdrant URL cannot be combined with a local index path")
    if configuration.index_path is not None:
        path = configuration.index_path
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(
                f"Profile index path must be absent or empty: {path}"
            )
    output = configuration.output_directory
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            f"Profile output directory must be absent or empty: {output}"
        )


def run_profile(
    configuration: ProfileConfiguration,
    *,
    image_embedding_factory: Callable[[], ImageEmbedding] | None = None,
    audio_embedding_factory: Callable[[], AudioEmbedding] | None = None,
) -> ProfileResult:
    """Run actual indexing while observing it from an external layer."""
    validate_configuration(configuration)
    if configuration.qdrant_url is not None:
        _validate_empty_qdrant_service(configuration.qdrant_url)
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder()
    samples: list[ResourceSample] = []
    exit_codes: dict[str, int] = {}
    call_profiler = cProfile.Profile() if configuration.cprofile else None

    image_factory = image_embedding_factory or (
        lambda: ClipImageEmbedding(
            model_id=configuration.image_model,
            device=configuration.device,
            use_fast_processor=configuration.use_fast_image_processor,
        )
    )
    audio_factory = audio_embedding_factory or (
        lambda: ClapAudioEmbedding(
            model_id=configuration.audio_model,
            device=configuration.device,
        )
    )

    with ExitStack() as stack:
        if configuration.qdrant_url is not None:
            index_path = None
        elif configuration.index_path is None:
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="filelore-index-profile-")
            )
            index_path = Path(temporary) / "qdrant"
        else:
            index_path = configuration.index_path

        instrumentation = stack.enter_context(ExternalInstrumentation(recorder))
        profiled_image_factory = instrumentation.profiled_factory(
            image_factory, "image"
        )
        profiled_audio_factory = instrumentation.profiled_factory(
            audio_factory, "audio"
        )
        sampler: ResourceSampler | None = None
        if configuration.resource_sampling:
            sampler = ResourceSampler(
                recorder, interval_ms=configuration.sample_interval_ms
            )
            sampler.start()
            stack.callback(sampler.stop)

        # A configured service must never redirect a diagnostic run away from
        # its isolated local index.
        environment = dict(os.environ)
        environment.pop("FILELORE_QDRANT_URL", None)
        stack.enter_context(patch.dict(os.environ, environment, clear=True))

        if call_profiler is not None:
            call_profiler.enable()
        try:
            with recorder.span("overall.run"):
                if configuration.image_directory is not None:
                    exit_codes["image"] = _run_dataset(
                        configuration.image_directory,
                        "image",
                        index_path,
                        configuration.batch_size,
                        recorder,
                        qdrant_url=configuration.qdrant_url,
                        image_factory=profiled_image_factory,
                        audio_factory=profiled_audio_factory,
                    )
                if configuration.audio_directory is not None:
                    exit_codes["audio"] = _run_dataset(
                        configuration.audio_directory,
                        "audio",
                        index_path,
                        configuration.batch_size,
                        recorder,
                        qdrant_url=configuration.qdrant_url,
                        image_factory=profiled_image_factory,
                        audio_factory=profiled_audio_factory,
                    )
        finally:
            if call_profiler is not None:
                call_profiler.disable()
            if sampler is not None:
                sampler.stop()
                samples.extend(sampler.samples)

    required = [
        "overall.run",
        "planning.discovery",
        "planning.existing_lookup",
        "planning.hash",
        "storage.ensure_collection",
        "storage.prepare_and_write",
        "storage.upsert",
    ]
    for modality in exit_codes:
        required.extend(
            [
                f"overall.{modality}",
                f"{modality}.model_load",
                f"{modality}.queue",
                f"{modality}.metadata",
                f"{modality}.processing",
                f"{modality}.embedding",
                f"{modality}.model_preprocessing",
                f"{modality}.gpu_forward",
            ]
        )
    if "image" in exit_codes:
        required.append("image.decode_convert")
    if "audio" in exit_codes:
        required.extend(
            [
                "audio.segment_planning",
                "audio.decode_downmix_resample",
                "storage.delete_segments",
            ]
        )
    recorder.require_stages(required)

    discovered_files: dict[str, int] = {}
    for event in recorder.events:
        if event.stage != "planning.discovery":
            continue
        for file_type, count in event.details.get("file_counts", {}).items():
            discovered_files[file_type] = (
                discovered_files.get(file_type, 0) + int(count)
            )

    configuration_data = {
        "image_directory": (
            str(configuration.image_directory.resolve())
            if configuration.image_directory is not None
            else None
        ),
        "audio_directory": (
            str(configuration.audio_directory.resolve())
            if configuration.audio_directory is not None
            else None
        ),
        "index_mode": (
            "service"
            if configuration.qdrant_url is not None
            else "temporary"
            if configuration.index_path is None
            else "persistent"
        ),
        "index_path": (
            str(configuration.index_path) if configuration.index_path else None
        ),
        "qdrant_url": configuration.qdrant_url,
        "batch_size": configuration.batch_size,
        "image_model": configuration.image_model,
        "audio_model": configuration.audio_model,
        "device": configuration.device,
        "use_fast_image_processor": configuration.use_fast_image_processor,
        "sample_interval_ms": configuration.sample_interval_ms,
        "resource_sampling": configuration.resource_sampling,
        "discovered_files": discovered_files,
    }
    write_profile_outputs(
        configuration.output_directory,
        configuration=configuration_data,
        exit_codes=exit_codes,
        events=recorder.events,
        samples=samples,
    )
    if call_profiler is not None:
        call_profiler.dump_stats(
            str(configuration.output_directory / "cprofile.prof")
        )
    return ProfileResult(
        output_directory=configuration.output_directory,
        exit_codes=exit_codes,
        events=tuple(recorder.events),
        resource_samples=tuple(samples),
    )


def _run_dataset(
    directory: Path,
    modality: str,
    index_path: Path | None,
    batch_size: int,
    recorder: StageRecorder,
    *,
    qdrant_url: str | None = None,
    image_factory: Callable[[], ImageEmbedding],
    audio_factory: Callable[[], AudioEmbedding],
) -> int:
    arguments = [
        "--index",
        str(directory),
        "--index-type",
        modality,
        "--batch-size",
        str(batch_size),
        "--yes",
    ]
    if qdrant_url is not None:
        arguments.extend(("--qdrant-url", qdrant_url))
    else:
        if index_path is None:
            raise ValueError("Local profiling requires an index path")
        arguments.extend(("--index-path", str(index_path)))
    with recorder.span(f"overall.{modality}"):
        return filelore_main(
            arguments,
            image_embedding_factory=image_factory,
            audio_embedding_factory=audio_factory,
        )


def _validate_empty_qdrant_service(url: str) -> None:
    """Refuse to mix a diagnostic run with an existing service index."""
    nonempty: dict[str, int] = {}
    try:
        with QdrantVectorDatabase(url=url) as database:
            for collection in ("files", "files_segments"):
                if not database.collection_exists(collection):
                    continue
                count = database.count(collection)
                if count:
                    nonempty[collection] = count
    except Exception as error:
        raise RuntimeError(
            f"Could not inspect Qdrant service at {url}: {error}"
        ) from error
    if nonempty:
        counts = ", ".join(
            f"{collection}={count}" for collection, count in nonempty.items()
        )
        raise ValueError(
            "Qdrant service profiling requires empty FileLore collections; "
            f"found {counts} at {url}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        configuration = configuration_from_args(parse_args(argv))
        result = run_profile(configuration)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Profile failed: {error}")
        return 1
    print(
        json.dumps(
            {
                "successful": result.successful,
                "output_directory": str(result.output_directory.resolve()),
                "exit_codes": result.exit_codes,
            },
            indent=2,
        )
    )
    return 0 if result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
