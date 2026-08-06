"""Command-line interface for FileLore indexing and semantic search."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from filelore.cli_display import CliDisplay
from filelore.embedding import (
    AudioEmbedding,
    BaseEmbedding,
    ClapAudioEmbedding,
    ClipImageEmbedding,
    ImageEmbedding,
    TextEmbedding,
)
from filelore.index import (
    FileIndexRepository,
    FileIndexer,
    FileMetadataQuery,
    IndexCoordinator,
    IndexHandler,
    IndexWorkQueue,
    file_metadata_filter,
    normalized_path,
)
from filelore.metadata import AudioMetadataParser, ImageMetadataParser
from filelore.processors import AudioProcessor, ImageProcessor
from filelore.storage import QdrantVectorDatabase, VectorDatabase


EmbeddingFactory = Callable[[], TextEmbedding[Any]]
ImageEmbeddingFactory = Callable[[], ImageEmbedding]
AudioEmbeddingFactory = Callable[[], AudioEmbedding]
InteractiveRunner = Callable[
    [
        FileIndexRepository,
        Mapping[str, IndexHandler],
        Sequence[str],
        int,
    ],
    int,
]
DEFAULT_INDEX_PATH = Path.home() / ".filelore" / "qdrant"
DEFAULT_BATCH_SIZE = 100
DEFAULT_RESULT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class IndexQueueOutcome:
    """Successful and failed records from one confirmed work queue."""

    added: int = 0
    updated: int = 0
    failed: int = 0

    @property
    def had_errors(self) -> bool:
        return self.failed > 0


def configured_qdrant_url() -> str | None:
    return os.environ.get("FILELORE_QDRANT_URL")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filelore",
        description="Search the FileLore index, or add supported files with --index."
    )
    parser.add_argument(
        "query",
        nargs="?",
        metavar="QUERY",
        help=(
            "semantic search query (omit in a terminal to open interactive "
            "search)"
        ),
    )
    parser.add_argument(
        "--target",
        "--type",
        dest="target",
        choices=("image", "audio"),
        help="search file type; required unless --format implies it",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="open the persistent full-screen search interface",
    )

    search_options = parser.add_argument_group("search options")
    search_options.add_argument(
        "--name",
        dest="name_contains",
        metavar="TEXT",
        help="only return files whose names contain TEXT",
    )
    search_options.add_argument(
        "--format",
        dest="file_format",
        metavar="FORMAT",
        help="only return files with this format, such as PNG, JPEG, WAV, or MP3",
    )
    search_options.add_argument(
        "--min-resolution",
        metavar="WIDTHxHEIGHT",
        help="minimum image resolution",
    )
    search_options.add_argument(
        "--max-resolution",
        metavar="WIDTHxHEIGHT",
        help="maximum image resolution",
    )
    search_options.add_argument(
        "--sample-rate",
        type=int,
        metavar="HZ",
        help="only return audio with this sample rate",
    )
    search_options.add_argument(
        "--bitrate",
        type=int,
        metavar="BPS",
        help="only return audio with this bitrate",
    )
    search_options.add_argument(
        "--longer-than",
        type=float,
        metavar="SECONDS",
        help="only return audio longer than this duration",
    )
    search_options.add_argument(
        "--shorter-than",
        type=float,
        metavar="SECONDS",
        help="only return audio shorter than this duration",
    )
    search_options.add_argument(
        "--modified-after",
        metavar="DATE",
        help="only return files modified after this date or ISO datetime",
    )
    search_options.add_argument(
        "--modified-before",
        metavar="DATE",
        help="only return files modified before this date or ISO datetime",
    )
    search_options.add_argument(
        "--limit",
        type=int,
        help=f"maximum search results to display (default: {DEFAULT_RESULT_LIMIT})",
    )

    index_options = parser.add_argument_group("index options")
    index_options.add_argument(
        "--index",
        dest="index_directory",
        type=Path,
        metavar="DIRECTORY",
        help="index supported files under DIRECTORY instead of searching",
    )
    index_options.add_argument(
        "--index-type",
        dest="index_types",
        action="append",
        choices=("image", "audio"),
        metavar="TYPE",
        help=(
            "only index this file type; repeat to select multiple types "
            "(default: all supported types)"
        ),
    )
    index_options.add_argument(
        "--no-recursive",
        action="store_true",
        help="only inspect files directly inside the directory",
    )
    index_options.add_argument(
        "--batch-size",
        type=int,
        help=(
            "number of records written to Qdrant per batch "
            f"(default: {DEFAULT_BATCH_SIZE})"
        ),
    )
    index_options.add_argument(
        "-y",
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="index all new and changed files without confirmation",
    )

    storage_options = parser.add_argument_group("storage options")
    storage_options.add_argument(
        "--qdrant-url",
        default=configured_qdrant_url(),
        help=(
            "use a Qdrant service instead of Python Local Mode; substantially "
            "faster for large indexes (or set FILELORE_QDRANT_URL)"
        ),
    )
    storage_options.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=(
            "local Qdrant index directory "
            f"(default: {DEFAULT_INDEX_PATH}); a configured Qdrant URL takes "
            "precedence"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    image_embedding_factory: ImageEmbeddingFactory | None = None,
    audio_embedding_factory: AudioEmbeddingFactory | None = None,
    interactive_runner: InteractiveRunner | None = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_argument_parser().parse_args(raw_argv)
    display = CliDisplay()
    embedding_factories = _embedding_factories(
        image=image_embedding_factory,
        audio=audio_embedding_factory,
    )
    index_handlers = _index_handlers(embedding_factories)
    handlers_by_type = {
        handler.file_type: handler for handler in index_handlers
    }
    search_target: str | None = None
    semantic_query = (args.query or "").strip()
    interactive_terminal = _is_interactive_terminal()
    interactive_mode = args.interactive or (
        not raw_argv and interactive_terminal
    )

    if interactive_mode:
        if args.query is not None or args.index_directory is not None:
            display.print_error(
                "Interactive search cannot be combined with a query or --index"
            )
            return 2
        if not interactive_terminal:
            display.print_error(
                "Interactive search requires an interactive terminal"
            )
            return 2
        if any(
            value is not None
            for value in (
                args.name_contains,
                args.file_format,
                args.min_resolution,
                args.max_resolution,
                args.sample_rate,
                args.bitrate,
                args.longer_than,
                args.shorter_than,
                args.modified_after,
                args.modified_before,
            )
        ):
            display.print_error(
                "Enter search filters inside the interactive search interface"
            )
            return 2
        if (
            args.no_recursive
            or args.batch_size is not None
            or args.index_types is not None
            or args.assume_yes
        ):
            display.print_error("Index options require --index")
            return 2
        result_limit = (
            args.limit if args.limit is not None else DEFAULT_RESULT_LIMIT
        )
        if result_limit < 1:
            display.print_error("Search limit must be positive")
            return 2
        metadata_query = None
    elif args.index_directory is not None:
        if args.query is not None:
            display.print_error("Search query cannot be combined with --index")
            return 2
        if any(
            value is not None
            for value in (
                args.name_contains,
                args.target,
                args.file_format,
                args.min_resolution,
                args.max_resolution,
                args.sample_rate,
                args.bitrate,
                args.longer_than,
                args.shorter_than,
                args.modified_after,
                args.modified_before,
                args.limit,
            )
        ):
            display.print_error("Search filters cannot be combined with --index")
            return 2
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else DEFAULT_BATCH_SIZE
        )
        if batch_size < 1:
            display.print_error("Batch size must be positive")
            return 2
        if not args.index_directory.expanduser().is_dir():
            display.print_error(
                f"Directory does not exist: {args.index_directory}"
            )
            return 2
        metadata_query = None
    else:
        if (
            args.no_recursive
            or args.batch_size is not None
            or args.index_types is not None
            or args.assume_yes
        ):
            display.print_error("Index options require --index")
            return 2
        if not semantic_query:
            display.print_error(
                "Search query is required unless --index is used"
            )
            return 2
        result_limit = (
            args.limit if args.limit is not None else DEFAULT_RESULT_LIMIT
        )
        if result_limit < 1:
            display.print_error("Search limit must be positive")
            return 2
        try:
            metadata_query = _metadata_query(args)
            search_target = _resolve_search_target(
                args.target,
                args.file_format,
            )
            _validate_target_filters(args, search_target)
        except ValueError as error:
            display.print_error(f"Invalid search: {error}")
            return 2

    qdrant_url = args.qdrant_url
    database_target = qdrant_url or str(args.index_path)
    try:
        with QdrantVectorDatabase(
            path=args.index_path,
            url=qdrant_url,
        ) as database:
            if args.index_directory is not None:
                coordinator = IndexCoordinator(index_handlers)
                return _index_directory(
                    database,
                    args.index_directory,
                    recursive=not args.no_recursive,
                    batch_size=batch_size,
                    allowed_types=args.index_types,
                    assume_yes=args.assume_yes,
                    coordinator=coordinator,
                    display=display,
                )
            file_index = FileIndexRepository(database)
            if interactive_mode:
                allowed_targets = (
                    (args.target,)
                    if args.target is not None
                    else tuple(handlers_by_type)
                )
                runner = interactive_runner or _run_interactive_search
                return runner(
                    file_index,
                    handlers_by_type,
                    allowed_targets,
                    result_limit,
                )
            assert metadata_query is not None
            assert search_target is not None
            return _search(
                file_index,
                semantic_query,
                metadata_query,
                result_limit,
                handlers_by_type[search_target],
                display,
            )
    except (EOFError, KeyboardInterrupt):
        display.print_error()
        return 130
    except Exception as error:
        display.print_error(
            f"Could not use Qdrant at {database_target}: {error}"
        )
        return 2


def _embedding_factories(
    *,
    image: ImageEmbeddingFactory | None = None,
    audio: AudioEmbeddingFactory | None = None,
) -> dict[str, EmbeddingFactory]:
    """Return the enabled default target-to-model factory registry."""
    return {
        "image": image or ClipImageEmbedding,
        "audio": audio or ClapAudioEmbedding,
    }


def _resolve_search_target(
    explicit_target: str | None,
    file_format: str | None,
) -> str:
    inferred_target = _target_for_format(file_format) if file_format else None
    if (
        explicit_target is not None
        and inferred_target is not None
        and explicit_target != inferred_target
    ):
        raise ValueError(
            f"Format {file_format!r} is {inferred_target}, not {explicit_target}"
        )
    if explicit_target is not None:
        return explicit_target
    if inferred_target is not None:
        return inferred_target
    raise ValueError(
        "Search file type is required; use --target image or --target audio"
    )


def _target_for_format(file_format: str) -> str | None:
    extension = f".{file_format.strip().removeprefix('.').casefold()}"
    matches = tuple(
        file_type
        for file_type, extensions in (
            ("image", ImageMetadataParser.supported_extensions),
            ("audio", AudioMetadataParser.supported_extensions),
        )
        if extension in extensions
    )
    return matches[0] if len(matches) == 1 else None


def _validate_target_filters(args: argparse.Namespace, target: str) -> None:
    if target == "audio" and (
        args.min_resolution is not None or args.max_resolution is not None
    ):
        raise ValueError("Resolution filters require the image target")
    if target == "image" and any(
        value is not None
        for value in (
            args.sample_rate,
            args.bitrate,
            args.longer_than,
            args.shorter_than,
        )
    ):
        raise ValueError("Audio metadata filters require the audio target")


def _is_interactive_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _can_prompt() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _run_interactive_search(
    file_index: FileIndexRepository,
    handlers: Mapping[str, IndexHandler],
    allowed_targets: Sequence[str],
    limit: int,
) -> int:
    from filelore.tui import run_interactive_search

    return run_interactive_search(
        file_index,
        handlers,
        allowed_targets,
        limit,
    )


def _index_handlers(
    embedding_factories: Mapping[str, EmbeddingFactory],
) -> tuple[IndexHandler, ...]:
    return (
        IndexHandler(
            file_type="image",
            extensions=ImageMetadataParser.supported_extensions,
            embedding_factory=embedding_factories["image"],
            processor_factory=_image_processor_for,
            vector_scope="file",
        ),
        IndexHandler(
            file_type="audio",
            extensions=AudioMetadataParser.supported_extensions,
            embedding_factory=embedding_factories["audio"],
            processor_factory=_audio_processor_for,
            vector_scope="segment",
        ),
    )


def _image_processor_for(
    embedding: BaseEmbedding[Any],
) -> ImageProcessor:
    if not isinstance(embedding, ImageEmbedding):
        raise TypeError("Image index handler requires an image embedding")
    return ImageProcessor(embedding=embedding)


def _audio_processor_for(
    embedding: BaseEmbedding[Any],
) -> AudioProcessor:
    if not isinstance(embedding, AudioEmbedding):
        raise TypeError("Audio index handler requires an audio embedding")
    return AudioProcessor(embedding=embedding)


def _index_directory(
    database: VectorDatabase,
    directory: Path,
    *,
    recursive: bool,
    batch_size: int,
    allowed_types: Sequence[str] | None,
    assume_yes: bool,
    coordinator: IndexCoordinator,
    display: CliDisplay,
) -> int:
    with display.status("Discovering supported files…"):
        plan = coordinator.discover(
            directory,
            recursive=recursive,
            allowed_types=allowed_types,
        )

    progress_labels = (
        "Checking file changes",
        *(f"Indexing {queue.file_type} files" for queue in plan.queues),
    )
    progress_label_width = max(len(label) for label in progress_labels)
    repository = FileIndexRepository(database)
    with display.indexing(
        plan.total_files,
        label="Checking file changes",
        label_width=progress_label_width,
    ) as progress:
        work_plan = coordinator.classify_changes(
            plan,
            repository,
            on_progress=progress.advance,
        )

    display.print_index_discovery(work_plan)
    had_errors = False
    for queue in work_plan.queues:
        if queue.failures:
            had_errors = True
            for failure in queue.failures:
                display.print_error(
                    f"Could not inspect {failure.path}: {failure.error}"
                )

    if work_plan.work_count and not assume_yes and not _can_prompt():
        display.print_error(
            "Indexing requires confirmation in a terminal; rerun with --yes"
        )
        return 2

    for queue in work_plan.queues:
        if not queue.candidates:
            continue
        if not assume_yes and not display.confirm(
            _index_confirmation_message(queue)
        ):
            display.print_skipped(queue.file_type, queue.work_count)
            continue
        try:
            with ExitStack() as model_session:
                with display.status(
                    f"Initializing {queue.file_type} model…"
                ):
                    file_indexer = model_session.enter_context(
                        queue.handler.open_indexer(database)
                    )
                outcome = _index_queue(
                    file_indexer,
                    queue,
                    batch_size=batch_size,
                    coordinator=coordinator,
                    display=display,
                    label_width=progress_label_width,
                )
                had_errors |= outcome.had_errors
                display.print_index_result(
                    queue.file_type,
                    added=outcome.added,
                    updated=outcome.updated,
                    failed=outcome.failed,
                )
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception as error:
            display.print_error(
                f"Could not index {queue.file_type} files: {error}"
            )
            had_errors = True
    return 1 if had_errors else 0


def _index_confirmation_message(queue: IndexWorkQueue) -> str:
    file_label = "file" if queue.work_count == 1 else "files"
    details: list[str] = []
    if queue.new_count:
        details.append(f"{queue.new_count} new")
    if queue.updated_count:
        details.append(f"{queue.updated_count} changed")
    return (
        f"Index {queue.work_count} {queue.file_type} {file_label} "
        f"({', '.join(details)})?"
    )


def _index_queue(
    file_indexer: FileIndexer[Any],
    queue: IndexWorkQueue,
    *,
    batch_size: int,
    coordinator: IndexCoordinator,
    display: CliDisplay,
    label_width: int,
) -> IndexQueueOutcome:
    added = 0
    updated = 0
    failed = 0
    with display.indexing(
        queue.work_count,
        label=f"Indexing {queue.file_type} files",
        label_width=label_width,
    ) as progress:
        for candidates in coordinator.work_batches(queue, batch_size):
            try:
                result = file_indexer.index_candidates(candidates)
                indexed_paths = {
                    normalized_path(entry.path) for entry in result.entries
                }
                for candidate in candidates:
                    if normalized_path(candidate.path) not in indexed_paths:
                        continue
                    if candidate.change == "new":
                        added += 1
                    else:
                        updated += 1
                if result.failures:
                    failed += len(result.failures)
                    with display.suspend():
                        for failure in result.failures:
                            display.print_error(
                                f"Could not index {failure.path}: {failure.error}"
                            )
            finally:
                progress.advance(len(candidates))
    return IndexQueueOutcome(added=added, updated=updated, failed=failed)


def _metadata_query(args: argparse.Namespace) -> FileMetadataQuery:
    min_width, min_height = _parse_resolution(
        args.min_resolution or "", "minimum"
    )
    max_width, max_height = _parse_resolution(
        args.max_resolution or "", "maximum"
    )
    modified_after = _parse_datetime(
        args.modified_after or "",
        end_of_day=False,
    )
    modified_before = _parse_datetime(
        args.modified_before or "",
        end_of_day=True,
    )
    if (
        min_width is not None
        and min_height is not None
        and max_width is not None
        and max_height is not None
        and (min_width > max_width or min_height > max_height)
    ):
        raise ValueError("minimum resolution cannot exceed maximum resolution")
    if (
        modified_after is not None
        and modified_before is not None
        and modified_after > modified_before
    ):
        raise ValueError("modified-after cannot be later than modified-before")
    if args.sample_rate is not None and args.sample_rate < 1:
        raise ValueError("sample rate must be positive")
    if args.bitrate is not None and args.bitrate < 1:
        raise ValueError("bitrate must be positive")
    if args.longer_than is not None and args.longer_than < 0:
        raise ValueError("longer-than duration must be non-negative")
    if args.shorter_than is not None and args.shorter_than <= 0:
        raise ValueError("shorter-than duration must be positive")
    if (
        args.longer_than is not None
        and args.shorter_than is not None
        and args.longer_than >= args.shorter_than
    ):
        raise ValueError("longer-than duration must be less than shorter-than")

    return FileMetadataQuery(
        name_contains=args.name_contains,
        file_format=args.file_format,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        sample_rate_hz=args.sample_rate,
        bitrate_bps=args.bitrate,
        duration_longer_than=args.longer_than,
        duration_shorter_than=args.shorter_than,
        modified_after=modified_after,
        modified_before=modified_before,
    )


def _search(
    file_index: FileIndexRepository,
    semantic_query: str,
    metadata_query: FileMetadataQuery,
    limit: int,
    handler: IndexHandler,
    display: CliDisplay,
) -> int:
    total_started = perf_counter()

    initialization_started = perf_counter()
    with display.status(f"Initializing {handler.file_type} model…"):
        embedding = handler.embedding_factory()
    initialization_ms = (perf_counter() - initialization_started) * 1000

    try:
        if not isinstance(embedding, TextEmbedding):
            raise TypeError(
                f"{handler.file_type.title()} search requires a text embedding"
            )
        embedding_started = perf_counter()
        query_vector = embedding.predict_text(semantic_query)
        embedding_ms = (perf_counter() - embedding_started) * 1000

        fetch_started = perf_counter()
        search = {
            "file": file_index.semantic_search,
            "segment": file_index.semantic_segment_search,
        }[handler.vector_scope]
        results = search(
            query_vector,
            vector_name=embedding.vector_name,
            limit=limit,
            metadata_filter=file_metadata_filter(metadata_query),
        )
        fetch_ms = (perf_counter() - fetch_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000
    finally:
        embedding.close()

    display.print_search_results(
        results,
        query=semantic_query,
        limit=limit,
        timings=(
            ("model", _format_duration(initialization_ms)),
            ("embedding", _format_duration(embedding_ms)),
            ("search", _format_duration(fetch_ms)),
            ("total", _format_duration(total_ms)),
        ),
    )
    return 0


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"
    return f"{duration_ms:.2f} ms"


def _parse_resolution(value: str, label: str) -> tuple[int | None, int | None]:
    normalized = value.strip().casefold().replace("×", "x")
    if not normalized:
        return None, None
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise ValueError(
            f"Invalid {label} resolution; expected WIDTHxHEIGHT"
        ) from error
    if width < 1 or height < 1:
        raise ValueError(f"Invalid {label} resolution; values must be positive")
    return width, height


def _parse_datetime(value: str, *, end_of_day: bool) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        if len(normalized) == 10:
            parsed_date = date.fromisoformat(normalized)
            boundary = time.max if end_of_day else time.min
            parsed = datetime.combine(parsed_date, boundary)
        else:
            parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "Invalid datetime; use YYYY-MM-DD or an ISO datetime"
        ) from error
    return parsed.astimezone() if parsed.tzinfo is None else parsed
