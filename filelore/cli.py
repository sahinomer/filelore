"""Command-line interface for FileLore indexing and semantic search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from filelore.embedding import ClipImageEmbedding, ImageEmbedding
from filelore.index import (
    FileIndexEntry,
    FileIndexRepository,
    FileMetadataQuery,
    file_metadata_filter,
)
from filelore.processors import ImageProcessor, PreparedFile
from filelore.storage import (
    DistanceMetric,
    QdrantVectorDatabase,
    VectorConfig,
)


EmbeddingFactory = Callable[[], ImageEmbedding]
DEFAULT_INDEX_PATH = Path.home() / ".filelore" / "qdrant"
DEFAULT_BATCH_SIZE = 100
DEFAULT_RESULT_LIMIT = 50


def configured_qdrant_url() -> str | None:
    return os.environ.get("FILELORE_QDRANT_URL")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filelore",
        description="Search the FileLore index, or add images with --index."
    )
    parser.add_argument(
        "query",
        nargs="?",
        metavar="QUERY",
        help="semantic search query (required unless --index is used)",
    )
    parser.add_argument(
        "--target",
        choices=("image",),
        default="image",
        help="embedding target (default: %(default)s)",
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
        help="only return files with this format, such as PNG or JPEG",
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
        help="index images under DIRECTORY instead of searching",
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
    embedding_factory: EmbeddingFactory | None = None,
) -> int:
    args = build_argument_parser().parse_args(argv)
    selected_embedding_factory = _embedding_factory_for_target(
        args.target,
        override=embedding_factory,
    )
    semantic_query = (args.query or "").strip()

    if args.index_directory is not None:
        if args.query is not None:
            print("Search query cannot be combined with --index", file=sys.stderr)
            return 2
        if any(
            value is not None
            for value in (
                args.name_contains,
                args.file_format,
                args.min_resolution,
                args.max_resolution,
                args.modified_after,
                args.modified_before,
                args.limit,
            )
        ):
            print("Search filters cannot be combined with --index", file=sys.stderr)
            return 2
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else DEFAULT_BATCH_SIZE
        )
        if batch_size < 1:
            print("Batch size must be positive", file=sys.stderr)
            return 2
        if not args.index_directory.expanduser().is_dir():
            print(
                f"Directory does not exist: {args.index_directory}",
                file=sys.stderr,
            )
            return 2
        metadata_query = None
    else:
        if args.no_recursive or args.batch_size is not None:
            print("Index options require --index", file=sys.stderr)
            return 2
        if not semantic_query:
            print("Search query is required unless --index is used", file=sys.stderr)
            return 2
        result_limit = (
            args.limit if args.limit is not None else DEFAULT_RESULT_LIMIT
        )
        if result_limit < 1:
            print("Search limit must be positive", file=sys.stderr)
            return 2
        try:
            metadata_query = _metadata_query(args)
        except ValueError as error:
            print(f"Invalid search: {error}", file=sys.stderr)
            return 2

    qdrant_url = args.qdrant_url
    database_target = qdrant_url or str(args.index_path)
    try:
        with QdrantVectorDatabase(
            path=args.index_path,
            url=qdrant_url,
        ) as database:
            if args.index_directory is not None:
                embedding = selected_embedding_factory()
                vector_configs = {
                    embedding.vector_name: VectorConfig(
                        embedding.dimensions,
                        distance=DistanceMetric.COSINE,
                    )
                }
                file_index = FileIndexRepository(
                    database,
                    vector_configs=vector_configs,
                )
                return _index_images(
                    file_index,
                    args.index_directory,
                    recursive=not args.no_recursive,
                    batch_size=batch_size,
                    processor=ImageProcessor(embedding=embedding),
                )
            file_index = FileIndexRepository(database)
            assert metadata_query is not None
            return _search(
                file_index,
                semantic_query,
                metadata_query,
                result_limit,
                selected_embedding_factory,
            )
    except (EOFError, KeyboardInterrupt):
        print()
        return 130
    except Exception as error:
        print(f"Could not use Qdrant at {database_target}: {error}", file=sys.stderr)
        return 2


def _embedding_factory_for_target(
    target: str,
    *,
    override: EmbeddingFactory | None = None,
) -> EmbeddingFactory:
    if target == "image":
        return override or ClipImageEmbedding
    raise ValueError(f"Unsupported embedding target: {target}")


def _index_images(
    file_index: FileIndexRepository,
    directory: Path,
    *,
    recursive: bool,
    batch_size: int,
    processor: ImageProcessor,
) -> int:
    paths = processor.discover(directory, recursive=recursive)
    had_errors = False
    pending_paths: list[Path] = []
    for path in paths:
        pending_paths.append(path)
        if len(pending_paths) >= batch_size:
            had_errors |= _process_and_store(file_index, pending_paths, processor)
            pending_paths.clear()
    had_errors |= _process_and_store(file_index, pending_paths, processor)
    return 1 if had_errors else 0


def _process_and_store(
    file_index: FileIndexRepository,
    paths: Sequence[Path],
    processor: ImageProcessor,
) -> bool:
    batch = processor.process_batch(paths)
    for failure in batch.failures:
        print(f"Could not index {failure.path}: {failure.error}", file=sys.stderr)
    _store_and_print(file_index, batch.files, processor.embedding)
    return bool(batch.failures)


def _store_and_print(
    file_index: FileIndexRepository,
    processed_files: Sequence[PreparedFile],
    embedding: ImageEmbedding | None,
) -> None:
    if not processed_files:
        return
    metadata_items = [processed.metadata for processed in processed_files]
    indexed_files = file_index.store_many(
        metadata_items,
        vector_sets=[processed.vectors for processed in processed_files],
    )
    for metadata, indexed_file in zip(metadata_items, indexed_files):
        output = metadata.to_dict()
        output["index_id"] = indexed_file.id
        output["content_hash"] = indexed_file.content_hash
        if embedding is not None:
            output["embedding"] = {
                "model_id": embedding.model_id,
                "vector_name": embedding.vector_name,
                "dimensions": embedding.dimensions,
            }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))


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

    return FileMetadataQuery(
        name_contains=args.name_contains,
        file_format=args.file_format,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        modified_after=modified_after,
        modified_before=modified_before,
    )


def _search(
    file_index: FileIndexRepository,
    semantic_query: str,
    metadata_query: FileMetadataQuery,
    limit: int,
    embedding_factory: EmbeddingFactory,
) -> int:
    total_started = perf_counter()

    initialization_started = perf_counter()
    embedding = embedding_factory()
    initialization_ms = (perf_counter() - initialization_started) * 1000

    embedding_started = perf_counter()
    query_vector = embedding.predict_text(semantic_query)
    embedding_ms = (perf_counter() - embedding_started) * 1000

    fetch_started = perf_counter()
    results = file_index.semantic_search(
        query_vector,
        vector_name=embedding.vector_name,
        limit=limit,
        metadata_filter=file_metadata_filter(metadata_query),
    )
    fetch_ms = (perf_counter() - fetch_started) * 1000
    total_ms = (perf_counter() - total_started) * 1000

    for result in results:
        _print_search_result(result.file, score=result.score)
    print(f"Found {len(results)} semantic result(s) (limit {limit}).")
    print(
        "Timing: "
        f"model initialization={_format_duration(initialization_ms)}; "
        f"query embedding={_format_duration(embedding_ms)}; "
        f"Qdrant fetch={_format_duration(fetch_ms)}; "
        f"total={_format_duration(total_ms)}."
    )
    return 0


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"
    return f"{duration_ms:.2f} ms"


def _print_search_result(
    result: FileIndexEntry,
    *,
    score: float | None = None,
) -> None:
    width = result.metadata.get("width", "?")
    height = result.metadata.get("height", "?")
    image_format = result.metadata.get(
        "image_format", result.metadata.get("extension")
    )
    modified_at = result.metadata.get("modified_at", "unknown")
    print(result.path)
    details = (
        f"  format={image_format} resolution={width}x{height} "
        f"modified={modified_at}"
    )
    if score is not None:
        details += f" score={score:.6f}"
    print(details)


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
