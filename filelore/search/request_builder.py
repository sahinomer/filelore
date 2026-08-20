"""Build complete search requests from interactive or structured fields."""

from __future__ import annotations

from datetime import date, datetime, time
from math import isfinite
from pathlib import Path
from typing import Mapping

from filelore.index import FileMetadataQuery
from filelore.search.execution import validate_query_file
from filelore.search.models import SearchRequest, SearchSource
from filelore.search.protocols import FileQueryVectorizer
from filelore.search.query_parser import (
    parse_search_filters,
    parse_search_query,
    target_for_format,
    validate_search_metadata,
)


def build_interactive_search_request(
    value: str,
    *,
    target: str,
    file_query_vectorizers: Mapping[str, FileQueryVectorizer] | None = None,
    query_file: Path | None = None,
    base_directory: Path | None = None,
) -> SearchRequest:
    """Build a TUI request from unified text, file, and filter input."""
    vectorizers = file_query_vectorizers or {}
    if query_file is None:
        parsed = parse_search_query(value)
        detected = _detect_interactive_query_file(
            parsed.semantic_query,
            vectorizers,
            base_directory=base_directory,
        )
        if detected is None:
            request = SearchRequest(
                source=SearchSource.from_text(parsed.semantic_query),
                target=target,
                metadata_query=parsed.metadata_query,
                filters=parsed.filters,
            )
        else:
            path, file_target = detected
            request = SearchRequest(
                source=SearchSource.from_file(path),
                target=file_target,
                metadata_query=parsed.metadata_query,
                filters=parsed.filters,
            )
    else:
        path, file_target = _prepare_interactive_query_file(
            query_file,
            vectorizers,
            base_directory=base_directory,
        )
        parsed_filters = parse_search_filters(value)
        request = SearchRequest(
            source=SearchSource.from_file(path),
            target=file_target,
            metadata_query=parsed_filters.metadata_query,
            filters=parsed_filters.filters,
        )
    validate_search_metadata(request.metadata_query, request.target)
    return request


def _detect_interactive_query_file(
    value: str,
    file_query_vectorizers: Mapping[str, FileQueryVectorizer],
    *,
    base_directory: Path | None,
) -> tuple[Path, str] | None:
    candidate = _resolve_query_path(value, base_directory)
    if not candidate.is_file():
        if _looks_like_file_path(value):
            raise ValueError(f"Query file does not exist: {candidate}")
        return None
    return _prepare_interactive_query_file(
        candidate,
        file_query_vectorizers,
        base_directory=None,
    )


def _prepare_interactive_query_file(
    path: Path,
    file_query_vectorizers: Mapping[str, FileQueryVectorizer],
    *,
    base_directory: Path | None,
) -> tuple[Path, str]:
    candidate = _resolve_query_path(path, base_directory)
    file_target = target_for_format(candidate.suffix)
    if file_target is None:
        extension = candidate.suffix or "<none>"
        raise ValueError(f"Unsupported query file extension: {extension}")
    vectorizer = file_query_vectorizers.get(file_target)
    if vectorizer is None:
        raise ValueError(f"File similarity search is not enabled for {file_target}")
    return (
        validate_query_file(candidate, vectorizer.supported_extensions),
        file_target,
    )


def _resolve_query_path(value: str | Path, base_directory: Path | None) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and base_directory is not None:
        candidate = base_directory / candidate
    return candidate


def _looks_like_file_path(value: str) -> bool:
    candidate = Path(value).expanduser()
    return (
        candidate.is_absolute()
        or value.startswith((".", "~"))
        or "/" in value
        or "\\" in value
        or target_for_format(candidate.suffix) is not None
    )


def build_structured_search_request(
    *,
    text: str | None,
    query_file: Path | None,
    explicit_target: str | None,
    file_query_vectorizers: Mapping[str, FileQueryVectorizer],
    name_contains: str | None = None,
    file_format: str | None = None,
    min_resolution: str | None = None,
    max_resolution: str | None = None,
    sample_rate_hz: int | None = None,
    bitrate_bps: int | None = None,
    duration_longer_than: float | None = None,
    duration_shorter_than: float | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> SearchRequest:
    """Build a validated request from separately supplied search fields."""
    semantic_text = (text or "").strip()
    if text is not None and query_file is not None:
        raise ValueError("A text query cannot be combined with --query-file")
    if not semantic_text and query_file is None:
        raise ValueError(
            "A text query or --query-file is required unless --index is used"
        )
    metadata_query = _structured_metadata_query(
        name_contains=name_contains,
        file_format=file_format,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        sample_rate_hz=sample_rate_hz,
        bitrate_bps=bitrate_bps,
        duration_longer_than=duration_longer_than,
        duration_shorter_than=duration_shorter_than,
        modified_after=modified_after,
        modified_before=modified_before,
    )
    target = resolve_search_target(explicit_target, file_format, query_file)
    validate_search_metadata(metadata_query, target)
    if query_file is None:
        source = SearchSource.from_text(semantic_text)
    else:
        vectorizer = file_query_vectorizers.get(target)
        if vectorizer is None:
            raise ValueError(f"File similarity search is not enabled for {target}")
        source = SearchSource.from_file(
            validate_query_file(query_file, vectorizer.supported_extensions)
        )
    return SearchRequest(source=source, target=target, metadata_query=metadata_query)


def resolve_search_target(
    explicit_target: str | None,
    file_format: str | None,
    query_file: Path | None = None,
) -> str:
    """Resolve a target from explicit, result-format, and query-file hints."""
    inferred_target = target_for_format(file_format) if file_format else None
    if (
        explicit_target is not None
        and inferred_target is not None
        and explicit_target != inferred_target
    ):
        raise ValueError(
            f"Format {file_format!r} is {inferred_target}, not {explicit_target}"
        )
    selected_target = explicit_target or inferred_target
    query_target = (
        target_for_format(query_file.suffix) if query_file is not None else None
    )
    if query_file is not None and query_target is None:
        extension = query_file.suffix or "<none>"
        raise ValueError(f"Unsupported query file extension: {extension}")
    if (
        selected_target is not None
        and query_target is not None
        and selected_target != query_target
    ):
        raise ValueError(f"Query file is {query_target}, not {selected_target}")
    target = selected_target or query_target
    if target is None:
        raise ValueError(
            "Search file type is required; use --target image, audio, or text"
        )
    return target


def _structured_metadata_query(
    *,
    name_contains: str | None,
    file_format: str | None,
    min_resolution: str | None,
    max_resolution: str | None,
    sample_rate_hz: int | None,
    bitrate_bps: int | None,
    duration_longer_than: float | None,
    duration_shorter_than: float | None,
    modified_after: str | None,
    modified_before: str | None,
) -> FileMetadataQuery:
    min_width, min_height = _parse_resolution(min_resolution, "minimum")
    max_width, max_height = _parse_resolution(max_resolution, "maximum")
    if (
        min_width is not None
        and min_height is not None
        and max_width is not None
        and max_height is not None
        and (min_width > max_width or min_height > max_height)
    ):
        raise ValueError("minimum resolution cannot exceed maximum resolution")
    if sample_rate_hz is not None and sample_rate_hz < 1:
        raise ValueError("sample rate must be positive")
    if bitrate_bps is not None and bitrate_bps < 1:
        raise ValueError("bitrate must be positive")
    if duration_longer_than is not None:
        if not isfinite(duration_longer_than):
            raise ValueError("longer-than duration must be finite")
        if duration_longer_than < 0:
            raise ValueError("longer-than duration must be non-negative")
    if duration_shorter_than is not None:
        if not isfinite(duration_shorter_than):
            raise ValueError("shorter-than duration must be finite")
        if duration_shorter_than <= 0:
            raise ValueError("shorter-than duration must be positive")
    if (
        duration_longer_than is not None
        and duration_shorter_than is not None
        and duration_longer_than >= duration_shorter_than
    ):
        raise ValueError("longer-than duration must be less than shorter-than")
    after = _parse_datetime(modified_after, end_of_day=False)
    before = _parse_datetime(modified_before, end_of_day=True)
    if after is not None and before is not None and after > before:
        raise ValueError("modified-after cannot be later than modified-before")
    return FileMetadataQuery(
        name_contains=name_contains,
        file_format=file_format,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        sample_rate_hz=sample_rate_hz,
        bitrate_bps=bitrate_bps,
        duration_longer_than=duration_longer_than,
        duration_shorter_than=duration_shorter_than,
        modified_after=after,
        modified_before=before,
    )


def _parse_resolution(
    value: str | None,
    label: str,
) -> tuple[int | None, int | None]:
    normalized = (value or "").strip().casefold().replace("×", "x")
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


def _parse_datetime(value: str | None, *, end_of_day: bool) -> datetime | None:
    normalized = (value or "").strip()
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
