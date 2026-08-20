"""Parse compact semantic text and ``key:value`` metadata filters."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time

from filelore.documents import SUPPORTED_DOCUMENT_EXTENSIONS
from filelore.index import FileMetadataQuery
from filelore.metadata import AudioMetadataParser, ImageMetadataParser


_FILTER_KEYS = frozenset(
    {
        "name",
        "format",
        "min-res",
        "max-res",
        "sample-rate",
        "bitrate",
        "longer-than",
        "shorter-than",
        "after",
        "before",
    }
)
_FILTER_TOKEN = re.compile(r"^(?P<key>[A-Za-z][A-Za-z-]*):(?P<value>.*)$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_YEAR = re.compile(r"^\d{4}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ParsedSearchQuery:
    semantic_query: str
    metadata_query: FileMetadataQuery
    filters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedSearchFilters:
    metadata_query: FileMetadataQuery
    filters: tuple[tuple[str, str], ...] = ()


def parse_search_query(value: str) -> ParsedSearchQuery:
    """Split semantic text from supported metadata filter tokens."""
    semantic_tokens, filter_values, filters = _split_search_tokens(value)
    semantic_query = " ".join(semantic_tokens).strip()
    if not semantic_query:
        raise ValueError("Semantic search text is required")
    return ParsedSearchQuery(
        semantic_query=semantic_query,
        metadata_query=_metadata_query(filter_values),
        filters=tuple(filters),
    )


def parse_search_filters(value: str) -> ParsedSearchFilters:
    """Parse filter-only input used alongside a separate file query."""
    semantic_tokens, filter_values, filters = _split_search_tokens(value)
    if semantic_tokens:
        raise ValueError("File search filters must use key:value syntax")
    return ParsedSearchFilters(
        metadata_query=_metadata_query(filter_values),
        filters=tuple(filters),
    )


def validate_search_target(parsed_query: ParsedSearchQuery, target: str) -> None:
    """Reject metadata filters incompatible with the selected file type."""
    validate_search_metadata(parsed_query.metadata_query, target)


def validate_search_metadata(query: FileMetadataQuery, target: str) -> None:
    """Reject metadata filters incompatible with the selected file type."""
    if target != "image" and any(
        value is not None
        for value in (
            query.min_width,
            query.min_height,
            query.max_width,
            query.max_height,
        )
    ):
        raise ValueError("Resolution filters require the image target")
    if target != "audio" and any(
        value is not None
        for value in (
            query.sample_rate_hz,
            query.bitrate_bps,
            query.duration_longer_than,
            query.duration_shorter_than,
        )
    ):
        raise ValueError("Audio metadata filters require the audio target")
    if query.file_format:
        inferred_target = target_for_format(query.file_format)
        if inferred_target is not None and inferred_target != target:
            raise ValueError(
                f"Format {query.file_format!r} is {inferred_target}, not {target}"
            )


def target_for_format(file_format: str) -> str | None:
    """Return the unique supported target associated with a file extension."""
    extension = f".{file_format.strip().removeprefix('.').casefold()}"
    matches = tuple(
        file_type
        for file_type, extensions in (
            ("image", ImageMetadataParser.supported_extensions),
            ("audio", AudioMetadataParser.supported_extensions),
            ("text", SUPPORTED_DOCUMENT_EXTENSIONS),
        )
        if extension in extensions
    )
    return matches[0] if len(matches) == 1 else None


def _split_search_tokens(
    value: str,
) -> tuple[list[str], dict[str, str], list[tuple[str, str]]]:
    tokens = _tokenize_search_value(value)
    semantic_tokens: list[str] = []
    filter_values: dict[str, str] = {}
    filters: list[tuple[str, str]] = []
    for token in tokens:
        match = _FILTER_TOKEN.match(token)
        if match is None or _WINDOWS_ABSOLUTE_PATH.match(token):
            semantic_tokens.append(token)
            continue
        key = match.group("key").casefold()
        raw_value = match.group("value").strip()
        if key not in _FILTER_KEYS:
            raise ValueError(f"Unknown filter: {key}")
        if not raw_value:
            raise ValueError(f"Filter '{key}' requires a value")
        if key in filter_values:
            raise ValueError(f"Filter '{key}' may only be used once")
        filter_values[key] = raw_value
        filters.append((key, raw_value))
    return semantic_tokens, filter_values, filters


def _tokenize_search_value(value: str) -> list[str]:
    """Split query tokens while preserving Windows path separators."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in value:
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
        elif character in {'"', "'"} and (
            not current or current[-1] == ":"
        ):
            quote = character
        elif character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(character)
    if quote is not None:
        raise ValueError("Invalid query quoting: No closing quotation")
    if current:
        tokens.append("".join(current))
    return tokens


def _metadata_query(filter_values: dict[str, str]) -> FileMetadataQuery:
    min_width, min_height = _parse_resolution_filter(
        filter_values.get("min-res"), "min-res"
    )
    max_width, max_height = _parse_resolution_filter(
        filter_values.get("max-res"), "max-res"
    )
    if (
        min_width is not None
        and min_height is not None
        and max_width is not None
        and max_height is not None
        and (min_width > max_width or min_height > max_height)
    ):
        raise ValueError("min-res cannot exceed max-res")
    modified_after = _parse_date_boundary(filter_values.get("after"), "after")
    modified_before = _parse_date_boundary(filter_values.get("before"), "before")
    if (
        modified_after is not None
        and modified_before is not None
        and modified_after >= modified_before
    ):
        raise ValueError("after must be earlier than before")
    longer_than = _parse_duration_filter(
        filter_values.get("longer-than"), "longer-than", allow_zero=True
    )
    shorter_than = _parse_duration_filter(
        filter_values.get("shorter-than"), "shorter-than", allow_zero=False
    )
    if (
        longer_than is not None
        and shorter_than is not None
        and longer_than >= shorter_than
    ):
        raise ValueError("longer-than must be less than shorter-than")
    return FileMetadataQuery(
        name_contains=filter_values.get("name"),
        file_format=filter_values.get("format"),
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        sample_rate_hz=_parse_positive_int_filter(
            filter_values.get("sample-rate"), "sample-rate"
        ),
        bitrate_bps=_parse_positive_int_filter(
            filter_values.get("bitrate"), "bitrate"
        ),
        duration_longer_than=longer_than,
        duration_shorter_than=shorter_than,
        modified_after=modified_after,
        modified_before=modified_before,
    )


def _parse_resolution_filter(
    value: str | None,
    key: str,
) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    normalized = value.casefold().replace("×", "x")
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise ValueError(f"Invalid {key}; expected WIDTHxHEIGHT") from error
    if width < 1 or height < 1:
        raise ValueError(f"Invalid {key}; values must be positive")
    return width, height


def _parse_positive_int_filter(value: str | None, key: str) -> int | None:
    if value is None:
        return None
    try:
        prepared = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid {key}; expected a positive integer") from error
    if prepared < 1:
        raise ValueError(f"Invalid {key}; value must be positive")
    return prepared


def _parse_duration_filter(
    value: str | None,
    key: str,
    *,
    allow_zero: bool,
) -> float | None:
    if value is None:
        return None
    try:
        prepared = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {key}; expected seconds") from error
    if (
        not math.isfinite(prepared)
        or prepared < 0
        or (prepared == 0 and not allow_zero)
    ):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Invalid {key}; value must be {requirement}")
    return prepared


def _parse_date_boundary(value: str | None, key: str) -> datetime | None:
    if value is None:
        return None
    try:
        if _YEAR.fullmatch(value):
            parsed = datetime(int(value), 1, 1)
        elif _MONTH.fullmatch(value):
            year_text, month_text = value.split("-")
            parsed = datetime(int(year_text), int(month_text), 1)
        elif _DAY.fullmatch(value):
            parsed = datetime.combine(date.fromisoformat(value), time.min)
        else:
            parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid {key} date; use YYYY, YYYY-MM, YYYY-MM-DD, "
            "or an ISO datetime"
        ) from error
    return parsed.astimezone() if parsed.tzinfo is None else parsed
