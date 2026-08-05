"""Parse the compact filter syntax used by interactive semantic search."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import date, datetime, time

from filelore.index import FileMetadataQuery


_FILTER_KEYS = frozenset(
    {"name", "format", "min-res", "max-res", "after", "before"}
)
_FILTER_TOKEN = re.compile(r"^(?P<key>[A-Za-z][A-Za-z-]*):(?P<value>.*)$")
_YEAR = re.compile(r"^\d{4}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ParsedSearchQuery:
    semantic_query: str
    metadata_query: FileMetadataQuery
    filters: tuple[tuple[str, str], ...] = ()


def parse_search_query(value: str) -> ParsedSearchQuery:
    """Split semantic text from supported ``key:value`` metadata filters."""
    try:
        tokens = shlex.split(value)
    except ValueError as error:
        raise ValueError(f"Invalid query quoting: {error}") from error

    semantic_tokens: list[str] = []
    filter_values: dict[str, str] = {}
    filters: list[tuple[str, str]] = []
    for token in tokens:
        match = _FILTER_TOKEN.match(token)
        if match is None:
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

    semantic_query = " ".join(semantic_tokens).strip()
    if not semantic_query:
        raise ValueError("Semantic search text is required")

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
    modified_before = _parse_date_boundary(
        filter_values.get("before"), "before"
    )
    if (
        modified_after is not None
        and modified_before is not None
        and modified_after >= modified_before
    ):
        raise ValueError("after must be earlier than before")

    return ParsedSearchQuery(
        semantic_query=semantic_query,
        metadata_query=FileMetadataQuery(
            name_contains=filter_values.get("name"),
            file_format=filter_values.get("format"),
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            modified_after=modified_after,
            modified_before=modified_before,
        ),
        filters=tuple(filters),
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
