from __future__ import annotations

from datetime import datetime

import pytest

from filelore.index import FileMetadataQuery, file_metadata_filter
from filelore.search_query import parse_search_query
from filelore.storage import ConditionOperator


def test_interactive_query_parses_semantic_text_and_current_metadata_filters() -> None:
    parsed = parse_search_query(
        'orange cat name:"summer photo" format:jpg '
        "min-res:1280x720 max-res:3840x2160 after:2025 before:2026"
    )

    assert parsed.semantic_query == "orange cat"
    assert parsed.metadata_query.name_contains == "summer photo"
    assert parsed.metadata_query.file_format == "jpg"
    assert (parsed.metadata_query.min_width, parsed.metadata_query.min_height) == (
        1280,
        720,
    )
    assert (parsed.metadata_query.max_width, parsed.metadata_query.max_height) == (
        3840,
        2160,
    )
    assert parsed.metadata_query.modified_after == datetime(2025, 1, 1).astimezone()
    assert parsed.metadata_query.modified_before == datetime(2026, 1, 1).astimezone()
    assert parsed.filters == (
        ("name", "summer photo"),
        ("format", "jpg"),
        ("min-res", "1280x720"),
        ("max-res", "3840x2160"),
        ("after", "2025"),
        ("before", "2026"),
    )


@pytest.mark.parametrize(
    ("filter_text", "metadata_field", "expected"),
    [
        ("after:2025", "modified_after", datetime(2025, 1, 1).astimezone()),
        ("after:2024-02", "modified_after", datetime(2024, 2, 1).astimezone()),
        ("after:2024-05-06", "modified_after", datetime(2024, 5, 6).astimezone()),
        ("before:2025", "modified_before", datetime(2025, 1, 1).astimezone()),
        ("before:2024-02", "modified_before", datetime(2024, 2, 1).astimezone()),
        ("before:2024-05-06", "modified_before", datetime(2024, 5, 6).astimezone()),
    ],
)
def test_interactive_query_accepts_partial_iso_dates(
    filter_text: str,
    metadata_field: str,
    expected: datetime,
) -> None:
    parsed = parse_search_query(f"cat {filter_text}")

    assert getattr(parsed.metadata_query, metadata_field) == expected


def test_before_filter_uses_an_exclusive_partial_date_boundary() -> None:
    parsed = parse_search_query("cat before:2026")

    metadata_filter = file_metadata_filter(parsed.metadata_query)

    assert metadata_filter is not None
    before_condition = next(
        condition
        for condition in metadata_filter.all_of
        if condition.field == "modified_at"
    )
    assert before_condition.value == datetime(2026, 1, 1).astimezone()
    assert before_condition.operator is ConditionOperator.LESS_THAN


def test_audio_metadata_query_uses_exact_stream_and_strict_duration_filters() -> None:
    metadata_filter = file_metadata_filter(
        FileMetadataQuery(
            sample_rate_hz=48_000,
            bitrate_bps=192_000,
            duration_longer_than=5.0,
            duration_shorter_than=30.0,
        )
    )

    assert metadata_filter is not None
    conditions = {
        condition.field: (condition.value, condition.operator)
        for condition in metadata_filter.all_of
        if condition.field != "metadata.duration_seconds"
    }
    durations = [
        (condition.value, condition.operator)
        for condition in metadata_filter.all_of
        if condition.field == "metadata.duration_seconds"
    ]
    assert conditions == {
        "metadata.sample_rate_hz": (48_000, ConditionOperator.EQUAL),
        "metadata.bitrate_bps": (192_000, ConditionOperator.EQUAL),
    }
    assert durations == [
        (5.0, ConditionOperator.GREATER_THAN),
        (30.0, ConditionOperator.LESS_THAN),
    ]


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("format:png", "Semantic search text is required"),
        ("cat width:100", "Unknown filter: width"),
        ("cat format:", "requires a value"),
        ("cat min-res:large", "expected WIDTHxHEIGHT"),
        ("cat after:2025-13", "Invalid after date"),
        ("cat after:2026 before:2025", "after must be earlier"),
        ("cat format:png format:jpg", "may only be used once"),
    ],
)
def test_interactive_query_reports_clear_validation_errors(
    query: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_search_query(query)
