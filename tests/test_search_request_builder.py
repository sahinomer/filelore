from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from filelore.embedding import BaseEmbedding, EmbeddingVector
from filelore.search import (
    build_interactive_search_request,
    build_structured_search_request,
)


class ExampleFileVectorizer:
    supported_extensions = frozenset({".example", ".png"})

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        raise AssertionError("Request construction must not embed the query file")


def test_interactive_text_request_separates_semantics_and_filters() -> None:
    request = build_interactive_search_request(
        "  orange cat format:png min-res:640x480  ",
        target="image",
    )

    assert request.source.text == "orange cat"
    assert request.target == "image"
    assert request.metadata_query.file_format == "png"
    assert (request.metadata_query.min_width, request.metadata_query.min_height) == (
        640,
        480,
    )
    assert request.filters == (("format", "png"), ("min-res", "640x480"))


def test_interactive_file_request_validates_path_and_filter_only_input(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.png"
    query_path.write_bytes(b"query")

    request = build_interactive_search_request(
        "name:holiday after:2025",
        target="image",
        file_query_vectorizers={"image": ExampleFileVectorizer()},
        query_file=query_path,
    )

    assert request.source.file == query_path.resolve()
    assert request.source.text is None
    assert request.metadata_query.name_contains == "holiday"
    assert request.filters == (("name", "holiday"), ("after", "2025"))


def test_interactive_request_detects_relative_file_and_infers_target(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.png"
    query_path.write_bytes(b"query")

    request = build_interactive_search_request(
        "reference.png name:holiday",
        target="audio",
        file_query_vectorizers={"image": ExampleFileVectorizer()},
        base_directory=tmp_path,
    )

    assert request.source.file == query_path.resolve()
    assert request.target == "image"
    assert request.filters == (("name", "holiday"),)


def test_interactive_request_reports_missing_path_like_query(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Query file does not exist"):
        build_interactive_search_request(
            "missing/reference.png",
            target="image",
            file_query_vectorizers={"image": ExampleFileVectorizer()},
            base_directory=tmp_path,
        )


def test_structured_request_builds_text_source_target_and_metadata() -> None:
    request = build_structured_search_request(
        text="  ocean waves  ",
        query_file=None,
        explicit_target="audio",
        file_query_vectorizers={},
        file_format="wav",
        sample_rate_hz=48_000,
        duration_longer_than=1.5,
        duration_shorter_than=30.0,
    )

    assert request.source.text == "ocean waves"
    assert request.target == "audio"
    assert request.metadata_query.file_format == "wav"
    assert request.metadata_query.sample_rate_hz == 48_000
    assert request.metadata_query.duration_longer_than == 1.5
    assert request.metadata_query.duration_shorter_than == 30.0


def test_structured_file_request_infers_target_from_extension(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.png"
    query_path.write_bytes(b"query")

    request = build_structured_search_request(
        text=None,
        query_file=query_path,
        explicit_target=None,
        file_query_vectorizers={"image": ExampleFileVectorizer()},
    )

    assert request.target == "image"
    assert request.source.file == query_path.resolve()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"text": "cat", "query_file": Path("cat.example")},
            "cannot be combined",
        ),
        ({"text": "  ", "query_file": None}, "query or --query-file"),
        (
            {
                "text": "cat",
                "query_file": None,
                "explicit_target": "audio",
                "file_format": "png",
            },
            "is image, not audio",
        ),
    ],
)
def test_structured_request_rejects_conflicting_or_missing_source_fields(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "text": None,
        "query_file": None,
        "explicit_target": "image",
        "file_query_vectorizers": {},
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_structured_search_request(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("duration_longer_than", "longer-than duration must be finite"),
        ("duration_shorter_than", "shorter-than duration must be finite"),
    ],
)
def test_structured_request_rejects_non_finite_duration_filters(
    field: str,
    message: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_structured_search_request(
            text="rain",
            query_file=None,
            explicit_target="audio",
            file_query_vectorizers={},
            **{field: value},
        )
