from __future__ import annotations

import math

import pytest

from filelore.search import build_structured_search_request


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
