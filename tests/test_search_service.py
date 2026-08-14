from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import pytest

from filelore.embedding import EmbeddingVector, TextEmbedding
from filelore.index import (
    FileIndexEntry,
    FileMetadataQuery,
    FileSearchResult,
    FileSegmentMatch,
)
from filelore.search import (
    SEGMENT_GROUP_OVERFETCH_FACTOR,
    SearchRequest,
    SearchService,
    SearchSource,
    SearchTarget,
)


class RecordingEmbedding(TextEmbedding[str]):
    def __init__(self, target: str) -> None:
        super().__init__(
            model_id=f"{target}-test",
            vector_name=f"{target}_test",
            dimensions=2,
        )
        self.texts: list[str] = []
        self.close_count = 0

    def predict_batch(
        self,
        items: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0) for _ in items)

    def predict_text_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        self.texts.extend(texts)
        return tuple((0.0, 1.0) for _ in texts)

    def close(self) -> None:
        self.close_count += 1


class RecordingRepository:
    def __init__(
        self,
        *,
        file_results: Sequence[FileSearchResult] = (),
        segment_results: Sequence[FileSearchResult] = (),
    ) -> None:
        self.file_results = tuple(file_results)
        self.segment_results = tuple(segment_results)
        self.calls: list[dict[str, Any]] = []

    def semantic_search(
        self,
        vector: Sequence[float],
        **kwargs: Any,
    ) -> tuple[FileSearchResult, ...]:
        self.calls.append({"scope": "file", "vector": tuple(vector), **kwargs})
        return self.file_results

    def semantic_segment_search(
        self,
        vector: Sequence[float],
        **kwargs: Any,
    ) -> tuple[FileSearchResult, ...]:
        self.calls.append({"scope": "segment", "vector": tuple(vector), **kwargs})
        return self.segment_results


def handler(
    factory: Callable[[], RecordingEmbedding],
    *,
    vector_scope: Literal["file", "segment"] = "file",
) -> SearchTarget:
    return SearchTarget(
        embedding_factory=factory,
        vector_scope=vector_scope,
    )


def result(
    entry_id: str,
    score: float,
    *,
    segment_index: int | None = None,
) -> FileSearchResult:
    entry = FileIndexEntry(
        id=entry_id,
        path=Path(f"{entry_id}.example"),
        content_hash=f"hash-{entry_id}",
        file_type="example",
        metadata={},
        indexed_at=datetime.now().astimezone(),
    )
    segment = (
        FileSegmentMatch(segment_index, float(segment_index), segment_index + 1.0)
        if segment_index is not None
        else None
    )
    return FileSearchResult(file=entry, score=score, segment=segment)


def test_search_target_requires_a_supported_vector_scope() -> None:
    with pytest.raises(ValueError, match="scope must be file or segment"):
        SearchTarget(
            embedding_factory=lambda: RecordingEmbedding("image"),
            vector_scope="other",  # type: ignore[arg-type]
        )


def test_service_executes_text_search_reuses_model_and_reports_stages() -> None:
    repository = RecordingRepository(file_results=(result("cat", 0.9),))
    created: list[RecordingEmbedding] = []

    def factory() -> RecordingEmbedding:
        created.append(RecordingEmbedding("image"))
        return created[-1]

    service = SearchService(repository, {"image": handler(factory)})
    stages: list[str] = []
    request = SearchRequest(
        SearchSource.from_text("orange cat"),
        "image",
        FileMetadataQuery(name_contains="holiday"),
    )

    first = service.search(request, 5, on_stage=stages.append)
    second = service.search(request, 2, on_stage=stages.append)

    assert len(created) == 1
    assert created[0].texts == ["orange cat", "orange cat"]
    assert [item.result.file.id for item in first.results] == ["cat"]
    assert second.limit == 2
    assert first.query_vector_count == 1
    assert repository.calls[0]["scope"] == "file"
    assert repository.calls[0]["vector_name"] == "image_test"
    assert repository.calls[0]["limit"] == 5
    assert repository.calls[0]["metadata_filter"] is not None
    assert stages == [
        "Initializing image model…",
        "Searching image files…",
        "Searching image files…",
    ]

    service.close()
    assert created[0].close_count == 1
    assert service.active_target is None


def test_service_switches_models_when_target_changes() -> None:
    repository = RecordingRepository()
    images: list[RecordingEmbedding] = []
    audios: list[RecordingEmbedding] = []

    def image_factory() -> RecordingEmbedding:
        images.append(RecordingEmbedding("image"))
        return images[-1]

    def audio_factory() -> RecordingEmbedding:
        audios.append(RecordingEmbedding("audio"))
        return audios[-1]

    service = SearchService(
        repository,
        {
            "image": handler(image_factory),
            "audio": handler(audio_factory, vector_scope="segment"),
        },
    )

    service.search(SearchRequest(SearchSource.from_text("cat"), "image"), 5)
    service.search(SearchRequest(SearchSource.from_text("rain"), "audio"), 5)

    assert images[0].close_count == 1
    assert audios[0].close_count == 0
    assert service.active_target == "audio"
    service.close()
    assert audios[0].close_count == 1


def test_service_overfetches_and_groups_segment_results() -> None:
    repository = RecordingRepository(
        segment_results=(
            result("rain", 0.7, segment_index=0),
            result("thunder", 0.8, segment_index=0),
            result("rain", 0.9, segment_index=2),
        )
    )
    service = SearchService(
        repository,
        {
            "audio": handler(
                lambda: RecordingEmbedding("audio"),
                vector_scope="segment",
            )
        },
    )

    response = service.search(
        SearchRequest(SearchSource.from_text("storm"), "audio"),
        1,
        group_segments=True,
    )

    assert repository.calls[0]["limit"] == SEGMENT_GROUP_OVERFETCH_FACTOR
    assert len(response.results) == 1
    assert response.results[0].result.file.id == "rain"
    assert [match.score for match in response.results[0].matches] == [0.9, 0.7]
    service.close()


@pytest.mark.parametrize(
    ("search_request", "limit", "message"),
    [
        (
            SearchRequest(SearchSource.from_text("rain"), "audio"),
            5,
            "target is not enabled",
        ),
        (
            SearchRequest(SearchSource.from_text("cat"), "image"),
            0,
            "limit must be positive",
        ),
    ],
)
def test_service_rejects_invalid_request_before_loading_model(
    search_request: SearchRequest,
    limit: int,
    message: str,
) -> None:
    created: list[RecordingEmbedding] = []

    def factory() -> RecordingEmbedding:
        created.append(RecordingEmbedding("image"))
        return created[-1]

    service = SearchService(
        RecordingRepository(),
        {"image": handler(factory)},
    )

    with pytest.raises(ValueError, match=message):
        service.search(search_request, limit)

    assert created == []
