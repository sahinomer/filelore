from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pytest

from filelore.embedding import BaseEmbedding, EmbeddingVector, TextEmbedding
from filelore.index import FileIndexEntry, FileSearchResult, FileSegmentMatch
from filelore.search import SearchSource, embed_search_source, search_vectors


class RecordingTextEmbedding(TextEmbedding[str]):
    def __init__(self) -> None:
        super().__init__(model_id="text-test", vector_name="text_test", dimensions=2)
        self.texts: list[str] = []

    def predict_batch(
        self, items: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0) for _ in items)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        self.texts.extend(texts)
        return tuple((0.0, 1.0) for _ in texts)


class RecordingFileVectorizer:
    supported_extensions = frozenset({".example"})

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def predict_file(
        self,
        path: Path,
        embedding: BaseEmbedding[Any],
    ) -> tuple[EmbeddingVector, ...]:
        assert embedding.vector_name == "text_test"
        self.paths.append(path)
        return ((1.0, 0.0), (0.0, 1.0))


class RecordingRepository:
    def __init__(
        self,
        results: Sequence[Sequence[FileSearchResult]],
    ) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def semantic_search(self, vector: Sequence[float], **kwargs: Any):
        self.calls.append({"scope": "file", "vector": tuple(vector), **kwargs})
        return tuple(self.results.pop(0))

    def semantic_segment_search(self, vector: Sequence[float], **kwargs: Any):
        self.calls.append({"scope": "segment", "vector": tuple(vector), **kwargs})
        return tuple(self.results.pop(0))


def result(
    entry_id: str,
    score: float,
    *,
    segment_index: int | None = None,
) -> FileSearchResult:
    entry = FileIndexEntry(
        id=entry_id,
        path=Path(f"{entry_id}.bin"),
        content_hash=f"hash-{entry_id}",
        file_type="example",
        metadata={},
        indexed_at=datetime.now().astimezone(),
    )
    segment = (
        FileSegmentMatch(segment_index, 0.0, 1.0)
        if segment_index is not None
        else None
    )
    return FileSearchResult(file=entry, score=score, segment=segment)


def test_search_source_requires_exactly_one_non_empty_input() -> None:
    assert SearchSource.from_text("  example query ").text == "example query"
    assert SearchSource.from_file("query.example").file == Path("query.example")

    with pytest.raises(ValueError, match="exactly one"):
        SearchSource()
    with pytest.raises(ValueError, match="exactly one"):
        SearchSource(text="query", file=Path("query.example"))
    with pytest.raises(ValueError, match="must not be empty"):
        SearchSource.from_text(" ")


def test_embed_search_source_supports_text_and_file_vectors() -> None:
    embedding = RecordingTextEmbedding()
    vectorizer = RecordingFileVectorizer()

    assert embed_search_source(
        SearchSource.from_text("orange cat"), embedding
    ) == ((0.0, 1.0),)
    assert embed_search_source(
        SearchSource.from_file("query.example"),
        embedding,
        file_vectorizer=vectorizer,
    ) == ((1.0, 0.0), (0.0, 1.0))
    assert embedding.texts == ["orange cat"]
    assert vectorizer.paths == [Path("query.example")]


def test_search_vectors_merges_multi_vector_results_by_best_score() -> None:
    repository = RecordingRepository(
        (
            (result("one", 0.7, segment_index=0), result("two", 0.8, segment_index=0)),
            (result("one", 0.9, segment_index=0), result("one", 0.6, segment_index=1)),
        )
    )

    results = search_vectors(
        repository,
        ((1.0, 0.0), (0.0, 1.0)),
        vector_name="example_vector",
        vector_scope="segment",
        limit=2,
    )

    assert [(item.file.id, item.segment.index, item.score) for item in results if item.segment] == [
        ("one", 0, 0.9),
        ("two", 0, 0.8),
    ]
    assert [call["scope"] for call in repository.calls] == ["segment", "segment"]
    assert all(call["limit"] == 2 for call in repository.calls)


@pytest.mark.parametrize(
    ("vectors", "vector_scope", "limit", "message"),
    (
        ((), "file", 10, "at least one"),
        (((1.0,),), "other", 10, "file or segment"),
        (((1.0,),), "file", 0, "positive"),
    ),
)
def test_search_vectors_rejects_invalid_requests(
    vectors: Sequence[Sequence[float]],
    vector_scope: Any,
    limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        search_vectors(
            RecordingRepository(()),
            vectors,
            vector_name="example",
            vector_scope=vector_scope,
            limit=limit,
        )
