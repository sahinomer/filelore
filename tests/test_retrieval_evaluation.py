from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from evaluation.retrieval import (
    EligibleQuery,
    RetrievalAnnotation,
    build_argument_parser,
    calculate_latency_metrics,
    calculate_retrieval_metrics,
    default_result_path,
    evaluate_queries,
    filter_indexed_annotations,
    measure_query_latency,
    parse_clotho_annotations,
    parse_coco_annotations,
    percentile,
    select_latency_queries,
    unique_file_results,
    validate_args,
)
from filelore.embedding import EmbeddingVector, TextEmbedding
from filelore.index import FileIndexEntry, FileSearchResult


def entry(identifier: str, name: str, file_type: str) -> FileIndexEntry:
    return FileIndexEntry(
        id=identifier,
        path=Path("dataset") / name,
        content_hash="hash",
        file_type=file_type,
        metadata={},
        indexed_at=datetime.now(timezone.utc),
    )


def result(identifier: str, name: str, score: float) -> FileSearchResult:
    return FileSearchResult(entry(identifier, name, "audio"), score)


class FakeTextEmbedding(TextEmbedding[str]):
    def __init__(self) -> None:
        super().__init__(model_id="fake", vector_name="fake_vector", dimensions=1)
        self.text_batches: list[tuple[str, ...]] = []

    def predict_batch(self, items: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0,) for _ in items)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        self.text_batches.append(tuple(texts))
        return tuple((1.0,) for _ in texts)


class FakeIndex:
    def __init__(self, search_results: Sequence[FileSearchResult]) -> None:
        self.search_results = tuple(search_results)
        self.file_calls = 0
        self.segment_calls = 0

    def semantic_search(self, *_: Any, **__: Any) -> tuple[FileSearchResult, ...]:
        self.file_calls += 1
        return self.search_results

    def semantic_segment_search(
        self, *_: Any, **__: Any
    ) -> tuple[FileSearchResult, ...]:
        self.segment_calls += 1
        return self.search_results


def test_parse_clotho_expands_caption_columns(tmp_path: Path) -> None:
    annotations = tmp_path / "clotho.csv"
    annotations.write_text(
        "file_name,caption_1,caption_2\n"
        "sound.wav,first sound,second sound\n",
        encoding="utf-8",
    )

    parsed = parse_clotho_annotations(annotations)

    assert parsed == (
        RetrievalAnnotation("first sound", "sound.wav"),
        RetrievalAnnotation("second sound", "sound.wav"),
    )


def test_parse_coco_joins_captions_to_filenames(tmp_path: Path) -> None:
    annotations = tmp_path / "coco.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "000007.jpg"}],
                "annotations": [
                    {"id": 1, "image_id": 7, "caption": "a test image"}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert parse_coco_annotations(annotations) == (
        RetrievalAnnotation("a test image", "000007.jpg"),
    )


def test_filter_only_keeps_unique_indexed_target_files() -> None:
    annotations = (
        RetrievalAnnotation("first", "present.JPG"),
        RetrievalAnnotation("second", "missing.jpg"),
        RetrievalAnnotation("third", "duplicate.jpg"),
    )
    indexed = (
        entry("present", "PRESENT.jpg", "image"),
        entry("wrong-type", "missing.jpg", "audio"),
        entry("duplicate-1", "duplicate.jpg", "image"),
        entry("duplicate-2", "DUPLICATE.JPG", "image"),
    )

    coverage = filter_indexed_annotations(annotations, indexed, target="image")

    assert coverage.annotation_queries == 3
    assert coverage.annotation_files == 3
    assert coverage.indexed_target_files == 3
    assert coverage.eligible_queries == (
        EligibleQuery("first", "present.JPG", "present"),
    )
    assert coverage.eligible_files == 1
    assert coverage.missing_files == ("missing.jpg",)
    assert coverage.ambiguous_files == ("duplicate.jpg",)


def test_metrics_for_one_relevant_file_per_query() -> None:
    metrics = calculate_retrieval_metrics((1, 2, None), (1, 2))

    assert metrics[0].mrr == pytest.approx(1 / 3)
    assert metrics[0].recall == pytest.approx(1 / 3)
    assert metrics[0].ndcg == pytest.approx(1 / 3)
    assert metrics[1].mrr == pytest.approx(0.5)
    assert metrics[1].recall == pytest.approx(2 / 3)
    assert metrics[1].ndcg == pytest.approx(
        (1 + 1 / 1.584962500721156) / 3
    )


def test_unique_file_results_deduplicates_audio_parents() -> None:
    results = (
        result("a", "a.wav", 0.9),
        result("a", "a.wav", 0.8),
        result("b", "b.wav", 0.7),
    )

    assert [item.file.id for item in unique_file_results(results)] == ["a", "b"]


def test_latency_uses_average_and_interpolated_percentiles() -> None:
    values = tuple(float(value) for value in range(1, 101))

    latency = calculate_latency_metrics(values)

    assert latency.average_ms == pytest.approx(50.5)
    assert latency.p95_ms == pytest.approx(95.05)
    assert latency.p99_ms == pytest.approx(99.01)
    assert percentile((10.0,), 0.99) == 10.0


def test_evaluate_audio_batches_inference_and_deduplicates_results() -> None:
    index = FakeIndex(
        (
            result("other", "other.wav", 0.9),
            result("other", "other.wav", 0.8),
            result("relevant", "relevant.wav", 0.7),
        )
    )
    clock_value = -0.01

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.01
        return clock_value

    progress: list[int] = []
    embedding = FakeTextEmbedding()

    evaluated = evaluate_queries(
        index,  # type: ignore[arg-type]
        embedding,
        (
            EligibleQuery("first sound", "relevant.wav", "relevant"),
            EligibleQuery("second sound", "relevant.wav", "relevant"),
            EligibleQuery("third sound", "relevant.wav", "relevant"),
        ),
        target="audio",
        cutoffs=(1, 2),
        candidate_limit=10,
        batch_size=2,
        clock=clock,
        on_progress=progress.append,
    )

    assert index.file_calls == 0
    assert index.segment_calls == 3
    assert embedding.text_batches == [
        ("first sound", "second sound"),
        ("third sound",),
    ]
    assert evaluated.ranks == (2, 2, 2)
    assert evaluated.metrics[0].recall == 0
    assert evaluated.metrics[1].recall == 1
    assert evaluated.throughput.batch_size == 2
    assert progress == [1, 1, 1]


def test_single_query_latency_uses_warmup_and_individual_inference() -> None:
    index = FakeIndex((result("relevant", "relevant.wav", 0.9),))
    embedding = FakeTextEmbedding()
    clock_values = iter((1.0, 1.025, 2.0, 2.030))
    queries = (
        EligibleQuery("first", "relevant.wav", "relevant"),
        EligibleQuery("second", "relevant.wav", "relevant"),
    )

    latency = measure_query_latency(
        index,  # type: ignore[arg-type]
        embedding,
        queries,
        target="audio",
        candidate_limit=10,
        warmup_queries=1,
        clock=lambda: next(clock_values),
    )

    assert index.segment_calls == 3
    assert embedding.text_batches == [("first",), ("first",), ("second",)]
    assert latency.samples == 2
    assert latency.warmup_queries == 1
    assert latency.average_ms == pytest.approx(27.5)


def test_latency_query_sampling_is_reproducible() -> None:
    queries = tuple(
        EligibleQuery(str(index), f"{index}.wav", str(index))
        for index in range(10)
    )

    first = select_latency_queries(queries, sample_size=4, seed=42)
    second = select_latency_queries(queries, sample_size=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert len({query.relevant_id for query in first}) == 4


def test_cli_validation_rejects_candidate_limit_below_cutoff(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "coco.json"
    annotations.write_text("{}", encoding="utf-8")
    index_path = tmp_path / "index"
    index_path.mkdir()
    args = build_argument_parser().parse_args(
        [
            str(annotations),
            "--target",
            "image",
            "--k",
            "1",
            "10",
            "--candidate-limit",
            "5",
            "--index-path",
            str(index_path),
        ]
    )

    with pytest.raises(ValueError, match="at least the largest cutoff"):
        validate_args(args)


def test_default_result_path_uses_ignored_evaluation_directory() -> None:
    output = default_result_path("audio")

    assert output.parent.name == "results"
    assert output.parent.parent.name == "evaluation"
    assert output.name.endswith("-audio.json")


def test_metrics_reject_empty_query_set() -> None:
    with pytest.raises(ValueError, match="without queries"):
        calculate_retrieval_metrics((), (1,))

    with pytest.raises(ValueError, match="without queries"):
        calculate_latency_metrics(())
