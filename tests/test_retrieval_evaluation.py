from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from evaluation.retrieval import (
    DocumentQueryOutcome,
    DocumentRetrievalAnnotation,
    DocumentViewQuery,
    EligibleQuery,
    RetrievalAnnotation,
    build_argument_parser,
    calculate_document_retrieval_metrics,
    calculate_latency_metrics,
    calculate_retrieval_metrics,
    default_result_path,
    document_queries_for_view,
    document_relevant_ranks,
    embed_document_queries,
    evaluate_document_queries,
    evaluate_queries,
    filter_indexed_document_annotations,
    filter_indexed_annotations,
    measure_query_latency,
    parse_clotho_annotations,
    parse_coco_annotations,
    parse_document_annotations,
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


def document_entry(
    identifier: str,
    enterprise: str,
    file_format: str,
    stem: str,
) -> FileIndexEntry:
    return FileIndexEntry(
        id=identifier,
        path=Path("dataset") / enterprise / file_format / f"{stem}.{file_format}",
        content_hash="hash",
        file_type="text",
        metadata={},
        indexed_at=datetime.now(timezone.utc),
    )


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
        self.segment_arguments: list[dict[str, Any]] = []

    def semantic_search(self, *_: Any, **__: Any) -> tuple[FileSearchResult, ...]:
        self.file_calls += 1
        return self.search_results

    def semantic_segment_search(
        self, *_: Any, **__: Any
    ) -> tuple[FileSearchResult, ...]:
        self.segment_calls += 1
        self.segment_arguments.append(__)
        return self.search_results


def test_parse_document_annotations_merges_duplicate_queries(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "documents.csv"
    annotations.write_text(
        "Enterprise Name,Query Type,Query,Supporting Facts\n"
        'Example,Descriptive,Where is it?,"[{""filename"": '
        '""first.md"", ""text"": ""one""}]"\n'
        'Example,Descriptive,Where is it?,"[{""filename"": '
        '""second.md"", ""text"": ""two""}]"\n'
        'Example,Safety,Is it safe?,"[{""filename"": '
        '""safety.md"", ""text"": ""yes""}]"\n',
        encoding="utf-8",
    )

    parsed = parse_document_annotations(annotations)

    assert parsed == (
        DocumentRetrievalAnnotation(
            enterprise="Example",
            query_type="Descriptive",
            query="Where is it?",
            relevant_files=("first.md", "second.md"),
            source_rows=2,
        ),
        DocumentRetrievalAnnotation(
            enterprise="Example",
            query_type="Other",
            query="Is it safe?",
            relevant_files=("safety.md",),
            source_rows=1,
        ),
    )


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


def test_document_coverage_keeps_partially_resolved_queries() -> None:
    annotations = (
        DocumentRetrievalAnnotation(
            enterprise="CloudWay 24",
            query_type="Comparative",
            query="Compare them",
            relevant_files=("Available.md", "Corrupted.md"),
            source_rows=2,
        ),
        DocumentRetrievalAnnotation(
            enterprise="CloudWay 24",
            query_type="Boolean",
            query="Is the missing document available?",
            relevant_files=("Missing.md",),
            source_rows=1,
        ),
    )
    indexed = (
        document_entry("md", "CloudWay-24", "md", "Available"),
        document_entry("pdf", "CloudWay-24", "pdf", "Available"),
    )

    coverage = filter_indexed_document_annotations(annotations, indexed)

    assert coverage.annotation_rows == 3
    assert coverage.annotation_queries == 2
    assert coverage.partially_covered_queries == 1
    assert coverage.fully_covered_queries == 0
    assert coverage.skipped_queries == 1
    assert coverage.missing_files == (
        "CloudWay 24/Corrupted.md",
        "CloudWay 24/Missing.md",
    )
    assert coverage.missing_format_variants == {
        "CloudWay 24/Available.md": ("docx", "html", "pptx")
    }
    assert len(coverage.eligible_queries) == 1
    assert coverage.eligible_queries[0].relevant_documents[0].logical_id
    assert document_queries_for_view(
        coverage.eligible_queries, file_format="md"
    )[0].relevant_logical_ids == (
        coverage.eligible_queries[0].relevant_documents[0].logical_id,
    )
    assert document_queries_for_view(
        coverage.eligible_queries, file_format="docx"
    ) == ()


def test_document_filename_resolution_handles_filesystem_punctuation() -> None:
    annotation = DocumentRetrievalAnnotation(
        enterprise="ZX Bank",
        query_type="Open-Ended",
        query="What can Zia do?",
        relevant_files=("ASK Zia – Your 24:7 Banking Assistant.md",),
        source_rows=1,
    )
    indexed = (
        document_entry(
            "zia",
            "ZX Bank",
            "md",
            "ASK Zia – Your 24_7 Banking Assistant",
        ),
    )

    coverage = filter_indexed_document_annotations((annotation,), indexed)

    assert len(coverage.eligible_queries) == 1
    assert coverage.missing_files == ()


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


def test_document_metrics_score_multiple_relevant_files() -> None:
    outcomes = (
        DocumentQueryOutcome(
            enterprise="Example",
            query_type="Comparative",
            query="Compare them",
            relevant_count=2,
            relevant_logical_ids=("a", "b"),
            relevant_ranks=(1, 3),
            returned_physical_files=3,
        ),
        DocumentQueryOutcome(
            enterprise="Example",
            query_type="Comparative",
            query="Find it",
            relevant_count=1,
            relevant_logical_ids=("c",),
            relevant_ranks=(),
            returned_physical_files=3,
        ),
    )

    metrics = calculate_document_retrieval_metrics(outcomes, (1, 3))

    assert metrics[0].hit == pytest.approx(0.5)
    assert metrics[0].recall == pytest.approx(0.25)
    assert metrics[0].complete == 0
    assert metrics[1].mrr == pytest.approx(0.5)
    assert metrics[1].recall == pytest.approx(0.5)
    assert metrics[1].map == pytest.approx((1 + 2 / 3) / 4)
    assert metrics[1].complete == pytest.approx(0.5)


def test_unique_file_results_deduplicates_audio_parents() -> None:
    results = (
        result("a", "a.wav", 0.9),
        result("a", "a.wav", 0.8),
        result("b", "b.wav", 0.7),
    )

    assert [item.file.id for item in unique_file_results(results)] == ["a", "b"]


def test_document_ranks_penalize_repeated_format_variants() -> None:
    results = (
        FileSearchResult(
            document_entry("a-md", "Example", "md", "A"), 0.9
        ),
        FileSearchResult(
            document_entry("a-pdf", "Example", "pdf", "A"), 0.8
        ),
        FileSearchResult(
            document_entry("b-md", "Example", "md", "B"), 0.7
        ),
    )

    ranks = document_relevant_ranks(
        results,
        ("logical-a", "logical-b"),
        {
            "a-md": "logical-a",
            "a-pdf": "logical-a",
            "b-md": "logical-b",
        },
    )

    assert ranks == (1, 3)


def test_document_evaluation_uses_segment_search_and_format_filter() -> None:
    relevant = document_entry("relevant", "Example", "md", "Relevant")
    index = FakeIndex((FileSearchResult(relevant, 0.9),))
    clock_value = -0.01

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.01
        return clock_value

    query = DocumentViewQuery(
        enterprise="Example",
        query_type="Descriptive",
        query="Find it",
        relevant_logical_ids=("logical",),
    )
    embedding = FakeTextEmbedding()
    embedding_cache = embed_document_queries(
        embedding,
        (query,),
        batch_size=2,
    )

    evaluated = evaluate_document_queries(
        index,  # type: ignore[arg-type]
        (query,),
        embedding_cache=embedding_cache,
        vector_name=embedding.vector_name,
        logical_id_by_file_id={"relevant": "logical"},
        file_format="md",
        cutoffs=(1,),
        candidate_limit=10,
        batch_size=2,
        clock=clock,
    )
    evaluate_document_queries(
        index,  # type: ignore[arg-type]
        (query,),
        embedding_cache=embedding_cache,
        vector_name=embedding.vector_name,
        logical_id_by_file_id={"relevant": "logical"},
        file_format=None,
        cutoffs=(1,),
        candidate_limit=10,
        batch_size=2,
        clock=clock,
    )

    assert index.file_calls == 0
    assert index.segment_calls == 2
    assert embedding.text_batches == [("Find it",)]
    metadata_filter = index.segment_arguments[0]["metadata_filter"]
    assert [(item.field, item.value) for item in metadata_filter.all_of] == [
        ("file_type", "text"),
        ("format_key", "md"),
    ]
    assert evaluated.metrics[0].recall == 1


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
