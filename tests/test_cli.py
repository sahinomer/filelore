from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest
from PIL import Image

from filelore.cli import (
    DEFAULT_INDEX_PATH,
    _format_duration,
    build_argument_parser,
    main,
)
from filelore.embedding import EmbeddingVector, ImageEmbedding


def create_image(path: Path, *, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(25, 50, 75)).save(path)


class ColorCliEmbedding(ImageEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="test-color-model",
            vector_name="image_test_color",
            dimensions=3,
        )

    def predict_batch(
        self, items: Sequence[str | Path | Image.Image]
    ) -> tuple[EmbeddingVector, ...]:
        vectors: list[tuple[float, float, float]] = []
        for item in items:
            if isinstance(item, Image.Image):
                prepared = item.convert("RGB")
                try:
                    color = prepared.getpixel((0, 0))
                finally:
                    prepared.close()
            else:
                with Image.open(item) as image:
                    color = image.convert("RGB").getpixel((0, 0))
            vectors.append(tuple(float(value) for value in color))
        return self._prepare_vectors(
            vectors,
            expected_count=len(items),
            normalize=True,
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        colors = {
            "red": (1.0, 0.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }
        return self._prepare_vectors(
            [colors[text] for text in texts],
            expected_count=len(texts),
            normalize=True,
        )


def test_cli_defaults_to_persistent_local_qdrant_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FILELORE_QDRANT_URL", raising=False)

    args = build_argument_parser().parse_args([])

    assert args.qdrant_url is None
    assert args.index_path == DEFAULT_INDEX_PATH
    assert args.target == "image"


def test_cli_accepts_qdrant_url_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILELORE_QDRANT_URL", "http://qdrant.test:6333")

    args = build_argument_parser().parse_args([])

    assert args.qdrant_url == "http://qdrant.test:6333"


def test_cli_parses_semantic_query_and_optional_search_filters() -> None:
    args = build_argument_parser().parse_args(
        [
            "a red sports car",
            "--target",
            "image",
            "--name",
            "holiday",
            "--format",
            "jpeg",
            "--min-resolution",
            "1280x720",
            "--max-resolution",
            "3840x2160",
            "--modified-after",
            "2025-01-01",
            "--modified-before",
            "2025-12-31",
        ]
    )

    assert args.query == "a red sports car"
    assert args.target == "image"
    assert args.name_contains == "holiday"
    assert args.file_format == "jpeg"
    assert args.min_resolution == "1280x720"
    assert args.max_resolution == "3840x2160"
    assert args.modified_after == "2025-01-01"
    assert args.modified_before == "2025-12-31"


def test_cli_formats_durations_with_readable_units() -> None:
    assert _format_duration(10_476.421) == "10.48 s"
    assert _format_duration(162.858) == "162.86 ms"
    assert _format_duration(1.788) == "1.79 ms"


def test_cli_prints_one_json_record_per_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "nested" / "b.jpg"
    create_image(first)
    create_image(second)

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(tmp_path / "qdrant-index"),
        ],
        embedding_factory=ColorCliEmbedding,
    )

    assert exit_code == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["path"] for record in records] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert all(record["width"] == 12 for record in records)
    assert all(record["content_hash"] for record in records)
    assert all(record["index_id"] for record in records)
    assert all(
        record["embedding"]["vector_name"] == "image_test_color"
        for record in records
    )


def test_cli_reports_invalid_images_and_indexes_the_remaining_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.png"
    create_image(valid_path)
    invalid_path.write_text("not an image", encoding="utf-8")

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(tmp_path / "qdrant-index"),
        ],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert exit_code == 1
    assert [record["path"] for record in records] == [str(valid_path.resolve())]
    assert f"Could not index {invalid_path}" in captured.err


def test_cli_reports_a_missing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--index", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Directory does not exist" in captured.err


def test_cli_requires_a_query_when_not_indexing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Search query is required unless --index is used" in captured.err


def test_cli_rejects_a_search_query_with_indexing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["red", "--index", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Search query cannot be combined with --index" in captured.err


def test_cli_rejects_search_filters_with_indexing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--index", str(tmp_path), "--name", "holiday"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Search filters cannot be combined with --index" in captured.err


def test_cli_reports_an_invalid_resolution_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["red", "--min-resolution", "large"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Invalid minimum resolution" in captured.err


def test_cli_rejects_index_options_during_search(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["red", "--no-recursive"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Index options require --index" in captured.err


def test_cli_searches_with_optional_metadata_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = tmp_path / "qdrant-index"
    matching = tmp_path / "SummerPhoto.PNG"
    non_matching = tmp_path / "notes.jpg"
    create_image(matching, size=(12, 8))
    create_image(non_matching, size=(30, 20))
    assert (
        main(
            [
                "--index",
                str(tmp_path),
                "--index-path",
                str(database_path),
            ],
            embedding_factory=ColorCliEmbedding,
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "red",
            "--index-path",
            str(database_path),
            "--name",
            "summer",
            "--format",
            "png",
            "--min-resolution",
            "10x5",
            "--max-resolution",
            "20x10",
            "--modified-after",
            "2000-01-01",
            "--modified-before",
            "2100-01-01",
        ],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(matching.resolve()) in captured.out
    assert str(non_matching.resolve()) not in captured.out
    assert "score=" in captured.out
    assert "Found 1 semantic result(s)" in captured.out
    assert "model initialization=" in captured.out
    assert "query embedding=" in captured.out
    assert "Qdrant fetch=" in captured.out
    assert "total=" in captured.out


def test_cli_indexes_embeddings_and_searches_by_semantic_description(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = tmp_path / "qdrant-index"
    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(red_path)
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(blue_path)

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(database_path),
        ],
        embedding_factory=ColorCliEmbedding,
    )

    assert exit_code == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(
        record["embedding"]
        == {
            "dimensions": 3,
            "model_id": "test-color-model",
            "vector_name": "image_test_color",
        }
        for record in records
    )
    exit_code = main(
        ["red", "--index-path", str(database_path), "--target", "image"],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.index(str(red_path.resolve())) < captured.out.index(
        str(blue_path.resolve())
    )
    assert "score=" in captured.out
    assert "Found 2 semantic result(s)" in captured.out
    assert "model initialization=" in captured.out
    assert "query embedding=" in captured.out
    assert "Qdrant fetch=" in captured.out
