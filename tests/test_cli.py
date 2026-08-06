from __future__ import annotations

import sys
import wave
from io import StringIO
from pathlib import Path
from typing import Sequence

import pytest
from PIL import Image

from filelore.cli import (
    DEFAULT_INDEX_PATH,
    DEFAULT_RESULT_LIMIT,
    _format_duration,
    build_argument_parser,
    main,
)
from filelore.cli_display import _directory_text, _format_modified_at
from filelore.embedding import (
    AudioEmbedding,
    AudioInput,
    EmbeddingVector,
    ImageEmbedding,
)
from filelore.index import FileIndexRepository
from filelore.storage import QdrantVectorDatabase


def create_image(path: Path, *, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(25, 50, 75)).save(path)


def create_wave(path: Path, *, duration_seconds: float = 0.25) -> None:
    sample_rate = 8_000
    frame_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


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


class AudioCliEmbedding(AudioEmbedding):
    sampling_rate = 48_000
    max_length_seconds = 10.0
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(
            model_id="test-audio-model",
            vector_name="audio_test",
            dimensions=3,
        )

    def predict_batch(
        self, items: Sequence[AudioInput]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in items)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)


class TerminalStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def test_cli_defaults_to_persistent_local_qdrant_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FILELORE_QDRANT_URL", raising=False)

    args = build_argument_parser().parse_args([])

    assert args.qdrant_url is None
    assert args.index_path == DEFAULT_INDEX_PATH
    assert args.target == "image"
    assert args.index_types is None


def test_cli_accepts_repeatable_index_type_filters() -> None:
    args = build_argument_parser().parse_args(
        ["--index-type", "image", "--index-type", "audio"]
    )

    assert args.index_types == ["image", "audio"]


def test_cli_accepts_qdrant_url_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILELORE_QDRANT_URL", "http://qdrant.test:6333")

    args = build_argument_parser().parse_args([])

    assert args.qdrant_url == "http://qdrant.test:6333"


def test_cli_accepts_the_short_interactive_flag() -> None:
    args = build_argument_parser().parse_args(["-i"])

    assert args.interactive is True


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


def test_cli_formats_search_result_dates() -> None:
    assert _format_modified_at("2025-01-02T15:04:05") == (
        "Jan 2, 2025 at 15:04"
    )


def test_cli_links_search_result_directories(tmp_path: Path) -> None:
    directory = tmp_path.resolve()

    text = _directory_text(directory)

    assert any(
        getattr(span.style, "link", None) == directory.as_uri()
        for span in text.spans
    )


def test_cli_indexing_emits_only_expected_progress_feedback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    first = tmp_path / "a.png"
    second = tmp_path / "b.jpg"
    create_image(first)
    create_image(second)

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(tmp_path / "qdrant-index"),
            "--batch-size",
            "1",
        ],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    stderr_lines = captured.err.splitlines()
    assert len(stderr_lines) == 3
    assert stderr_lines[0].startswith("Discovering supported files")
    assert stderr_lines[1].startswith("Initializing image model")
    assert stderr_lines[2].startswith("Indexing images")
    assert "2/2" in stderr_lines[2]
    assert "100%" in stderr_lines[2]


def test_cli_failed_files_still_complete_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
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
    assert exit_code == 1
    assert captured.out == ""
    assert f"Could not index {invalid_path}" in captured.err
    assert "2/2" in captured.err
    assert "100%" in captured.err


def test_cli_empty_discovery_has_no_incomplete_progress_bar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

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
    assert exit_code == 0
    assert captured.out == ""
    assert "Discovering supported files" in captured.err
    assert "Indexing images" not in captured.err
    assert "0%" not in captured.err


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
    assert exit_code == 1
    assert captured.out == ""
    assert f"Could not index {invalid_path}" in captured.err


def test_cli_smart_indexing_loads_models_sequentially(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "qdrant-index"
    image_path = tmp_path / "photo.png"
    audio_path = tmp_path / "effect.wav"
    create_image(image_path)
    create_wave(audio_path)
    lifecycle: list[str] = []

    class TrackedImageEmbedding(ColorCliEmbedding):
        def close(self) -> None:
            lifecycle.append("image:close")

    class TrackedAudioEmbedding(AudioCliEmbedding):
        def close(self) -> None:
            lifecycle.append("audio:close")

    def image_factory() -> TrackedImageEmbedding:
        lifecycle.append("image:load")
        return TrackedImageEmbedding()

    def audio_factory() -> TrackedAudioEmbedding:
        assert lifecycle[-1] == "image:close"
        lifecycle.append("audio:load")
        return TrackedAudioEmbedding()

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(database_path),
        ],
        embedding_factory=image_factory,
        audio_embedding_factory=audio_factory,
    )

    assert exit_code == 0
    assert lifecycle == [
        "image:load",
        "image:close",
        "audio:load",
        "audio:close",
    ]
    with QdrantVectorDatabase(database_path) as database:
        repository = FileIndexRepository(database)
        assert repository.count() == 2
        assert database.count(repository.segment_collection_name) == 1


def test_cli_index_type_ignores_other_queues_without_loading_their_model(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "qdrant-index"
    image_path = tmp_path / "photo.png"
    audio_path = tmp_path / "effect.wav"
    create_image(image_path)
    create_wave(audio_path)

    def unexpected_audio_factory() -> AudioCliEmbedding:
        raise AssertionError("Audio model must remain unloaded")

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-type",
            "image",
            "--index-path",
            str(database_path),
        ],
        embedding_factory=ColorCliEmbedding,
        audio_embedding_factory=unexpected_audio_factory,
    )

    assert exit_code == 0
    with QdrantVectorDatabase(database_path) as database:
        repository = FileIndexRepository(database)
        assert repository.count() == 1
        assert repository.get_by_path(image_path) is not None
        assert repository.get_by_path(audio_path) is None


def test_cli_continues_other_queues_after_a_model_load_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "qdrant-index"
    image_path = tmp_path / "photo.png"
    audio_path = tmp_path / "effect.wav"
    create_image(image_path)
    create_wave(audio_path)

    def failing_image_factory() -> ColorCliEmbedding:
        raise RuntimeError("image model unavailable")

    exit_code = main(
        [
            "--index",
            str(tmp_path),
            "--index-path",
            str(database_path),
        ],
        embedding_factory=failing_image_factory,
        audio_embedding_factory=AudioCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Could not index image files: image model unavailable" in captured.err
    with QdrantVectorDatabase(database_path) as database:
        repository = FileIndexRepository(database)
        assert repository.get_by_path(image_path) is None
        assert repository.get_by_path(audio_path) is not None


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


def test_cli_rejects_explicit_interactive_search_without_a_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["-i"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires an interactive terminal" in captured.err


def test_cli_without_arguments_opens_interactive_search_on_a_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_stdin = TerminalStringIO()
    terminal_stdout = TerminalStringIO()
    monkeypatch.setattr(sys, "stdin", terminal_stdin)
    monkeypatch.setattr(sys, "stdout", terminal_stdout)
    monkeypatch.setattr("filelore.cli.DEFAULT_INDEX_PATH", tmp_path / "index")
    factory_calls: list[ColorCliEmbedding] = []
    runner_calls: list[tuple[ColorCliEmbedding, int]] = []

    def embedding_factory() -> ColorCliEmbedding:
        embedding = ColorCliEmbedding()
        factory_calls.append(embedding)
        return embedding

    def interactive_runner(
        file_index: object,
        embedding: ImageEmbedding,
        limit: int,
    ) -> int:
        assert file_index is not None
        assert isinstance(embedding, ColorCliEmbedding)
        runner_calls.append((embedding, limit))
        embedding.predict_text("red")
        embedding.predict_text("blue")
        return 0

    exit_code = main(
        [],
        embedding_factory=embedding_factory,
        interactive_runner=interactive_runner,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert len(factory_calls) == 1
    assert runner_calls == [(factory_calls[0], DEFAULT_RESULT_LIMIT)]


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


@pytest.mark.parametrize(
    "index_arguments",
    (["--no-recursive"], ["--index-type", "audio"]),
)
def test_cli_rejects_index_options_during_search(
    index_arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["red", *index_arguments])

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
    assert matching.name in captured.out
    assert "Directory" in captured.out
    assert non_matching.name not in captured.out
    assert "100% match" in captured.out
    assert "PNG" in captured.out
    assert "12 × 8 px" in captured.out
    assert "RGB" in captured.out
    assert "Modified" in captured.out
    assert "1 result" in captured.out
    assert "Timing" in captured.out
    assert "model " in captured.out
    assert "embedding " in captured.out
    assert "search " in captured.out
    assert "total " in captured.out


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
    assert capsys.readouterr().out == ""
    exit_code = main(
        ["red", "--index-path", str(database_path), "--target", "image"],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.index(red_path.name) < captured.out.index(blue_path.name)
    assert "100% match" in captured.out
    assert "0% match" in captured.out
    assert "2 results" in captured.out
    assert "Timing" in captured.out


def test_cli_shows_search_model_initialization_on_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    database_path = tmp_path / "qdrant-index"
    image_path = tmp_path / "red.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
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
        ["red", "--index-path", str(database_path)],
        embedding_factory=ColorCliEmbedding,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Initializing image model" in captured.err
    assert "Discovering supported files" not in captured.err
    assert "Initializing image model" not in captured.out
    assert image_path.name in captured.out
    assert "Directory" in captured.out
    assert "1 result" in captured.out
    assert "Timing" in captured.out


def test_cli_search_uses_score_colors_in_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "qdrant-index"
    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(red_path)
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(blue_path)
    assert (
        main(
            ["--index", str(tmp_path), "--index-path", str(database_path)],
            embedding_factory=ColorCliEmbedding,
        )
        == 0
    )
    capsys.readouterr()

    terminal_stdout = TerminalStringIO()
    monkeypatch.setattr(sys, "stdout", terminal_stdout)
    exit_code = main(
        ["red", "--index-path", str(database_path)],
        embedding_factory=ColorCliEmbedding,
    )

    rendered = terminal_stdout.getvalue()
    assert exit_code == 0
    assert "\x1b[" in rendered
    assert "[32m" in rendered or ";32m" in rendered
    assert "[31m" in rendered or ";31m" in rendered
