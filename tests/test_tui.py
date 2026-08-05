from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pytest
from PIL import Image
from textual.widgets import Input, Select, Static

from filelore.embedding import EmbeddingVector, ImageEmbedding
from filelore.index import FileIndexEntry, FileSearchResult
from filelore.tui import FileLoreSearchApp, QueryHelpScreen, SearchSession


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecordingTextEmbedding(ImageEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="interactive-test-model",
            vector_name="image_interactive_test",
            dimensions=3,
        )
        self.texts: list[str] = []

    def predict_batch(
        self,
        items: Sequence[str | Path | Image.Image],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in items)

    def predict_text_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        self.texts.extend(texts)
        return tuple((1.0, 0.0, 0.0) for _ in texts)


class RecordingSearchRepository:
    def __init__(self, result: FileSearchResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def semantic_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int,
        metadata_filter: Any,
    ) -> tuple[FileSearchResult, ...]:
        self.calls.append(
            {
                "vector": tuple(vector),
                "vector_name": vector_name,
                "limit": limit,
                "metadata_filter": metadata_filter,
            }
        )
        return (self.result,)


def search_result(path: Path) -> FileSearchResult:
    entry = FileIndexEntry(
        id="result-id",
        path=path,
        content_hash="content-hash",
        file_type="image",
        metadata={
            "image_format": "PNG",
            "width": 12,
            "height": 8,
            "color_mode": "RGB",
            "size_bytes": 128,
            "modified_at": "2025-06-01T12:00:00+03:00",
        },
        indexed_at=datetime.now().astimezone(),
    )
    return FileSearchResult(file=entry, score=0.8)


@pytest.mark.anyio
async def test_tui_searches_only_after_enter_and_reuses_the_session(
    tmp_path: Path,
) -> None:
    embedding = RecordingTextEmbedding()
    repository = RecordingSearchRepository(search_result(tmp_path / "cat.png"))
    app = FileLoreSearchApp(
        SearchSession(repository, embedding),  # type: ignore[arg-type]
        limit=25,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        assert app.SUB_TITLE == "Interactive semantic file search"
        assert len(app.query("#query-help")) == 0
        assert len(app.query("#intro")) == 0
        assert app._limit_options() == (
            ("5", 5),
            ("10", 10),
            ("20", 20),
            ("25", 25),
            ("50", 50),
            ("100", 100),
        )

        limit_select = app.query_one("#limit", Select)
        limit_select.value = 10
        query.value = "orange cat format:png after:2025"
        await pilot.pause()
        assert embedding.texts == []
        assert repository.calls == []

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert embedding.texts == ["orange cat"]
        assert len(repository.calls) == 1
        assert repository.calls[0]["limit"] == 10
        assert repository.calls[0]["metadata_filter"] is not None
        assert "format:png" in str(
            app.query_one("#active-filters", Static).content
        )
        status = str(app.query_one("#status", Static).content)
        assert "Found 1 result in" in status
        assert "Timing" not in str(app.query_one("#results", Static).content)

        query.value = "blue dog before:2026"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert embedding.texts == ["orange cat", "blue dog"]
        assert len(repository.calls) == 2
        assert "before:2026" in str(
            app.query_one("#active-filters", Static).content
        )


@pytest.mark.anyio
async def test_tui_opens_query_help_as_a_modal(tmp_path: Path) -> None:
    embedding = RecordingTextEmbedding()
    repository = RecordingSearchRepository(search_result(tmp_path / "cat.png"))
    app = FileLoreSearchApp(
        SearchSession(repository, embedding),  # type: ignore[arg-type]
        limit=50,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("f1")
        await pilot.pause()

        assert isinstance(app.screen, QueryHelpScreen)
        help_content = str(app.screen.query_one("#help-content", Static).content)
        assert "name:holiday" in help_content
        assert "after:2025" in help_content

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen is app.screen_stack[0]
        assert app.query_one("#query", Input).has_focus


@pytest.mark.anyio
async def test_tui_shows_query_validation_errors_without_searching(
    tmp_path: Path,
) -> None:
    embedding = RecordingTextEmbedding()
    repository = RecordingSearchRepository(search_result(tmp_path / "cat.png"))
    app = FileLoreSearchApp(
        SearchSession(repository, embedding),  # type: ignore[arg-type]
        limit=25,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        app.query_one("#query", Input).value = "cat after:2025-13"
        await pilot.press("enter")
        await pilot.pause()

        assert embedding.texts == []
        assert repository.calls == []
        assert "Invalid after date" in str(
            app.query_one("#status", Static).content
        )
