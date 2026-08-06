from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from PIL import Image
from textual.containers import Vertical
from textual.widgets import Collapsible, Input, Select, Static

from filelore.embedding import (
    AudioEmbedding,
    AudioInput,
    BaseEmbedding,
    EmbeddingVector,
    ImageEmbedding,
)
from filelore.index import (
    FileIndexEntry,
    FileSearchResult,
    FileSegmentMatch,
    IndexHandler,
)
from filelore.search_query import parse_search_query
from filelore.tui import (
    AUDIO_OVERFETCH_FACTOR,
    FileLoreSearchApp,
    QueryHelpScreen,
    SearchResultCard,
    SearchSession,
    group_audio_results,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecordingImageEmbedding(ImageEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="interactive-image-test-model",
            vector_name="image_interactive_test",
            dimensions=3,
        )
        self.texts: list[str] = []
        self.close_count = 0

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

    def close(self) -> None:
        self.close_count += 1


class RecordingAudioEmbedding(AudioEmbedding):
    sampling_rate = 48_000
    max_length_seconds = 10.0

    def __init__(self) -> None:
        super().__init__(
            model_id="interactive-audio-test-model",
            vector_name="audio_interactive_test",
            dimensions=3,
        )
        self.texts: list[str] = []
        self.close_count = 0

    def predict_batch(
        self,
        items: Sequence[AudioInput],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((0.0, 1.0, 0.0) for _ in items)

    def predict_text_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        self.texts.extend(texts)
        return tuple((0.0, 1.0, 0.0) for _ in texts)

    def close(self) -> None:
        self.close_count += 1


class RecordingSearchRepository:
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
        *,
        vector_name: str,
        limit: int,
        metadata_filter: Any,
    ) -> tuple[FileSearchResult, ...]:
        self.calls.append(
            {
                "scope": "file",
                "vector": tuple(vector),
                "vector_name": vector_name,
                "limit": limit,
                "metadata_filter": metadata_filter,
            }
        )
        return self.file_results

    def semantic_segment_search(
        self,
        vector: Sequence[float],
        *,
        vector_name: str,
        limit: int,
        metadata_filter: Any,
    ) -> tuple[FileSearchResult, ...]:
        self.calls.append(
            {
                "scope": "segment",
                "vector": tuple(vector),
                "vector_name": vector_name,
                "limit": limit,
                "metadata_filter": metadata_filter,
            }
        )
        return self.segment_results


def handler(
    target: str,
    factory: Callable[[], BaseEmbedding[Any]],
) -> IndexHandler:
    return IndexHandler(
        file_type=target,
        extensions=frozenset({".png" if target == "image" else ".wav"}),
        embedding_factory=factory,
        processor_factory=lambda embedding: None,  # type: ignore[arg-type]
        vector_scope="file" if target == "image" else "segment",
    )


def file_entry(path: Path, *, file_type: str, entry_id: str) -> FileIndexEntry:
    metadata: dict[str, Any] = {
        "size_bytes": 128,
        "modified_at": "2025-06-01T12:00:00+03:00",
    }
    if file_type == "image":
        metadata.update(
            image_format="PNG",
            width=12,
            height=8,
            color_mode="RGB",
        )
    else:
        metadata.update(
            audio_format="WAV",
            duration_seconds=18.0,
            sample_rate_hz=48_000,
            bitrate_bps=192_000,
        )
    return FileIndexEntry(
        id=entry_id,
        path=path,
        content_hash=f"hash-{entry_id}",
        file_type=file_type,
        metadata=metadata,
        indexed_at=datetime.now().astimezone(),
    )


def image_result(path: Path, *, score: float = 0.8) -> FileSearchResult:
    return FileSearchResult(
        file=file_entry(path, file_type="image", entry_id="image-result"),
        score=score,
    )


def audio_result(
    path: Path,
    *,
    entry_id: str,
    segment_index: int,
    score: float,
) -> FileSearchResult:
    return FileSearchResult(
        file=file_entry(path, file_type="audio", entry_id=entry_id),
        score=score,
        segment=FileSegmentMatch(
            index=segment_index,
            start_seconds=segment_index * 8.0,
            end_seconds=(segment_index + 1) * 8.0,
        ),
    )


def image_session(
    repository: RecordingSearchRepository,
    created: list[RecordingImageEmbedding],
) -> SearchSession:
    def factory() -> RecordingImageEmbedding:
        embedding = RecordingImageEmbedding()
        created.append(embedding)
        return embedding

    return SearchSession(
        repository,  # type: ignore[arg-type]
        {"image": handler("image", factory)},
        ("image",),
    )


@pytest.mark.anyio
async def test_tui_uses_target_search_layout_and_limit_options(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "cat.png"),)
    )
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=25)

    async with app.run_test(size=(100, 32)):
        assert app.SUB_TITLE == "Interactive semantic file search"
        assert app.query_one("#target", Select).disabled
        assert app._target_options() == (("Image", "image"),)
        assert app._limit_options() == (
            ("5", 5),
            ("10", 10),
            ("20", 20),
            ("25", 25),
            ("50", 50),
            ("100", 100),
        )
        assert created == []


@pytest.mark.anyio
async def test_tui_searches_only_after_enter_and_reuses_active_model(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "cat.png"),)
    )
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=25)

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        app.query_one("#limit", Select).value = 10
        query.value = "orange cat format:png after:2025"
        await pilot.pause()
        assert created == []
        assert repository.calls == []

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(created) == 1
        assert created[0].texts == ["orange cat"]
        assert repository.calls[0]["scope"] == "file"
        assert repository.calls[0]["limit"] == 10
        assert repository.calls[0]["metadata_filter"] is not None
        assert "format:png" in str(
            app.query_one("#active-filters", Static).content
        )
        assert "Found 1 file" in str(app.query_one("#status", Static).content)
        assert len(app.query(SearchResultCard)) == 1

        query.value = "blue dog before:2026"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(created) == 1
        assert created[0].texts == ["orange cat", "blue dog"]
        assert len(repository.calls) == 2


def test_session_switches_models_only_when_a_new_target_is_searched(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "cat.png"),),
        segment_results=(
            audio_result(
                tmp_path / "crash.wav",
                entry_id="audio-result",
                segment_index=0,
                score=0.9,
            ),
        ),
    )
    images: list[RecordingImageEmbedding] = []
    audios: list[RecordingAudioEmbedding] = []

    def image_factory() -> RecordingImageEmbedding:
        images.append(RecordingImageEmbedding())
        return images[-1]

    def audio_factory() -> RecordingAudioEmbedding:
        audios.append(RecordingAudioEmbedding())
        return audios[-1]

    session = SearchSession(
        repository,  # type: ignore[arg-type]
        {
            "image": handler("image", image_factory),
            "audio": handler("audio", audio_factory),
        },
        ("image", "audio"),
    )

    assert images == [] and audios == []
    session.search(parse_search_query("orange cat"), "image", 10)
    session.search(parse_search_query("blue dog"), "image", 10)
    assert len(images) == 1 and images[0].close_count == 0
    assert audios == []

    session.search(parse_search_query("glass breaking"), "audio", 10)
    assert images[0].close_count == 1
    assert len(audios) == 1 and audios[0].close_count == 0
    assert session.active_target == "audio"

    session.close()
    assert audios[0].close_count == 1


def test_audio_results_group_by_file_and_sort_chunks_by_similarity(
    tmp_path: Path,
) -> None:
    crash = tmp_path / "crash.wav"
    rain = tmp_path / "rain.wav"
    results = (
        audio_result(
            crash,
            entry_id="crash",
            segment_index=0,
            score=0.6,
        ),
        audio_result(
            rain,
            entry_id="rain",
            segment_index=0,
            score=0.8,
        ),
        audio_result(
            crash,
            entry_id="crash",
            segment_index=2,
            score=0.9,
        ),
    )

    grouped = group_audio_results(results, limit=1)

    assert len(grouped) == 1
    assert grouped[0].result.file.id == "crash"
    assert [chunk.score for chunk in grouped[0].chunks] == [0.9, 0.6]


@pytest.mark.anyio
async def test_tui_audio_search_uses_filters_overfetches_and_groups_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crash.wav"
    repository = RecordingSearchRepository(
        segment_results=(
            audio_result(
                path,
                entry_id="crash",
                segment_index=0,
                score=0.7,
            ),
            audio_result(
                path,
                entry_id="crash",
                segment_index=2,
                score=0.9,
            ),
        )
    )
    audios: list[RecordingAudioEmbedding] = []

    def factory() -> RecordingAudioEmbedding:
        audios.append(RecordingAudioEmbedding())
        return audios[-1]

    session = SearchSession(
        repository,  # type: ignore[arg-type]
        {"audio": handler("audio", factory)},
        ("audio",),
    )
    app = FileLoreSearchApp(session, limit=10)

    async with app.run_test(size=(110, 38)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "glass breaking sample-rate:48000 longer-than:1"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(audios) == 1
        assert audios[0].texts == ["glass breaking"]
        assert repository.calls[0]["scope"] == "segment"
        assert repository.calls[0]["limit"] == 10 * AUDIO_OVERFETCH_FACTOR
        assert repository.calls[0]["metadata_filter"] is not None
        assert len(app.query(SearchResultCard)) == 1
        assert len(app.query(Collapsible)) == 1
        status = str(app.query_one("#status", Static).content)
        assert "Found 1 file" in status
        assert "grouped 2 audio chunks into 1 file" in status


@pytest.mark.anyio
async def test_tui_help_tracks_selected_target(tmp_path: Path) -> None:
    repository = RecordingSearchRepository()
    session = SearchSession(
        repository,  # type: ignore[arg-type]
        {
            "image": handler("image", RecordingImageEmbedding),
            "audio": handler("audio", RecordingAudioEmbedding),
        },
        ("image", "audio"),
    )
    app = FileLoreSearchApp(session, limit=20)

    async with app.run_test(size=(100, 32)) as pilot:
        target = app.query_one("#target", Select)
        assert not target.disabled
        target.value = "audio"
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()

        assert isinstance(app.screen, QueryHelpScreen)
        help_content = str(app.screen.query_one("#help-content", Static).content)
        assert "sample-rate:48000" in help_content
        assert "min-res:1280x720" not in help_content

        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#query", Input).has_focus


@pytest.mark.anyio
async def test_tui_validation_errors_do_not_load_a_model(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=25)

    async with app.run_test(size=(100, 32)) as pilot:
        app.query_one("#query", Input).value = "cat sample-rate:48000"
        await pilot.press("enter")
        await pilot.pause()

        assert created == []
        assert repository.calls == []
        assert "Audio metadata filters require the audio target" in str(
            app.query_one("#status", Static).content
        )


@pytest.mark.anyio
async def test_clear_search_restores_empty_results_message(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "cat.png"),)
    )
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=20)

    async with app.run_test(size=(100, 32)) as pilot:
        app.query_one("#query", Input).value = "cat"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.press("ctrl+l")
        await pilot.pause()

        results = app.query_one("#results", Vertical)
        assert "Search results will appear here." in str(
            results.query_one(Static).content
        )
