from __future__ import annotations

import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from PIL import Image
from textual.containers import Vertical
from textual.widgets import Button, Collapsible, Input, Select, Static

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
)
from filelore.search import (
    AudioFileQueryVectorizer,
    ImageFileQueryVectorizer,
    SEGMENT_GROUP_OVERFETCH_FACTOR,
    SearchRequest,
    SearchService,
    SearchSource,
    SearchTarget,
    group_segment_results,
    parse_search_query,
)
from filelore.tui import (
    FileLoreSearchApp,
    QueryHelpScreen,
    SearchResultCard,
)
from filelore.ui import (
    FilePickerScreen,
    FileSystemPathSuggester,
    QueryBar,
    SupportedFileTree,
    TailPathLabel,
    shorten_path_from_start,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def create_wave(path: Path, *, duration_seconds: float = 0.25) -> None:
    sample_rate = 8_000
    frame_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


def test_long_attachment_paths_are_shortened_from_the_start() -> None:
    path = r"collection\several\nested\directories\reference-audio.wav"

    shortened = shorten_path_from_start(path, 24)

    assert shortened.startswith("...")
    assert shortened.endswith("reference-audio.wav")
    assert len(shortened) <= 24


@pytest.mark.anyio
async def test_path_suggester_completes_supported_files_and_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "samples").mkdir()
    (tmp_path / "reference.wav").touch()
    (tmp_path / "reference recording.wav").touch()
    (tmp_path / "readme.txt").touch()
    suggester = FileSystemPathSuggester(tmp_path, {".wav", ".png"})

    assert await suggester.get_suggestion("reference.") == "reference.wav"
    assert await suggester.get_suggestion("reference r") == (
        "reference recording.wav"
    )
    assert await suggester.get_suggestion("sam") == f"samples{os.sep}"
    assert await suggester.get_suggestion("readm") is None


class RecordingImageEmbedding(ImageEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="interactive-image-test-model",
            vector_name="image_interactive_test",
            dimensions=3,
        )
        self.texts: list[str] = []
        self.images: list[Path] = []
        self.close_count = 0

    def predict_batch(
        self,
        items: Sequence[str | Path | Image.Image],
    ) -> tuple[EmbeddingVector, ...]:
        self.images.extend(
            Path(item) for item in items if not isinstance(item, Image.Image)
        )
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
        self.audio_inputs: list[AudioInput] = []
        self.close_count = 0

    def predict_batch(
        self,
        items: Sequence[AudioInput],
    ) -> tuple[EmbeddingVector, ...]:
        self.audio_inputs.extend(items)
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
) -> SearchTarget:
    return SearchTarget(
        embedding_factory=factory,
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
        file=file_entry(
            path,
            file_type="image",
            entry_id=f"image-{path.name}",
        ),
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
) -> SearchService:
    def factory() -> RecordingImageEmbedding:
        embedding = RecordingImageEmbedding()
        created.append(embedding)
        return embedding

    return SearchService(
        repository,  # type: ignore[arg-type]
        {"image": handler("image", factory)},
        ("image",),
        file_query_vectorizers={"image": ImageFileQueryVectorizer()},
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
        assert len(app.query("#query-mode")) == 0
        assert len(app.query("#file-filters")) == 0
        assert app.query_one(QueryBar).attached_file is None
        browse = app.query_one("#browse-query-file", Button)
        assert str(browse.label) == "Browse"
        assert browse.content_region.width >= len("Browse")
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
async def test_tui_file_picker_filters_files_for_selected_target(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    audio_path = tmp_path / "reference.wav"
    unsupported_path = tmp_path / "notes.txt"
    image_path.touch()
    create_wave(audio_path)
    unsupported_path.touch()
    session = SearchService(
        RecordingSearchRepository(),  # type: ignore[arg-type]
        {
            "image": handler("image", RecordingImageEmbedding),
            "audio": handler("audio", RecordingAudioEmbedding),
        },
        ("image", "audio"),
        file_query_vectorizers={
            "image": ImageFileQueryVectorizer(),
            "audio": AudioFileQueryVectorizer(),
        },
    )
    app = FileLoreSearchApp(
        session,
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query_bar = app.query_one(QueryBar)
        query_bar.value = "text that will be replaced"

        await pilot.click("#browse-query-file")
        await pilot.pause()

        assert isinstance(app.screen, FilePickerScreen)
        tree = app.screen.query_one(SupportedFileTree)
        assert set(
            tree.filter_paths((image_path, audio_path, unsupported_path))
        ) == {
            image_path
        }
        app.screen.dismiss(None)
        await pilot.pause()

        app.query_one("#target", Select).value = "audio"
        await pilot.pause()
        await pilot.click("#browse-query-file")
        await pilot.pause()

        assert isinstance(app.screen, FilePickerScreen)
        tree = app.screen.query_one(SupportedFileTree)
        assert set(
            tree.filter_paths((image_path, audio_path, unsupported_path))
        ) == {
            audio_path
        }
        app.screen.dismiss(audio_path.resolve())
        await pilot.pause()

        assert query_bar.attached_file == audio_path.resolve()
        assert query_bar.value == ""
        assert query_bar.input.suggester is None
        assert app.query_one("#target", Select).value == "audio"
        assert query_bar.input.placeholder == "Filter similar audio files…"

        await pilot.click("#clear-query-file")
        await pilot.pause()

        assert query_bar.attached_file is None
        assert query_bar.input.suggester is query_bar.path_suggester

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert isinstance(app.screen, FilePickerScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, FilePickerScreen)


@pytest.mark.anyio
async def test_file_picker_can_switch_between_filesystem_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    first_root.mkdir()
    second_root.mkdir()
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=first_root,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        picker = FilePickerScreen(
            first_root,
            {".png"},
            roots=(first_root, second_root),
        )
        await app.push_screen(picker)
        await pilot.pause()

        root_select = picker.query_one("#file-picker-root", Select)
        assert root_select.value == str(first_root.resolve())

        root_select.value = str(second_root.resolve())
        await pilot.pause()

        assert picker.current_directory == second_root.resolve()
        assert picker.query_one(SupportedFileTree).path == second_root.resolve()

        picker.dismiss(None)
        await pilot.pause()


@pytest.mark.anyio
async def test_tui_path_suggestions_follow_selected_target(
    tmp_path: Path,
) -> None:
    (tmp_path / "reference.png").touch()
    create_wave(tmp_path / "reference.wav")
    session = SearchService(
        RecordingSearchRepository(),  # type: ignore[arg-type]
        {
            "image": handler("image", RecordingImageEmbedding),
            "audio": handler("audio", RecordingAudioEmbedding),
        },
        ("image", "audio"),
        file_query_vectorizers={
            "image": ImageFileQueryVectorizer(),
            "audio": AudioFileQueryVectorizer(),
        },
    )
    app = FileLoreSearchApp(
        session,
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query_bar = app.query_one(QueryBar)
        query_bar.value = "ref"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert query_bar.input._suggestion == "reference.png"

        app.query_one("#target", Select).value = "audio"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert query_bar.input._suggestion == "reference.wav"


@pytest.mark.anyio
@pytest.mark.parametrize("completion_key", ("right", "tab"))
async def test_tui_accepts_filesystem_completion(
    tmp_path: Path,
    completion_key: str,
) -> None:
    (tmp_path / "reference.png").touch()
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "ref"
        await pilot.pause()

        await pilot.press(completion_key)
        await pilot.pause()

        assert query.value == "reference.png"


@pytest.mark.anyio
@pytest.mark.parametrize("completion_key", ("right", "tab"))
async def test_tui_quotes_completed_file_paths_containing_spaces(
    tmp_path: Path,
    completion_key: str,
) -> None:
    query_path = tmp_path / "reference image.png"
    query_path.touch()
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "reference i"
        await pilot.pause()

        await pilot.press(completion_key)
        await pilot.pause()

        assert query.value == '"reference image.png"'

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one(QueryBar).attached_file == query_path.resolve()


@pytest.mark.anyio
async def test_tui_continues_completion_inside_directory_with_spaces(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "sample recordings"
    directory.mkdir()
    query_path = directory / "reference image.png"
    query_path.touch()
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "sample r"
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()
        assert query.value == f'"sample recordings{os.sep}'

        await pilot.press("tab")
        await pilot.pause()
        assert query.value == (
            f'"sample recordings{os.sep}reference image.png"'
        )


@pytest.mark.anyio
async def test_tui_tab_moves_focus_when_no_completion_exists(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "a semantic query without a local match"
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()

        assert app.query_one("#browse-query-file", Button).has_focus


@pytest.mark.anyio
async def test_tui_long_attachment_label_fits_and_preserves_path_end(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=20,
        working_directory=tmp_path,
    )
    long_path = (
        tmp_path
        / "a-very-long-collection-name"
        / "another-long-directory-name"
        / "reference-image.png"
    )

    async with app.run_test(size=(76, 30)) as pilot:
        app.query_one(QueryBar).attach_file(long_path)
        await pilot.pause()

        label = app.query_one(TailPathLabel)
        rendered = label.render().plain
        assert rendered.startswith("📎 ...")
        assert rendered.endswith(".png")
        assert len(rendered) <= label.content_size.width


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


@pytest.mark.anyio
async def test_tui_searches_for_images_similar_to_a_file(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(query_path)
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "similar.png"),)
    )
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(
        image_session(repository, created),
        limit=10,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "reference.png format:png"

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(created) == 1
        assert created[0].texts == []
        assert created[0].images == [query_path.resolve()]
        assert app.query_one(QueryBar).attached_file == query_path.resolve()
        assert query.value == "format:png"
        assert not app.query_one("#query-file-label", Static).has_class(
            "hidden"
        )
        assert repository.calls[0]["scope"] == "file"
        assert repository.calls[0]["metadata_filter"] is not None
        assert "format:png" in str(
            app.query_one("#active-filters", Static).content
        )
        assert "Found 1 file" in str(app.query_one("#status", Static).content)


@pytest.mark.anyio
async def test_tui_keeps_result_cards_clear_of_the_vertical_scrollbar(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=tuple(
            image_result(tmp_path / f"image-{index}.png")
            for index in range(20)
        )
    )
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=20)

    async with app.run_test(size=(104, 32)) as pilot:
        app.query_one("#query", Input).value = "cat"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        results_scroll = app.query_one("#results-scroll")
        first_card = app.query(SearchResultCard).first()
        assert results_scroll.show_vertical_scrollbar
        assert (
            results_scroll.vertical_scrollbar.region.x
            - first_card.region.right
            >= 2
        )


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

    session = SearchService(
        repository,  # type: ignore[arg-type]
        {
            "image": handler("image", image_factory),
            "audio": handler("audio", audio_factory),
        },
        ("image", "audio"),
    )

    assert images == [] and audios == []
    orange = parse_search_query("orange cat")
    blue = parse_search_query("blue dog")
    session.search(
        SearchRequest(
            SearchSource.from_text(orange.semantic_query),
            "image",
            orange.metadata_query,
        ),
        10,
    )
    session.search(
        SearchRequest(
            SearchSource.from_text(blue.semantic_query),
            "image",
            blue.metadata_query,
        ),
        10,
    )
    assert len(images) == 1 and images[0].close_count == 0
    assert audios == []

    glass = parse_search_query("glass breaking")
    session.search(
        SearchRequest(
            SearchSource.from_text(glass.semantic_query),
            "audio",
            glass.metadata_query,
        ),
        10,
    )
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

    grouped = group_segment_results(results, limit=1)

    assert len(grouped) == 1
    assert grouped[0].result.file.id == "crash"
    assert [chunk.score for chunk in grouped[0].matches] == [0.9, 0.6]


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

    session = SearchService(
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
        assert repository.calls[0]["limit"] == (
            10 * SEGMENT_GROUP_OVERFETCH_FACTOR
        )
        assert repository.calls[0]["metadata_filter"] is not None
        assert len(app.query(SearchResultCard)) == 1
        assert len(app.query(Collapsible)) == 1
        chunk_list = app.query_one(Collapsible)
        chunk_list.collapsed = False
        await pilot.pause()
        chunk_matches = chunk_list.query_one(".chunk-matches", Static)
        assert chunk_list.region.right - chunk_matches.region.right >= 2
        status = str(app.query_one("#status", Static).content)
        assert "Found 1 file" in status
        assert "grouped 2 audio chunks into 1 file" in status


@pytest.mark.anyio
async def test_tui_searches_for_audio_similar_to_a_file(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.wav"
    create_wave(query_path)
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
                segment_index=1,
                score=0.9,
            ),
        )
    )
    audios: list[RecordingAudioEmbedding] = []
    images: list[RecordingImageEmbedding] = []

    def image_factory() -> RecordingImageEmbedding:
        images.append(RecordingImageEmbedding())
        return images[-1]

    def factory() -> RecordingAudioEmbedding:
        audios.append(RecordingAudioEmbedding())
        return audios[-1]

    session = SearchService(
        repository,  # type: ignore[arg-type]
        {
            "image": handler("image", image_factory),
            "audio": handler("audio", factory),
        },
        ("image", "audio"),
        file_query_vectorizers={"audio": AudioFileQueryVectorizer()},
    )
    app = FileLoreSearchApp(
        session,
        limit=10,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(110, 38)) as pilot:
        app.query_one("#query", Input).value = "reference.wav format:wav"

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(audios) == 1
        assert images == []
        assert audios[0].texts == []
        assert len(audios[0].audio_inputs) == 1
        assert app.query_one(QueryBar).attached_file == query_path.resolve()
        assert app.query_one("#target", Select).value == "audio"
        assert repository.calls[0]["scope"] == "segment"
        assert repository.calls[0]["limit"] == (
            10 * SEGMENT_GROUP_OVERFETCH_FACTOR
        )
        assert len(app.query(SearchResultCard)) == 1
        assert "grouped 2 audio chunks into 1 file" in str(
            app.query_one("#status", Static).content
        )


@pytest.mark.anyio
async def test_tui_help_tracks_selected_target(tmp_path: Path) -> None:
    repository = RecordingSearchRepository()
    session = SearchService(
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
        query = app.query_one("#query", Input)
        assert query.placeholder == "Describe image or enter a file path…"
        assert not target.disabled
        target.value = "audio"
        await pilot.pause()
        assert query.placeholder == "Describe audio or enter a file path…"
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
async def test_tui_clears_incompatible_file_when_target_changes(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8)).save(query_path)
    repository = RecordingSearchRepository()
    session = SearchService(
        repository,  # type: ignore[arg-type]
        {
            "image": handler("image", RecordingImageEmbedding),
            "audio": handler("audio", RecordingAudioEmbedding),
        },
        ("image", "audio"),
    )
    app = FileLoreSearchApp(
        session,
        limit=20,
        working_directory=tmp_path,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        query_bar = app.query_one(QueryBar)
        query_bar.attach_file(query_path)
        query_bar.value = "format:png"

        app.query_one("#target", Select).value = "audio"
        await pilot.pause()

        assert query_bar.attached_file is None
        assert query_bar.value == "format:png"
        assert "Reference file cleared" in str(
            app.query_one("#status", Static).content
        )
        assert query_bar.input.placeholder == (
            "Describe audio or enter a file path…"
        )


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
async def test_tui_recovers_controls_after_background_search_failure(
    tmp_path: Path,
) -> None:
    repository = RecordingSearchRepository(
        file_results=(image_result(tmp_path / "cat.png"),)
    )
    original_search = repository.semantic_search
    attempts = 0

    def fail_once(
        vector: Sequence[float],
        **kwargs: Any,
    ) -> tuple[FileSearchResult, ...]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("database unavailable")
        return original_search(vector, **kwargs)

    repository.semantic_search = fail_once  # type: ignore[method-assign]
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=10)

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "cat"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = app.query_one("#status", Static)
        assert "Search failed: database unavailable" in str(status.content)
        assert status.has_class("error")
        assert not query.disabled
        assert not app.query_one("#clear-query-file", Button).disabled
        assert not app.query_one("#limit", Select).disabled
        assert app.query_one("#target", Select).disabled
        assert query.has_focus

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert attempts == 2
        assert "Found 1 file" in str(status.content)
        assert not status.has_class("error")
        assert len(app.query(SearchResultCard)) == 1


@pytest.mark.anyio
async def test_tui_renders_empty_search_results_and_restores_controls() -> None:
    repository = RecordingSearchRepository()
    created: list[RecordingImageEmbedding] = []
    app = FileLoreSearchApp(image_session(repository, created), limit=10)

    async with app.run_test(size=(100, 32)) as pilot:
        query = app.query_one("#query", Input)
        query.value = "missing subject"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        results = app.query_one("#results", Vertical)
        assert "No semantic matches found." in str(
            results.query_one(Static).content
        )
        assert len(app.query(SearchResultCard)) == 0
        assert "Found 0 files" in str(
            app.query_one("#status", Static).content
        )
        assert not query.disabled
        assert query.has_focus


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
