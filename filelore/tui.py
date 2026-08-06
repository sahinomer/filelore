"""Textual full-screen search interface for a persistent FileLore session."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Footer, Header, Input, Select, Static
from textual.worker import get_current_worker

from filelore.cli_display import search_result_item_renderable
from filelore.embedding import TextEmbedding
from filelore.index import (
    FileIndexRepository,
    FileSearchResult,
    IndexHandler,
    file_metadata_filter,
)
from filelore.search_query import (
    ParsedSearchQuery,
    parse_search_query,
    validate_search_target,
)


AUDIO_OVERFETCH_FACTOR = 5
StageCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SearchResultEntity:
    """One visible file result with optional matching audio chunks."""

    result: FileSearchResult
    chunks: tuple[FileSearchResult, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchResponse:
    parsed_query: ParsedSearchQuery
    target: str
    results: tuple[SearchResultEntity, ...]
    limit: int
    grouped_chunk_count: int
    grouped_file_count: int
    embedding_ms: float
    fetch_ms: float
    total_ms: float


class SearchSession:
    """Lazily retain one target model across interactive searches."""

    def __init__(
        self,
        file_index: FileIndexRepository,
        handlers: Mapping[str, IndexHandler],
        allowed_targets: Sequence[str],
    ) -> None:
        selected_targets = tuple(dict.fromkeys(allowed_targets))
        if not selected_targets:
            raise ValueError("Interactive search requires at least one target")
        unknown = set(selected_targets).difference(handlers)
        if unknown:
            raise ValueError(f"Unsupported interactive target: {sorted(unknown)[0]}")
        self.file_index = file_index
        self.handlers = {
            target: handlers[target] for target in selected_targets
        }
        self.targets = selected_targets
        self.default_target = (
            "image" if "image" in self.handlers else selected_targets[0]
        )
        self._active_target: str | None = None
        self._embedding: TextEmbedding[Any] | None = None
        self._lock = RLock()

    @property
    def active_target(self) -> str | None:
        return self._active_target

    def search(
        self,
        parsed_query: ParsedSearchQuery,
        target: str,
        limit: int,
        *,
        on_stage: StageCallback | None = None,
    ) -> SearchResponse:
        if target not in self.handlers:
            raise ValueError(f"Interactive target is not enabled: {target}")
        if limit < 1:
            raise ValueError("Search limit must be positive")
        validate_search_target(parsed_query, target)

        with self._lock:
            total_started = perf_counter()
            embedding = self._activate(target, on_stage=on_stage)
            if on_stage is not None:
                on_stage(f"Searching {target} files…")
            embedding_started = perf_counter()
            query_vector = embedding.predict_text(
                parsed_query.semantic_query
            )
            embedding_ms = (perf_counter() - embedding_started) * 1000

            fetch_started = perf_counter()
            handler = self.handlers[target]
            search = {
                "file": self.file_index.semantic_search,
                "segment": self.file_index.semantic_segment_search,
            }[handler.vector_scope]
            fetch_limit = (
                limit * AUDIO_OVERFETCH_FACTOR
                if handler.vector_scope == "segment"
                else limit
            )
            raw_results = search(
                query_vector,
                vector_name=embedding.vector_name,
                limit=fetch_limit,
                metadata_filter=file_metadata_filter(
                    parsed_query.metadata_query
                ),
            )
            fetch_ms = (perf_counter() - fetch_started) * 1000

            if handler.vector_scope == "segment":
                results = group_audio_results(raw_results, limit=limit)
            else:
                results = tuple(
                    SearchResultEntity(result) for result in raw_results[:limit]
                )
            grouped_results = tuple(item for item in results if item.chunks)
            return SearchResponse(
                parsed_query=parsed_query,
                target=target,
                results=results,
                limit=limit,
                grouped_chunk_count=sum(
                    len(item.chunks) for item in grouped_results
                ),
                grouped_file_count=len(grouped_results),
                embedding_ms=embedding_ms,
                fetch_ms=fetch_ms,
                total_ms=(perf_counter() - total_started) * 1000,
            )

    def close(self) -> None:
        """Release the currently active model, if any."""
        with self._lock:
            if self._embedding is not None:
                self._embedding.close()
            self._embedding = None
            self._active_target = None

    def _activate(
        self,
        target: str,
        *,
        on_stage: StageCallback | None,
    ) -> TextEmbedding[Any]:
        if self._embedding is not None and self._active_target == target:
            return self._embedding
        if self._embedding is not None:
            self._embedding.close()
            self._embedding = None
            self._active_target = None
        if on_stage is not None:
            on_stage(f"Loading {target} model…")
        embedding = self.handlers[target].embedding_factory()
        if not isinstance(embedding, TextEmbedding):
            embedding.close()
            raise TypeError(
                f"Interactive {target} search requires a text embedding"
            )
        self._embedding = embedding
        self._active_target = target
        return embedding


def group_audio_results(
    results: Sequence[FileSearchResult],
    *,
    limit: int,
) -> tuple[SearchResultEntity, ...]:
    """Group raw segment matches by parent file and keep best-first chunks."""
    groups: dict[str, list[FileSearchResult]] = {}
    for result in results:
        groups.setdefault(result.file.id, []).append(result)

    entities: list[SearchResultEntity] = []
    for matches in groups.values():
        ranked = tuple(sorted(matches, key=lambda item: item.score, reverse=True))
        entities.append(SearchResultEntity(result=ranked[0], chunks=ranked))
    entities.sort(key=lambda item: item.result.score, reverse=True)
    return tuple(entities[:limit])


class QueryHelpScreen(ModalScreen[None]):
    """Compact reference for interactive query syntax."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("f1", "dismiss", "Close"),
    ]
    CSS = """
    QueryHelpScreen {
        align: center middle;
        background: $background 70%;
    }

    #help-dialog {
        width: 90%;
        max-width: 76;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }

    #help-title {
        height: auto;
        margin-bottom: 1;
        color: $accent;
        text-style: bold;
        content-align: center middle;
    }

    #help-content {
        height: auto;
    }

    #help-close {
        height: auto;
        margin-top: 1;
        color: $text-muted;
        content-align: center middle;
    }
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("Query help", id="help-title")
            yield Static(
                _query_help_text(self.target),
                id="help-content",
            )
            yield Static("Press F1 or Esc to close", id="help-close")


class SearchResultCard(Vertical):
    """One visible file result with optional expandable audio chunks."""

    def __init__(
        self,
        entity: SearchResultEntity,
        *,
        rank: int,
    ) -> None:
        super().__init__(classes="result-card")
        self.entity = entity
        self.rank = rank

    def compose(self) -> ComposeResult:
        yield Static(
            search_result_item_renderable(
                self.entity.result,
                rank=self.rank,
                score=self.entity.result.score,
            ),
            classes="result-summary",
        )
        if self.entity.chunks:
            yield Collapsible(
                Static(_chunk_matches_renderable(self.entity.chunks)),
                title=(
                    f"{len(self.entity.chunks)} matching chunks "
                    "(best first)"
                ),
                collapsed=True,
                classes="chunk-list",
            )


class FileLoreSearchApp(App[None]):
    """Keyboard-first Textual interface that searches only on Enter."""

    TITLE = "FileLore"
    SUB_TITLE = "Interactive semantic file search"
    LIMIT_PRESETS = (5, 10, 20, 50, 100)
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_search", "Clear"),
        ("f1", "query_help", "Help"),
        ("escape", "focus_search", "Search"),
    ]
    CSS = """
    Screen {
        background: $surface;
    }

    #body {
        height: 1fr;
        padding: 1 2;
    }

    #search-row {
        height: 3;
        margin-bottom: 1;
    }

    #target {
        width: 16;
        height: 3;
        margin-right: 1;
    }

    #query {
        width: 1fr;
        height: 3;
        border: tall $accent;
    }

    #limit-label {
        width: 7;
        height: 3;
        margin-left: 1;
        content-align: right middle;
        color: $text-muted;
    }

    #limit {
        width: 14;
        height: 3;
        margin-left: 1;
    }

    #active-filters {
        color: $text-muted;
        height: auto;
    }

    #active-filters {
        color: $accent;
    }

    #status {
        height: auto;
        color: $success;
    }

    #status.searching {
        color: $warning;
    }

    #status.error {
        color: $error;
    }

    #results-scroll {
        height: 1fr;
        border: round $primary;
        padding: 1;
        background: $panel;
    }

    #results {
        height: auto;
    }

    .result-card {
        height: auto;
        padding-bottom: 1;
        margin-bottom: 1;
        border-bottom: solid $primary 35%;
    }

    .result-summary {
        height: auto;
    }

    .chunk-list {
        height: auto;
        margin-left: 4;
        margin-top: 1;
    }
    """

    def __init__(self, session: SearchSession, *, limit: int) -> None:
        super().__init__()
        self.session = session
        self.initial_limit = limit
        self._searching = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            with Horizontal(id="search-row"):
                yield Select[str](
                    self._target_options(),
                    value=self.session.default_target,
                    allow_blank=False,
                    id="target",
                )
                yield Input(
                    placeholder="Describe a file, then press Enter to search…",
                    id="query",
                )
                yield Static("Limit", id="limit-label")
                yield Select[int](
                    self._limit_options(),
                    value=self.initial_limit,
                    allow_blank=False,
                    id="limit",
                )
            yield Static("", id="active-filters")
            yield Static("", id="status")
            with VerticalScroll(id="results-scroll"):
                with Vertical(id="results"):
                    yield Static("Search results will appear here.")
        yield Footer()

    def on_mount(self) -> None:
        target_select = self.query_one("#target", Select)
        target_select.disabled = len(self.session.targets) == 1
        self._update_query_placeholder(self.session.default_target)
        self.query_one("#query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._searching:
            return
        try:
            parsed_query = parse_search_query(event.value)
            target = self._selected_target()
            validate_search_target(parsed_query, target)
        except ValueError as error:
            self._show_error(str(error))
            return

        self._show_filters(parsed_query)
        self._searching = True
        event.input.disabled = True
        limit_select = self.query_one("#limit", Select)
        limit_select.disabled = True
        self.query_one("#target", Select).disabled = True
        selected_limit = limit_select.value
        if not isinstance(selected_limit, int):
            selected_limit = self.initial_limit
        self._set_status("Searching…", "searching")
        self.execute_search(parsed_query, target, selected_limit)

    @work(exclusive=True, thread=True, group="search", exit_on_error=False)
    def execute_search(
        self,
        parsed_query: ParsedSearchQuery,
        target: str,
        limit: int,
    ) -> None:
        worker = get_current_worker()

        def update_stage(message: str) -> None:
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._set_status,
                    message,
                    "searching",
                )

        try:
            response = self.session.search(
                parsed_query,
                target,
                limit,
                on_stage=update_stage,
            )
        except Exception as error:
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._show_search_error,
                    f"Search failed: {error}",
                )
            return
        if not worker.is_cancelled:
            self.call_from_thread(self._show_results, response)

    async def _show_results(self, response: SearchResponse) -> None:
        results = self.query_one("#results", Vertical)
        await results.remove_children()
        if response.results:
            await results.mount(
                *(
                    SearchResultCard(
                        entity,
                        rank=rank,
                    )
                    for rank, entity in enumerate(response.results, start=1)
                )
            )
        else:
            await results.mount(Static("No semantic matches found."))

        file_label = "file" if len(response.results) == 1 else "files"
        status = f"Found {len(response.results)} {file_label}"
        if response.grouped_chunk_count:
            chunk_label = (
                "chunk" if response.grouped_chunk_count == 1 else "chunks"
            )
            grouped_file_label = (
                "file" if response.grouped_file_count == 1 else "files"
            )
            status += (
                f"  •  grouped {response.grouped_chunk_count} audio "
                f"{chunk_label} "
                f"into {response.grouped_file_count} {grouped_file_label}"
            )
        status += f"  •  {_format_duration(response.total_ms)}"
        self._set_status(status)
        self._finish_search()

    def _show_filters(self, parsed_query: ParsedSearchQuery) -> None:
        filters = self.query_one("#active-filters", Static)
        if parsed_query.filters:
            values = "  •  ".join(
                f"{key}:{value}" for key, value in parsed_query.filters
            )
            filters.update(f"Active filters  {values}")
        else:
            filters.update("")

    def _show_error(self, message: str) -> None:
        self._set_status(message, "error")

    def _show_search_error(self, message: str) -> None:
        self._show_error(message)
        self._finish_search()

    def _finish_search(self) -> None:
        self._searching = False
        query = self.query_one("#query", Input)
        query.disabled = False
        self.query_one("#limit", Select).disabled = False
        self.query_one("#target", Select).disabled = (
            len(self.session.targets) == 1
        )
        query.focus()

    def _set_status(self, message: str, class_name: str | None = None) -> None:
        status = self.query_one("#status", Static)
        status.set_classes(class_name or "")
        status.update(message)

    async def action_clear_search(self) -> None:
        query = self.query_one("#query", Input)
        query.value = ""
        self.query_one("#active-filters", Static).update("")
        results = self.query_one("#results", Vertical)
        await results.remove_children()
        await results.mount(Static("Search results will appear here."))
        self._set_status("")
        query.focus()

    def action_focus_search(self) -> None:
        self.query_one("#query", Input).focus()

    def action_query_help(self) -> None:
        self.push_screen(QueryHelpScreen(self._selected_target()))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "target" and isinstance(event.value, str):
            self._update_query_placeholder(event.value)

    def _limit_options(self) -> tuple[tuple[str, int], ...]:
        limits = sorted({*self.LIMIT_PRESETS, self.initial_limit})
        return tuple((str(limit), limit) for limit in limits)

    def _target_options(self) -> tuple[tuple[str, str], ...]:
        labels = {"image": "Image", "audio": "Audio"}
        return tuple(
            (labels.get(target, target.title()), target)
            for target in self.session.targets
        )

    def _selected_target(self) -> str:
        selected = self.query_one("#target", Select).value
        return (
            selected
            if isinstance(selected, str)
            else self.session.default_target
        )

    def _update_query_placeholder(self, target: str) -> None:
        label = "an image" if target == "image" else "an audio file"
        self.query_one("#query", Input).placeholder = (
            f"Describe {label}, then press Enter to search…"
        )

    def on_unmount(self) -> None:
        self.session.close()


def run_interactive_search(
    file_index: FileIndexRepository,
    handlers: Mapping[str, IndexHandler],
    allowed_targets: Sequence[str],
    limit: int,
) -> int:
    """Run the full-screen search app until the user exits."""
    session = SearchSession(file_index, handlers, allowed_targets)
    try:
        FileLoreSearchApp(session, limit=limit).run()
    finally:
        session.close()
    return 0


def _query_help_text(target: str) -> str:
    shared = (
        "Write a natural description, then add optional key:value filters. "
        "Quote values that contain spaces.\n\n"
        "Shared filters\n"
        "name:holiday          File name contains text\n"
        "format:png            File format\n"
        "after:2025            Modified on or after a date\n"
        "before:2026           Modified before a date\n"
    )
    if target == "audio":
        specific = (
            "\nAudio filters\n"
            "sample-rate:48000  Exact sampling rate in Hz\n"
            "bitrate:192000     Exact bitrate in bits per second\n"
            "longer-than:1.5    Duration greater than seconds\n"
            "shorter-than:30    Duration less than seconds\n\n"
            "Example: glass breaking format:wav longer-than:1"
        )
    else:
        specific = (
            "\nImage filters\n"
            "min-res:1280x720   Minimum image resolution\n"
            "max-res:3840x2160  Maximum image resolution\n\n"
            "Example: cat on a balcony format:jpg after:2025"
        )
    dates = (
        "\n\nDates accept YYYY, YYYY-MM, YYYY-MM-DD, or an ISO datetime."
    )
    return shared + specific + dates


def _chunk_matches_renderable(
    chunks: Sequence[FileSearchResult],
) -> Table:
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column("Chunk", width=9, no_wrap=True)
    table.add_column("Time", ratio=1)
    table.add_column("Score", width=10, justify="right", no_wrap=True)
    for chunk in chunks:
        segment = chunk.segment
        if segment is None:
            continue
        table.add_row(
            f"#{segment.index + 1}",
            (
                f"{_timestamp_text(segment.start_seconds)} – "
                f"{_timestamp_text(segment.end_seconds)}"
            ),
            Text(f"{chunk.score:.3f}", style="cyan"),
        )
    return table


def _timestamp_text(seconds: float) -> str:
    minutes, remaining = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes}:{remaining:05.2f}"


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"
    return f"{duration_ms:.2f} ms"
