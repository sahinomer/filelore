"""Textual full-screen search interface for a persistent FileLore session."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Footer, Header, Input, Select, Static
from textual.worker import get_current_worker

from filelore.cli_display import search_result_item_renderable
from filelore.index import (
    FileIndexRepository,
    FileSearchResult,
)
from filelore.search import (
    FileQueryVectorizer,
    SearchRequest,
    SearchResponse,
    SearchResultGroup,
    SearchService,
    SearchTarget,
    build_interactive_search_request,
    target_for_format,
)
from filelore.ui import FilePickerScreen, QueryBar


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
        entity: SearchResultGroup,
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
        if self.entity.matches:
            yield Collapsible(
                Static(
                    _chunk_matches_renderable(self.entity.matches),
                    classes="chunk-matches",
                ),
                title=(
                    f"{len(self.entity.matches)} matching chunks "
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
        ("ctrl+o", "open_file_picker", "Open file"),
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

    #query-bar {
        width: 1fr;
        height: 3;
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
        scrollbar-gutter: stable;
    }

    #results {
        height: auto;
        margin-right: 2;
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
        padding-bottom: 1;
        padding-right: 2;
    }
    """

    def __init__(
        self,
        session: SearchService,
        *,
        limit: int,
        working_directory: Path | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.initial_limit = limit
        self.working_directory = (working_directory or Path.cwd()).resolve()
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
                yield QueryBar(
                    working_directory=self.working_directory,
                    supported_extensions=self._supported_file_extensions(),
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
        query_bar = self.query_one(QueryBar)
        query_bar.update_placeholder(self.session.default_target)
        query_bar.input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._searching:
            return
        query_bar = self.query_one(QueryBar)
        try:
            request = build_interactive_search_request(
                query_bar.value,
                target=self._selected_target(),
                file_query_vectorizers=self.session.file_query_vectorizers,
                query_file=query_bar.attached_file,
                base_directory=self.working_directory,
            )
        except ValueError as error:
            self._show_error(str(error))
            return

        if request.source.file is not None:
            query_bar.attach_file(request.source.file)
            query_bar.value = _format_filters(request.filters)
            target_select = self.query_one("#target", Select)
            if target_select.value != request.target:
                target_select.value = request.target
            query_bar.update_placeholder(request.target)
        self._show_filters(request.filters)
        self._searching = True
        query_bar.set_controls_disabled(True)
        limit_select = self.query_one("#limit", Select)
        limit_select.disabled = True
        self.query_one("#target", Select).disabled = True
        selected_limit = limit_select.value
        if not isinstance(selected_limit, int):
            selected_limit = self.initial_limit
        self._set_status("Searching…", "searching")
        self.execute_search(request, selected_limit)

    @work(exclusive=True, thread=True, group="search", exit_on_error=False)
    def execute_search(
        self,
        request: SearchRequest,
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
                request,
                limit,
                group_segments=True,
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
        if response.grouped_match_count:
            chunk_label = (
                "chunk" if response.grouped_match_count == 1 else "chunks"
            )
            grouped_file_label = (
                "file" if response.grouped_file_count == 1 else "files"
            )
            status += (
                f"  •  grouped {response.grouped_match_count} audio "
                f"{chunk_label} "
                f"into {response.grouped_file_count} {grouped_file_label}"
            )
        status += f"  •  {_format_duration(response.timings.total_ms)}"
        self._set_status(status)
        self._finish_search()

    def _show_filters(self, active: Sequence[tuple[str, str]]) -> None:
        filters = self.query_one("#active-filters", Static)
        if active:
            values = "  •  ".join(
                f"{key}:{value}" for key, value in active
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
        query_bar = self.query_one(QueryBar)
        query_bar.set_controls_disabled(False)
        self.query_one("#limit", Select).disabled = False
        self.query_one("#target", Select).disabled = (
            len(self.session.targets) == 1
        )
        query_bar.input.focus()

    def _set_status(self, message: str, class_name: str | None = None) -> None:
        status = self.query_one("#status", Static)
        status.set_classes(class_name or "")
        status.update(message)

    async def action_clear_search(self) -> None:
        query_bar = self.query_one(QueryBar)
        query_bar.clear()
        self.query_one("#active-filters", Static).update("")
        results = self.query_one("#results", Vertical)
        await results.remove_children()
        await results.mount(Static("Search results will appear here."))
        self._set_status("")
        query_bar.input.focus()

    def action_focus_search(self) -> None:
        self.query_one(QueryBar).input.focus()

    def action_query_help(self) -> None:
        self.push_screen(QueryHelpScreen(self._selected_target()))

    def action_open_file_picker(self) -> None:
        if self._searching:
            return
        self.push_screen(
            FilePickerScreen(
                self.working_directory,
                self._supported_file_extensions(),
            ),
            self._attach_query_file,
        )

    def on_query_bar_browse_requested(
        self,
        _: QueryBar.BrowseRequested,
    ) -> None:
        self.action_open_file_picker()

    def _attach_query_file(self, path: Path | None) -> None:
        if path is None:
            self.query_one(QueryBar).input.focus()
            return

        target = target_for_format(path.suffix)
        if target is None or target not in self.session.file_query_vectorizers:
            self._show_error(f"Unsupported query file format: {path.suffix}")
            return

        target_select = self.query_one("#target", Select)
        if target_select.value != target:
            target_select.value = target
        query_bar = self.query_one(QueryBar)
        query_bar.value = ""
        query_bar.attach_file(path)
        query_bar.update_placeholder(target)
        self._set_status("")
        query_bar.input.focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "target" and isinstance(event.value, str):
            query_bar = self.query_one(QueryBar)
            attached_file = query_bar.attached_file
            if (
                attached_file is not None
                and target_for_format(attached_file.suffix) != event.value
            ):
                query_bar.clear_file()
                self._set_status(
                    "Reference file cleared because the target changed"
                )
            query_bar.update_placeholder(event.value)

    def _limit_options(self) -> tuple[tuple[str, int], ...]:
        limits = sorted({*self.LIMIT_PRESETS, self.initial_limit})
        return tuple((str(limit), limit) for limit in limits)

    def _target_options(self) -> tuple[tuple[str, str], ...]:
        labels = {"image": "Image", "audio": "Audio"}
        return tuple(
            (labels.get(target, target.title()), target)
            for target in self.session.targets
        )

    def _supported_file_extensions(self) -> frozenset[str]:
        return frozenset(
            extension
            for vectorizer in self.session.file_query_vectorizers.values()
            for extension in vectorizer.supported_extensions
        )

    def _selected_target(self) -> str:
        selected = self.query_one("#target", Select).value
        return (
            selected
            if isinstance(selected, str)
            else self.session.default_target
        )

    def on_unmount(self) -> None:
        self.session.close()


def run_interactive_search(
    file_index: FileIndexRepository,
    search_targets: Mapping[str, SearchTarget],
    file_query_vectorizers: Mapping[str, FileQueryVectorizer],
    allowed_targets: Sequence[str],
    limit: int,
) -> int:
    """Run the full-screen search app until the user exits."""
    session = SearchService(
        file_index,
        search_targets,
        allowed_targets,
        file_query_vectorizers=file_query_vectorizers,
    )
    try:
        FileLoreSearchApp(session, limit=limit).run()
    finally:
        session.close()
    return 0


def _query_help_text(target: str) -> str:
    shared = (
        "Write a natural description or enter a reference file path. Add "
        "optional key:value filters in the same field. Relative paths start "
        "from the terminal working directory. Press Tab or Right to accept a "
        "path completion, or Ctrl+O to browse. Quote values that contain "
        "spaces.\n\n"
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
            "Text example: glass breaking format:wav longer-than:1\n"
            "File example: samples/crash.wav format:wav longer-than:1"
        )
    else:
        specific = (
            "\nImage filters\n"
            "min-res:1280x720   Minimum image resolution\n"
            "max-res:3840x2160  Maximum image resolution\n\n"
            "Text example: cat on a balcony format:jpg after:2025\n"
            "File example: samples/cat.jpg format:jpeg after:2025"
        )
    dates = (
        "\n\nDates accept YYYY, YYYY-MM, YYYY-MM-DD, or an ISO datetime."
    )
    return shared + specific + dates


def _format_filters(filters: Sequence[tuple[str, str]]) -> str:
    tokens: list[str] = []
    for key, value in filters:
        token = f"{key}:{value}"
        if any(character.isspace() for character in token):
            quote = '"' if '"' not in token else "'"
            token = f"{quote}{token}{quote}"
        tokens.append(token)
    return " ".join(tokens)


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
