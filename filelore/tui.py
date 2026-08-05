"""Textual full-screen search interface for a persistent FileLore session."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Select, Static
from textual.worker import get_current_worker

from filelore.cli_display import search_results_renderable
from filelore.embedding import ImageEmbedding
from filelore.index import (
    FileIndexRepository,
    FileSearchResult,
    file_metadata_filter,
)
from filelore.search_query import ParsedSearchQuery, parse_search_query


@dataclass(frozen=True, slots=True)
class SearchResponse:
    parsed_query: ParsedSearchQuery
    results: tuple[FileSearchResult, ...]
    limit: int
    embedding_ms: float
    fetch_ms: float
    total_ms: float


class SearchSession:
    """Reuse one initialized embedding model and repository across searches."""

    def __init__(
        self,
        file_index: FileIndexRepository,
        embedding: ImageEmbedding,
    ) -> None:
        self.file_index = file_index
        self.embedding = embedding

    def search(self, parsed_query: ParsedSearchQuery, limit: int) -> SearchResponse:
        total_started = perf_counter()
        embedding_started = perf_counter()
        query_vector = self.embedding.predict_text(parsed_query.semantic_query)
        embedding_ms = (perf_counter() - embedding_started) * 1000

        fetch_started = perf_counter()
        results = self.file_index.semantic_search(
            query_vector,
            vector_name=self.embedding.vector_name,
            limit=limit,
            metadata_filter=file_metadata_filter(parsed_query.metadata_query),
        )
        fetch_ms = (perf_counter() - fetch_started) * 1000
        return SearchResponse(
            parsed_query=parsed_query,
            results=results,
            limit=limit,
            embedding_ms=embedding_ms,
            fetch_ms=fetch_ms,
            total_ms=(perf_counter() - total_started) * 1000,
        )


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

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("Query help", id="help-title")
            yield Static(
                "Write a natural description, then add optional key:value "
                "filters. Quote values that contain spaces.\n\n"
                "name:holiday       File name contains text\n"
                "format:png         File format\n"
                "min-res:1280x720   Minimum image resolution\n"
                "max-res:3840x2160  Maximum image resolution\n"
                "after:2025         Modified on or after a date\n"
                "before:2026        Modified before a date\n\n"
                "Dates accept YYYY, YYYY-MM, YYYY-MM-DD, or an ISO "
                "datetime.\n"
                "Example: cat on a balcony format:jpg after:2025",
                id="help-content",
            )
            yield Static("Press F1 or Esc to close", id="help-close")


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
                yield Static(
                    "Search results will appear here.",
                    id="results",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._searching:
            return
        try:
            parsed_query = parse_search_query(event.value)
        except ValueError as error:
            self._show_error(str(error))
            return

        self._show_filters(parsed_query)
        self._searching = True
        event.input.disabled = True
        limit_select = self.query_one("#limit", Select)
        limit_select.disabled = True
        selected_limit = limit_select.value
        if not isinstance(selected_limit, int):
            selected_limit = self.initial_limit
        self._set_status("Searching…", "searching")
        self.execute_search(parsed_query, selected_limit)

    @work(exclusive=True, thread=True, group="search", exit_on_error=False)
    def execute_search(
        self,
        parsed_query: ParsedSearchQuery,
        limit: int,
    ) -> None:
        worker = get_current_worker()
        try:
            response = self.session.search(parsed_query, limit)
        except Exception as error:
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._show_search_error,
                    f"Search failed: {error}",
                )
            return
        if not worker.is_cancelled:
            self.call_from_thread(self._show_results, response)

    def _show_results(self, response: SearchResponse) -> None:
        self.query_one("#results", Static).update(
            search_results_renderable(
                response.results,
                query=response.parsed_query.semantic_query,
                limit=response.limit,
                timings=(),
                show_footer=False,
            )
        )
        result_label = "result" if len(response.results) == 1 else "results"
        self._set_status(
            f"Found {len(response.results)} {result_label} in "
            f"{_format_duration(response.total_ms)}"
        )
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
        query.focus()

    def _set_status(self, message: str, class_name: str | None = None) -> None:
        status = self.query_one("#status", Static)
        status.set_classes(class_name or "")
        status.update(message)

    def action_clear_search(self) -> None:
        query = self.query_one("#query", Input)
        query.value = ""
        self.query_one("#active-filters", Static).update("")
        self.query_one("#results", Static).update(
            "Search results will appear here."
        )
        self._set_status("")
        query.focus()

    def action_focus_search(self) -> None:
        self.query_one("#query", Input).focus()

    def action_query_help(self) -> None:
        self.push_screen(QueryHelpScreen())

    def _limit_options(self) -> tuple[tuple[str, int], ...]:
        limits = sorted({*self.LIMIT_PRESETS, self.initial_limit})
        return tuple((str(limit), limit) for limit in limits)


def run_interactive_search(
    file_index: FileIndexRepository,
    embedding: ImageEmbedding,
    limit: int,
) -> int:
    """Run the full-screen search app until the user exits."""
    FileLoreSearchApp(SearchSession(file_index, embedding), limit=limit).run()
    return 0


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"
    return f"{duration_ms:.2f} ms"
