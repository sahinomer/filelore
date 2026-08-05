"""Textual full-screen search interface for a persistent FileLore session."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static
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
            embedding_ms=embedding_ms,
            fetch_ms=fetch_ms,
            total_ms=(perf_counter() - total_started) * 1000,
        )


class FileLoreSearchApp(App[None]):
    """Keyboard-first Textual interface that searches only on Enter."""

    TITLE = "FileLore"
    SUB_TITLE = "Interactive semantic image search"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_search", "Clear"),
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

    #intro {
        color: $text-muted;
        height: auto;
        margin-bottom: 1;
    }

    #query {
        width: 100%;
        border: tall $accent;
        margin-bottom: 1;
    }

    #filter-help, #active-filters {
        color: $text-muted;
        height: auto;
    }

    #active-filters {
        color: $accent;
        margin-top: 1;
    }

    #status {
        height: auto;
        margin: 1 0;
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
        self.limit = limit
        self._searching = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Static(
                "The model is ready. Enter a semantic query and press Enter.",
                id="intro",
            )
            yield Input(
                placeholder=(
                    "cat format:jpg min-res:1280x720 after:2025 before:2026"
                ),
                id="query",
            )
            yield Static(
                "Filters: name · format · min-res · max-res · after · before",
                id="filter-help",
            )
            yield Static("", id="active-filters")
            yield Static("Ready", id="status")
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
        self._set_status("Searching…", "searching")
        self.execute_search(parsed_query)

    @work(exclusive=True, thread=True, group="search", exit_on_error=False)
    def execute_search(self, parsed_query: ParsedSearchQuery) -> None:
        worker = get_current_worker()
        try:
            response = self.session.search(parsed_query, self.limit)
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
                limit=self.limit,
                timings=(
                    ("embedding", _format_duration(response.embedding_ms)),
                    ("search", _format_duration(response.fetch_ms)),
                    ("total", _format_duration(response.total_ms)),
                ),
            )
        )
        result_label = "result" if len(response.results) == 1 else "results"
        self._set_status(f"Found {len(response.results)} {result_label}")
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
        self._set_status("Ready")
        query.focus()

    def action_focus_search(self) -> None:
        self.query_one("#query", Input).focus()


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
