"""Unified semantic-text and reference-file query input."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Static

from filelore.ui.path_suggester import FileSystemPathSuggester


def shorten_path_from_start(value: str, width: int) -> str:
    """Fit a path by preserving its most useful trailing portion."""
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width <= 3:
        return "." * width

    available = width - 3
    suffix = ""
    for character in reversed(value):
        candidate = character + suffix
        if cell_len(candidate) > available:
            break
        suffix = candidate
    return f"...{suffix}"


class TailPathLabel(Static):
    """Attachment label that keeps the filename visible when space is tight."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self.path_text = ""

    def set_path(self, path: str) -> None:
        self.path_text = path
        self.refresh()

    def render(self) -> Text:
        available = max(0, self.content_size.width - 3)
        fitted = shorten_path_from_start(self.path_text, available)
        return Text(f"📎 {fitted}")


class QueryInput(Input):
    """Input that accepts a visible completion with Tab or Right."""

    BINDINGS = [
        *Input.BINDINGS,
        Binding(
            "tab",
            "complete_or_focus_next",
            "Complete path or focus next",
            show=False,
        ),
    ]

    def action_cursor_right(self, select: bool = False) -> None:
        """Accept path completions, quoting paths containing whitespace."""
        if not select and self.cursor_at_end and self._suggestion:
            completion = self._suggestion
            if (
                completion[0] not in {'"', "'"}
                and any(character.isspace() for character in completion)
            ):
                trailing_separator = completion.endswith(("/", "\\"))
                completion = f'"{completion}'
                if not trailing_separator:
                    completion += '"'
                self.value = completion
                self.cursor_position = len(self.value)
                return
        super().action_cursor_right(select)

    def action_complete_or_focus_next(self) -> None:
        if self.cursor_at_end and self._suggestion:
            self.action_cursor_right()
        else:
            self.app.action_focus_next()

    def refresh_completion(self) -> None:
        """Discard a stale completion and request one for the current value."""
        self._suggestion = ""
        suggester = self.suggester
        value = self.value
        if suggester is None or not value:
            return

        async def update_completion() -> None:
            suggestion = await suggester.get_suggestion(value)
            if self.suggester is suggester and self.value == value:
                self._suggestion = suggestion or ""

        self.run_worker(
            update_completion(),
            exclusive=True,
            group="target-path-completion",
        )


class QueryBar(Horizontal):
    """Display one query input with an optional removable file attachment."""

    class BrowseRequested(Message):
        """Request that the app show its file picker."""

    DEFAULT_CSS = """
    QueryBar {
        width: 1fr;
        height: 3;
        border: tall $accent;
    }

    QueryBar #query-file-label {
        width: 40%;
        max-width: 40%;
        height: 1;
        margin: 0 0 0 1;
        color: $accent;
    }

    QueryBar #clear-query-file {
        width: 3;
        min-width: 3;
        height: 1;
        border: none;
        padding: 0;
        margin: 0 1 0 0;
    }

    QueryBar #query {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0 1;
    }

    QueryBar #browse-query-file {
        width: 10;
        min-width: 10;
        height: 1;
        border: none;
        padding: 0 1;
        margin: 0;
    }

    QueryBar .hidden {
        display: none;
    }
    """

    def __init__(
        self,
        *,
        working_directory: Path,
        supported_extensions: Collection[str] = (),
    ) -> None:
        super().__init__(id="query-bar")
        self.working_directory = working_directory.resolve()
        self.path_suggester = FileSystemPathSuggester(
            self.working_directory,
            supported_extensions,
        )
        self.attached_file: Path | None = None
        self._target = "image"

    def compose(self) -> ComposeResult:
        yield TailPathLabel(id="query-file-label", classes="hidden")
        yield Button("×", id="clear-query-file", classes="hidden")
        yield QueryInput(id="query", suggester=self.path_suggester)
        yield Button("Browse", id="browse-query-file")

    @property
    def input(self) -> QueryInput:
        return self.query_one("#query", QueryInput)

    @property
    def value(self) -> str:
        return self.input.value

    @value.setter
    def value(self, value: str) -> None:
        self.input.value = value

    def attach_file(self, path: Path) -> None:
        prepared = path.resolve()
        self.attached_file = prepared
        try:
            display_path = prepared.relative_to(self.working_directory)
        except ValueError:
            display_path = prepared
        label = self.query_one("#query-file-label", TailPathLabel)
        label.set_path(str(display_path))
        label.remove_class("hidden")
        self.query_one("#clear-query-file", Button).remove_class("hidden")
        self.input.suggester = None

    def clear_file(self) -> None:
        self.attached_file = None
        label = self.query_one("#query-file-label", TailPathLabel)
        label.set_path("")
        label.add_class("hidden")
        self.query_one("#clear-query-file", Button).add_class("hidden")
        self.input.suggester = self.path_suggester
        self.update_placeholder(self._target)
        self.input.focus()

    def clear(self) -> None:
        self.value = ""
        self.clear_file()

    def update_placeholder(self, target: str) -> None:
        self._target = target
        label = "image" if target == "image" else "audio"
        self.input.placeholder = (
            f"Filter similar {label} files…"
            if self.attached_file is not None
            else f"Describe {label} or enter a file path…"
        )

    def update_supported_extensions(
        self,
        supported_extensions: Collection[str],
    ) -> None:
        """Restrict path completion to the currently selected target."""
        self.path_suggester.update_supported_extensions(supported_extensions)
        if self.attached_file is None:
            self.input.refresh_completion()

    def set_controls_disabled(self, disabled: bool) -> None:
        self.input.disabled = disabled
        self.query_one("#clear-query-file", Button).disabled = disabled
        self.query_one("#browse-query-file", Button).disabled = disabled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-query-file":
            event.stop()
            self.clear_file()
        elif event.button.id == "browse-query-file":
            event.stop()
            self.post_message(self.BrowseRequested())
