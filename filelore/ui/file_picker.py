"""Modal browser for selecting a supported reference file."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Static


class SupportedFileTree(DirectoryTree):
    """Directory tree that hides files unsupported by semantic search."""

    def __init__(
        self,
        path: Path,
        supported_extensions: Collection[str],
        **kwargs: object,
    ) -> None:
        super().__init__(path, **kwargs)
        self.supported_extensions = frozenset(
            extension.casefold()
            if extension.startswith(".")
            else f".{extension.casefold()}"
            for extension in supported_extensions
        )

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return (
            path
            for path in paths
            if self._safe_is_dir(path)
            or path.suffix.casefold() in self.supported_extensions
        )


class FilePickerScreen(ModalScreen[Path | None]):
    """Keyboard and mouse navigable supported-file picker."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("alt+up", "parent_directory", "Parent directory"),
    ]
    CSS = """
    FilePickerScreen {
        align: center middle;
        background: $background 70%;
    }

    #file-picker-dialog {
        width: 88%;
        height: 86%;
        border: round $accent;
        background: $panel;
        padding: 1;
    }

    #file-picker-toolbar {
        height: 3;
    }

    #file-picker-directory {
        width: 1fr;
        height: 3;
        padding: 1;
        color: $accent;
        text-overflow: ellipsis;
    }

    #file-picker-up, #file-picker-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        margin-left: 1;
    }

    #file-picker-tree {
        height: 1fr;
        border: solid $primary 35%;
    }

    #file-picker-hint {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        start_directory: Path,
        supported_extensions: Collection[str],
    ) -> None:
        super().__init__()
        self.current_directory = start_directory.resolve()
        self.supported_extensions = frozenset(supported_extensions)

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            with Horizontal(id="file-picker-toolbar"):
                yield Static(
                    str(self.current_directory),
                    id="file-picker-directory",
                )
                yield Button("Up", id="file-picker-up")
                yield Button("Cancel", id="file-picker-cancel")
            yield SupportedFileTree(
                self.current_directory,
                self.supported_extensions,
                id="file-picker-tree",
            )
            yield Static(
                "Select a file with Enter or the mouse · Alt+Up goes back",
                id="file-picker-hint",
            )

    def on_mount(self) -> None:
        self.query_one(SupportedFileTree).focus()

    def on_directory_tree_file_selected(
        self,
        event: DirectoryTree.FileSelected,
    ) -> None:
        event.stop()
        self.dismiss(event.path.resolve())

    def on_directory_tree_directory_selected(
        self,
        event: DirectoryTree.DirectorySelected,
    ) -> None:
        event.stop()
        self._open_directory(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-picker-up":
            event.stop()
            self.action_parent_directory()
        elif event.button.id == "file-picker-cancel":
            event.stop()
            self.action_cancel()

    def action_parent_directory(self) -> None:
        self._open_directory(self.current_directory.parent)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _open_directory(self, path: Path) -> None:
        prepared = path.resolve()
        if not prepared.is_dir():
            return
        self.current_directory = prepared
        self.query_one("#file-picker-directory", Static).update(str(prepared))
        self.query_one(SupportedFileTree).path = prepared
