"""Modal browser for selecting a supported reference file."""

from __future__ import annotations

import os
import string
from collections.abc import Collection, Iterable
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Select, Static


def available_filesystem_roots() -> tuple[Path, ...]:
    """Return mounted drive roots on Windows or the POSIX filesystem root."""
    if os.name != "nt":
        return (Path("/"),)

    try:
        from ctypes import windll

        drive_mask = int(windll.kernel32.GetLogicalDrives())
    except (AttributeError, OSError):
        drive_mask = 0

    roots = tuple(
        Path(f"{letter}:\\")
        for index, letter in enumerate(string.ascii_uppercase)
        if drive_mask & (1 << index)
    )
    if roots:
        return roots
    return (Path(Path.cwd().anchor),)


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

    #file-picker-root {
        width: 12;
        height: 3;
        margin-right: 1;
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
        *,
        roots: Collection[Path] | None = None,
    ) -> None:
        super().__init__()
        self.current_directory = start_directory.resolve()
        self.supported_extensions = frozenset(supported_extensions)
        discovered_roots = tuple(
            root.resolve()
            for root in (
                available_filesystem_roots() if roots is None else roots
            )
        )
        current_root = Path(self.current_directory.anchor).resolve()
        if not any(
            self._is_below(self.current_directory, root)
            for root in discovered_roots
        ):
            discovered_roots += (current_root,)
        self.roots = tuple(dict.fromkeys(discovered_roots))

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            with Horizontal(id="file-picker-toolbar"):
                yield Select[str](
                    tuple(
                        (self._root_label(root), str(root))
                        for root in self.roots
                    ),
                    value=str(self._root_for(self.current_directory)),
                    allow_blank=False,
                    id="file-picker-root",
                )
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
                "Choose a root, then select a file · Alt+Up goes back",
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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "file-picker-root":
            return
        event.stop()
        if isinstance(event.value, str):
            self._open_directory(Path(event.value))

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

    def _root_for(self, path: Path) -> Path:
        matches = tuple(root for root in self.roots if self._is_below(path, root))
        return max(matches, key=lambda root: len(root.parts))

    @staticmethod
    def _is_below(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _root_label(root: Path) -> str:
        filesystem_root = Path(root.anchor)
        if root == filesystem_root:
            return root.drive or root.anchor
        return root.name or str(root)
