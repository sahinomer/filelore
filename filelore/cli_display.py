"""Rich-backed presentation helpers for the non-interactive CLI."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

from rich.console import Console, Group, RenderableType
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from filelore.index import FileSearchResult


class ProcessingRateColumn(ProgressColumn):
    """Render a per-file processing rate for determinate indexing tasks."""

    def render(self, task: Task) -> Text:
        if task.speed is None:
            return Text("? files/s")
        return Text(f"{task.speed:.1f} files/s")


class IndexingProgress:
    """Small progress handle that also works when no task is displayed."""

    def __init__(
        self,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, completed: int) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, completed)


class CliDisplay:
    """Own terminal rendering without leaking Rich into application layers."""

    def __init__(
        self,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._stdout = stdout if stdout is not None else sys.stdout
        self._stderr = stderr if stderr is not None else sys.stderr
        self._output_console = Console(file=self._stdout, highlight=False)
        self._console = Console(file=self._stderr)
        self._active_progress: Progress | None = None
        self._suspension_depth = 0

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Display an indeterminate status for the duration of an operation."""
        if not self._console.is_terminal:
            self._console.print(message)
            yield
            return
        with self._console.status(message):
            yield

    @contextmanager
    def indexing(self, total: int) -> Iterator[IndexingProgress]:
        """Display determinate per-file indexing progress when there is work."""
        if total == 0:
            yield IndexingProgress()
            return

        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            ProcessingRateColumn(),
            console=self._console,
        )
        task_id = progress.add_task("Indexing images", total=total)
        self._active_progress = progress
        try:
            with progress:
                yield IndexingProgress(progress, task_id)
        finally:
            self._active_progress = None

    @contextmanager
    def suspend(self) -> Iterator[None]:
        """Temporarily clear a live bar while emitting ordinary output."""
        progress = self._active_progress
        should_resume = progress is not None and self._suspension_depth == 0
        self._suspension_depth += 1
        if should_resume:
            progress.stop()
        try:
            yield
        finally:
            self._suspension_depth -= 1
            if should_resume:
                progress.start()

    def print_error(self, message: str = "") -> None:
        """Write an unchanged line to stderr without colliding with a live bar."""
        with self.suspend():
            print(message, file=self._stderr)

    def print_search_results(
        self,
        results: Sequence[FileSearchResult],
        *,
        query: str,
        limit: int,
        timings: Sequence[tuple[str, str]],
    ) -> None:
        """Render readable, terminal-aware semantic search results."""
        self._output_console.print(
            search_results_renderable(
                results,
                query=query,
                limit=limit,
                timings=timings,
            )
        )


def search_results_renderable(
    results: Sequence[FileSearchResult],
    *,
    query: str,
    limit: int,
    timings: Sequence[tuple[str, str]],
    show_footer: bool = True,
) -> Group:
    """Build a Rich result group shared by CLI and full-screen search."""
    renderables: list[RenderableType] = []
    title = Text("Search results for ", style="bold")
    title.append(f'"{query}"', style="bold cyan")
    renderables.append(Rule(title, style="cyan"))

    if results:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=4, justify="right", no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        table.add_column(width=16, justify="right", no_wrap=True)

        relative_scores = _relative_scores(results)
        for rank, (result, relative_score) in enumerate(
            zip(results, relative_scores),
            start=1,
        ):
            path = result.file.path
            table.add_row(
                Text(f"{rank}.", style="bold cyan"),
                _linked_filename(path),
                _match_text(relative_score),
            )
            table.add_row("", _directory_text(path.parent), "")
            table.add_row("", _details_text(result.file.metadata), "")
            table.add_row("", _modified_text(result.file.metadata), "")
            if rank < len(results):
                table.add_row("", "", "")
        renderables.append(table)
    else:
        renderables.append(Text("No semantic matches found.", style="dim"))

    if show_footer:
        renderables.append(Rule(style="dim"))
        result_label = "result" if len(results) == 1 else "results"
        summary = Text()
        summary.append(f"{len(results)} {result_label}", style="bold")
        summary.append(f"  •  limit {limit}", style="dim")
        if results:
            summary.append("  •  relative match scale", style="dim")
        renderables.append(summary)

        if timings:
            timing_text = Text("Timing  ", style="bold dim")
            for index, (label, duration) in enumerate(timings):
                if index == 2:
                    timing_text.append("\n        ")
                elif index:
                    timing_text.append("  •  ", style="dim")
                timing_text.append(f"{label} ", style="dim")
                timing_text.append(duration)
            renderables.append(timing_text)
    return Group(*renderables)


def _relative_scores(results: Sequence[FileSearchResult]) -> tuple[float, ...]:
    scores = tuple(result.score for result in results)
    if len(scores) < 2:
        return tuple(1.0 for _ in scores)
    lowest = min(scores)
    highest = max(scores)
    span = highest - lowest
    if span <= 0:
        return tuple(1.0 for _ in scores)
    return tuple((score - lowest) / span for score in scores)


def _match_text(relative_score: float) -> Text:
    percentage = round(relative_score * 100)
    if relative_score >= 0.67:
        style = "bold green"
    elif relative_score >= 0.34:
        style = "bold yellow"
    else:
        style = "bold red"
    return Text(f"● {percentage:>3}% match", style=style)


def _linked_filename(path: Path) -> Text:
    return Text(
        path.name,
        style=Style(bold=True, color="bright_cyan", link=path.as_uri()),
        overflow="ellipsis",
    )


def _directory_text(directory: Path) -> Text:
    text = Text("Directory  ", style="dim")
    text.append(
        str(directory),
        style=Style(color="blue", underline=True, link=directory.as_uri()),
    )
    return text


def _details_text(metadata: dict[str, Any]) -> Text:
    details: list[str] = []
    image_format = metadata.get("image_format") or metadata.get("extension")
    if image_format:
        details.append(str(image_format).lstrip(".").upper())

    width = metadata.get("width")
    height = metadata.get("height")
    if isinstance(width, int) and isinstance(height, int):
        details.append(f"{width:,} × {height:,} px")

    color_mode = metadata.get("color_mode")
    if color_mode:
        details.append(str(color_mode))

    size_bytes = metadata.get("size_bytes")
    if isinstance(size_bytes, int):
        details.append(decimal(size_bytes))

    frame_count = metadata.get("frame_count")
    if isinstance(frame_count, int) and frame_count > 1:
        details.append(f"{frame_count:,} frames")

    text = Text("Image      ", style="dim")
    text.append("  •  ".join(details) if details else "Details unavailable")
    return text


def _modified_text(metadata: dict[str, Any]) -> Text:
    text = Text("Modified   ", style="dim")
    text.append(_format_modified_at(metadata.get("modified_at")))
    return text


def _format_modified_at(value: Any) -> str:
    try:
        modified_at = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
    except (TypeError, ValueError):
        return "Unknown"
    if modified_at.tzinfo is not None:
        modified_at = modified_at.astimezone()
    month = modified_at.strftime("%b")
    return f"{month} {modified_at.day}, {modified_at.year} at {modified_at:%H:%M}"
