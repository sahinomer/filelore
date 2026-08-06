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
from rich.prompt import Confirm
from rich.rule import Rule
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from filelore.index import FileSearchResult, FileSegmentMatch, IndexWorkPlan


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
    def indexing(
        self,
        total: int,
        *,
        label: str = "Indexing files",
        label_width: int | None = None,
    ) -> Iterator[IndexingProgress]:
        """Display determinate per-file indexing progress when there is work."""
        if total == 0:
            yield IndexingProgress()
            return

        progress = Progress(
            TextColumn(
                "{task.description}",
                table_column=Column(width=label_width, no_wrap=True),
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            ProcessingRateColumn(),
            console=self._console,
        )
        task_id = progress.add_task(label, total=total)
        self._active_progress = progress
        try:
            with progress:
                yield IndexingProgress(progress, task_id)
        finally:
            self._active_progress = None

    def confirm(self, message: str, *, default: bool = True) -> bool:
        """Ask a terminal confirmation question on the diagnostic stream."""
        return Confirm.ask(message, console=self._console, default=default)

    def print_index_discovery(self, plan: IndexWorkPlan) -> None:
        """Display aligned new, changed, and unchanged discovery counts."""
        if not plan.queues:
            self._console.print("Discovery complete: no supported files found")
            return

        self._console.print("Discovery complete")
        labels = tuple(
            f"{queue.file_type.title()} files" for queue in plan.queues
        )
        label_width = max(len(label) for label in labels)
        for queue, label in zip(plan.queues, labels):
            details = (
                f"{queue.discovered_count} found  •  "
                f"{queue.new_count} new  •  "
                f"{queue.updated_count} changed  •  "
                f"{queue.unchanged_count} unchanged"
            )
            if queue.failures:
                details += f"  •  {len(queue.failures)} unreadable"
            self._console.print(f"  {label:<{label_width}}  {details}")

    def print_index_result(
        self,
        file_type: str,
        *,
        added: int,
        updated: int,
        failed: int,
    ) -> None:
        """Display successful and failed work for one file type."""
        self._console.print(
            f"{file_type.title()} files: {added} added  •  "
            f"{updated} updated  •  {failed} failed"
        )

    def print_skipped(self, file_type: str, count: int) -> None:
        """Report a declined work queue without treating it as an error."""
        file_label = "file" if count == 1 else "files"
        self._console.print(f"Skipped {count} {file_type} {file_label}")

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
            _add_search_result_rows(
                table,
                result,
                rank=rank,
                relative_score=relative_score,
            )
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


def search_result_item_renderable(
    result: FileSearchResult,
    *,
    rank: int,
    relative_score: float,
) -> Table:
    """Build one reusable file-result table for interactive result cards."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=4, justify="right", no_wrap=True)
    table.add_column(ratio=1, overflow="fold")
    table.add_column(width=16, justify="right", no_wrap=True)
    _add_search_result_rows(
        table,
        result,
        rank=rank,
        relative_score=relative_score,
    )
    return table


def _add_search_result_rows(
    table: Table,
    result: FileSearchResult,
    *,
    rank: int,
    relative_score: float,
) -> None:
    path = result.file.path
    table.add_row(
        Text(f"{rank}.", style="bold cyan"),
        _linked_filename(path),
        _match_text(relative_score),
    )
    table.add_row("", _directory_text(path.parent), "")
    table.add_row(
        "",
        _details_text(
            result.file.metadata,
            file_type=result.file.file_type,
        ),
        "",
    )
    if result.segment is not None:
        table.add_row("", _segment_text(result.segment), "")
    table.add_row("", _modified_text(result.file.metadata), "")


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


def _details_text(metadata: dict[str, Any], *, file_type: str) -> Text:
    details: list[str] = []
    media_format = metadata.get(f"{file_type}_format") or metadata.get(
        "extension"
    )
    if media_format:
        details.append(str(media_format).lstrip(".").upper())

    width = metadata.get("width")
    height = metadata.get("height")
    if isinstance(width, int) and isinstance(height, int):
        details.append(f"{width:,} × {height:,} px")

    color_mode = metadata.get("color_mode")
    if color_mode:
        details.append(str(color_mode))

    duration = metadata.get("duration_seconds")
    if isinstance(duration, (int, float)):
        details.append(_duration_text(float(duration)))

    sample_rate = metadata.get("sample_rate_hz")
    if isinstance(sample_rate, int):
        details.append(f"{sample_rate / 1000:g} kHz")

    bitrate = metadata.get("bitrate_bps")
    if isinstance(bitrate, int):
        details.append(f"{bitrate / 1000:g} kbps")

    channels = metadata.get("channels")
    if isinstance(channels, int):
        channel_label = "channel" if channels == 1 else "channels"
        details.append(f"{channels} {channel_label}")

    codec = metadata.get("codec")
    if codec:
        details.append(str(codec))

    size_bytes = metadata.get("size_bytes")
    if isinstance(size_bytes, int):
        details.append(decimal(size_bytes))

    frame_count = metadata.get("frame_count")
    if isinstance(frame_count, int) and frame_count > 1:
        details.append(f"{frame_count:,} frames")

    type_label = file_type.strip().title() or "File"
    text = Text(f"{type_label:<11}", style="dim")
    text.append("  •  ".join(details) if details else "Details unavailable")
    return text


def _segment_text(segment: FileSegmentMatch) -> Text:
    text = Text("Chunk      ", style="dim")
    text.append(
        f"#{segment.index + 1}  "
        f"{_timestamp_text(segment.start_seconds)} – "
        f"{_timestamp_text(segment.end_seconds)}"
    )
    return text


def _duration_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    return _timestamp_text(seconds)


def _timestamp_text(seconds: float) -> str:
    minutes, remaining = divmod(max(seconds, 0.0), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes}:{remaining:05.2f}"


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
