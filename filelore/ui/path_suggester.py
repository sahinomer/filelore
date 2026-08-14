"""Filesystem-aware completions for reference-file queries."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Collection
from pathlib import Path

from textual.suggester import Suggester


class FileSystemPathSuggester(Suggester):
    """Suggest supported files and directories below a launch directory."""

    def __init__(
        self,
        working_directory: Path,
        supported_extensions: Collection[str],
    ) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self.working_directory = working_directory.resolve()
        self.supported_extensions = frozenset(
            extension.casefold()
            if extension.startswith(".")
            else f".{extension.casefold()}"
            for extension in supported_extensions
        )

    async def get_suggestion(self, value: str) -> str | None:
        """Return the first directory or supported-file completion."""
        if not value or value.endswith(('"', "'")):
            return None

        quote = value[0] if value[0] in {'"', "'"} else ""
        path_value = value[1:] if quote else value
        if not path_value:
            return None

        trailing_separator = path_value.endswith(("/", "\\"))
        expanded = Path(path_value).expanduser()
        resolved = (
            expanded
            if expanded.is_absolute()
            else self.working_directory / expanded
        )
        parent = resolved if trailing_separator else resolved.parent
        prefix = "" if trailing_separator else resolved.name
        if not parent.is_dir():
            return None

        candidates = await asyncio.to_thread(
            self._matching_entries,
            parent,
            prefix,
        )
        if not candidates:
            return None

        candidate = next(
            (
                entry
                for entry in candidates
                if quote or not any(character.isspace() for character in entry.name)
            ),
            None,
        )
        if candidate is None:
            return None
        typed_parent = (
            path_value
            if trailing_separator
            else path_value[: -len(prefix)]
        )
        completion = f"{typed_parent}{candidate.name}"
        if candidate.is_dir():
            completion += self._preferred_separator(path_value)
        elif quote:
            completion += quote
        return f"{quote}{completion}"

    def _matching_entries(self, parent: Path, prefix: str) -> list[Path]:
        try:
            entries = tuple(parent.iterdir())
        except OSError:
            return []

        matches = [
            entry
            for entry in entries
            if entry.name.casefold().startswith(prefix.casefold())
            and (
                entry.is_dir()
                or entry.suffix.casefold() in self.supported_extensions
            )
        ]
        return sorted(
            matches,
            key=lambda entry: (not entry.is_dir(), entry.name.casefold()),
        )

    @staticmethod
    def _preferred_separator(value: str) -> str:
        if "/" in value and "\\" not in value:
            return "/"
        if "\\" in value and "/" not in value:
            return "\\"
        return os.sep
