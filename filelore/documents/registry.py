"""Extension-based routing for format-specific document parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from filelore.documents.models import ParsedDocument
from filelore.documents.parsers import DocumentParser


class DocumentParserRegistry:
    """Route document paths to parsers while preventing ambiguous formats."""

    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        parser_by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for extension in parser.supported_extensions:
                if not extension.startswith(".") or extension != extension.casefold():
                    raise ValueError(
                        "Document parser extensions must be lowercase and start with a dot"
                    )
                if extension in parser_by_extension:
                    raise ValueError(
                        f"Multiple document parsers support extension: {extension}"
                    )
                parser_by_extension[extension] = parser
        self._parser_by_extension = parser_by_extension

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Return every extension recognized by registered parsers."""
        return frozenset(self._parser_by_extension)

    def parser_for(self, path: str | Path) -> DocumentParser:
        """Return the parser registered for a path's extension."""
        extension = Path(path).suffix.casefold()
        try:
            return self._parser_by_extension[extension]
        except KeyError as error:
            displayed_extension = Path(path).suffix or "<none>"
            raise ValueError(
                f"Unsupported document extension: {displayed_extension}"
            ) from error

    def parse(self, path: str | Path) -> ParsedDocument:
        """Parse a path using its registered format-specific parser."""
        return self.parser_for(path).parse(path)
