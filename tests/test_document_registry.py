from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from filelore.documents import (
    DocumentParser,
    DocumentParserRegistry,
    ParsedDocument,
)
from filelore.metadata import DocumentFormat, DocumentMetadata


class StubPdfParser(DocumentParser):
    supported_extensions = frozenset({".pdf"})

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def parse(self, path: str | Path) -> ParsedDocument:
        prepared_path = self.prepare_path(path)
        self.paths.append(prepared_path)
        stat = prepared_path.stat()
        return ParsedDocument(
            metadata=DocumentMetadata(
                path=prepared_path,
                extension=".pdf",
                mime_type="application/pdf",
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
                document_format=DocumentFormat.PDF,
            ),
            blocks=(),
        )


def test_document_parser_registry_routes_case_insensitive_extensions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "BELGE.PDF"
    path.write_bytes(b"placeholder")
    parser = StubPdfParser()
    registry = DocumentParserRegistry((parser,))

    document = registry.parse(path)

    assert registry.supported_extensions == frozenset({".pdf"})
    assert document.metadata.path == path.resolve()
    assert parser.paths == [path.resolve()]


def test_document_parser_registry_rejects_unknown_extensions() -> None:
    registry = DocumentParserRegistry((StubPdfParser(),))

    with pytest.raises(ValueError, match="Unsupported document extension"):
        registry.parser_for("notes.txt")


def test_document_parser_registry_rejects_ambiguous_extensions() -> None:
    first = StubPdfParser()
    second = StubPdfParser()

    with pytest.raises(ValueError, match="Multiple document parsers"):
        DocumentParserRegistry((first, second))


def test_document_parser_validates_extension_and_file_existence(
    tmp_path: Path,
) -> None:
    parser = StubPdfParser()

    with pytest.raises(ValueError, match="Unsupported document extension"):
        parser.prepare_path(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        parser.prepare_path(tmp_path / "missing.pdf")
