from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from filelore.documents import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
    TextChunk,
)
from filelore.metadata import DocumentFormat, DocumentMetadata


def document_metadata(path: Path) -> DocumentMetadata:
    return DocumentMetadata(
        path=path,
        extension=".md",
        mime_type="text/markdown",
        size_bytes=42,
        modified_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        document_format=DocumentFormat.MARKDOWN,
        title="Türkçe belge",
        authors=("Şahin",),
        language="tr",
        properties={"description": "Çok dilli içerik 日本語 العربية"},
    )


def test_document_models_preserve_unicode_and_serialize_metadata(
    tmp_path: Path,
) -> None:
    metadata = document_metadata(tmp_path / "örnek.md")
    location = SourceLocation(
        section_path=("Giriş", "İstanbul"),
        source_line_start=3,
        source_line_end=4,
    )
    block = TextBlock(
        index=0,
        block_type=TextBlockType.PARAGRAPH,
        text="Türkçe, 日本語 ve العربية aynı belgede kalır. 👋",
        location=location,
    )
    document = ParsedDocument(metadata=metadata, blocks=(block,))
    chunk = TextChunk(
        index=0,
        text=block.text,
        embedding_text="Giriş\n\n" + block.text,
        first_block_index=0,
        last_block_index=0,
        location=location,
    )

    assert document.blocks[0].text == block.text
    assert chunk.location.section_path == ("Giriş", "İstanbul")
    serialized = metadata.to_dict()
    assert serialized["document_format"] == "markdown"
    assert serialized["title"] == "Türkçe belge"
    assert serialized["properties"]["description"].endswith("日本語 العربية")
    json.dumps(serialized, ensure_ascii=False)


@pytest.mark.parametrize(
    "location",
    (
        SourceLocation(page_number=1),
        SourceLocation(slide_number=2),
        SourceLocation(source_line_start=4, source_line_end=4),
    ),
)
def test_source_location_accepts_one_based_coordinates(
    location: SourceLocation,
) -> None:
    assert location.page_number or location.slide_number or location.source_line_start


@pytest.mark.parametrize(
    "arguments",
    (
        {"page_number": 0},
        {"slide_number": -1},
        {"source_line_start": 0},
        {"source_line_end": 1},
        {"source_line_start": 3, "source_line_end": 2},
        {"section_path": ("Valid", " ")},
    ),
)
def test_source_location_rejects_invalid_coordinates(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SourceLocation(**arguments)


def test_parsed_document_requires_contiguous_block_indices(
    tmp_path: Path,
) -> None:
    block = TextBlock(
        index=1,
        block_type=TextBlockType.PARAGRAPH,
        text="Atlanmış sıra",
    )

    with pytest.raises(ValueError, match="contiguous"):
        ParsedDocument(metadata=document_metadata(tmp_path / "doc.md"), blocks=(block,))


@pytest.mark.parametrize(
    "factory",
    (
        lambda: TextBlock(0, TextBlockType.PARAGRAPH, "  \n"),
        lambda: TextChunk(0, "text", " ", 0, 0),
        lambda: TextChunk(0, "text", "text", 2, 1),
    ),
)
def test_text_units_reject_empty_text_and_invalid_ranges(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    ("page_count", "slide_count"),
    ((0, None), (None, -1)),
)
def test_document_metadata_rejects_invalid_counts(
    tmp_path: Path,
    page_count: int | None,
    slide_count: int | None,
) -> None:
    with pytest.raises(ValueError):
        replace(
            document_metadata(tmp_path / "doc.md"),
            page_count=page_count,
            slide_count=slide_count,
        )
