from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from filelore.documents import (
    ParagraphChunker,
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
    UnicodeSentenceSplitter,
)
from filelore.metadata import DocumentFormat, DocumentMetadata


def parsed_document(*blocks: TextBlock) -> ParsedDocument:
    return ParsedDocument(
        metadata=DocumentMetadata(
            path=Path("synthetic-document.md"),
            extension=".md",
            mime_type="text/markdown",
            size_bytes=100,
            modified_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            document_format=DocumentFormat.MARKDOWN,
            title="Sentetik belge",
            language="tr",
        ),
        blocks=blocks,
    )


def block(
    index: int,
    text: str,
    *,
    block_type: TextBlockType = TextBlockType.PARAGRAPH,
    page: int | None = None,
    slide: int | None = None,
    section: tuple[str, ...] = (),
    line_start: int | None = None,
    line_end: int | None = None,
) -> TextBlock:
    return TextBlock(
        index=index,
        block_type=block_type,
        text=text,
        location=SourceLocation(
            page_number=page,
            slide_number=slide,
            section_path=section,
            source_line_start=line_start,
            source_line_end=line_end,
        ),
    )


def test_paragraph_chunker_combines_blocks_and_preserves_source_range() -> None:
    section = ("Ana Başlık",)
    document = parsed_document(
        block(
            0,
            "Ana Başlık",
            block_type=TextBlockType.HEADING,
            section=section,
            line_start=1,
            line_end=1,
        ),
        block(
            1,
            "Birinci paragraf Türkçe içerik taşır.",
            section=section,
            line_start=3,
            line_end=3,
        ),
        block(
            2,
            "İkinci paragraf 日本語 ve العربية içerir.",
            section=section,
            line_start=5,
            line_end=5,
        ),
    )

    chunks = ParagraphChunker(max_characters=200).chunks(document)

    assert len(chunks) == 1
    assert chunks[0].text == (
        "Ana Başlık\n\n"
        "Birinci paragraf Türkçe içerik taşır.\n\n"
        "İkinci paragraf 日本語 ve العربية içerir."
    )
    assert chunks[0].embedding_text == chunks[0].text
    assert chunks[0].first_block_index == 0
    assert chunks[0].last_block_index == 2
    assert chunks[0].location == SourceLocation(
        section_path=section,
        source_line_start=1,
        source_line_end=5,
    )


def test_paragraph_chunker_never_crosses_page_slide_or_section_boundaries() -> None:
    document = parsed_document(
        block(0, "Birinci sayfa.", page=1, section=("Giriş",)),
        block(1, "İkinci sayfa.", page=2, section=("Giriş",)),
        block(2, "Yeni bölüm.", page=2, section=("Sonuç",)),
        block(3, "Birinci slayt.", slide=1, section=("Slaytlar",)),
        block(4, "İkinci slayt.", slide=2, section=("Slaytlar",)),
    )

    chunks = ParagraphChunker(max_characters=1_000).chunks(document)

    assert tuple(chunk.text for chunk in chunks) == (
        "Birinci sayfa.",
        "İkinci sayfa.",
        "Yeni bölüm.",
        "Birinci slayt.",
        "İkinci slayt.",
    )
    assert tuple(chunk.location.page_number for chunk in chunks) == (
        1,
        2,
        2,
        None,
        None,
    )
    assert tuple(chunk.location.slide_number for chunk in chunks) == (
        None,
        None,
        None,
        1,
        2,
    )
    assert chunks[2].location.section_path == ("Sonuç",)


def test_paragraph_chunker_splits_oversized_multilingual_prose_by_sentence() -> None:
    document = parsed_document(
        block(
            0,
            "İlk Türkçe cümle. 日本語です。 العربية؟ Son cümle.",
            section=("Diller",),
        )
    )

    chunks = ParagraphChunker(max_characters=20).chunks(document)

    assert tuple(chunk.text for chunk in chunks) == (
        "İlk Türkçe cümle.",
        "日本語です。 العربية؟",
        "Son cümle.",
    )
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert all(chunk.first_block_index == 0 for chunk in chunks)


def test_paragraph_chunker_uses_word_and_code_line_fallbacks() -> None:
    document = parsed_document(
        block(0, "birinci ikinci üçüncü", section=("Metin",)),
        block(
            1,
            "satır_1\n\nsatır_2",
            block_type=TextBlockType.CODE,
            section=("Kod",),
        ),
    )

    chunks = ParagraphChunker(max_characters=12).chunks(document)

    assert tuple(chunk.text for chunk in chunks) == (
        "birinci",
        "ikinci",
        "üçüncü",
        "satır_1",
        "satır_2",
    )
    assert chunks[3].location.section_path == ("Kod",)


def test_paragraph_chunker_preserves_code_line_breaks_within_a_chunk() -> None:
    document = parsed_document(
        block(
            0,
            "satır_1\n\nsatır_2",
            block_type=TextBlockType.CODE,
            section=("Kod",),
        )
    )

    chunk = ParagraphChunker(max_characters=50).chunks(document)[0]

    assert chunk.text == "satır_1\n\nsatır_2"


def test_paragraph_chunker_adds_missing_parent_heading_to_embedding_text() -> None:
    section = ("Üst Başlık", "Alt Başlık")
    document = parsed_document(
        block(
            0,
            "Alt Başlık",
            block_type=TextBlockType.HEADING,
            section=section,
        ),
        block(1, "Bölüm içeriği.", section=section),
    )

    chunk = ParagraphChunker(max_characters=100).chunks(document)[0]

    assert chunk.text == "Alt Başlık\n\nBölüm içeriği."
    assert chunk.embedding_text == (
        "Üst Başlık\n\nAlt Başlık\n\nBölüm içeriği."
    )


def test_paragraph_chunker_restores_heading_context_on_continuation() -> None:
    section = ("Konu",)
    document = parsed_document(
        block(
            0,
            "Konu",
            block_type=TextBlockType.HEADING,
            section=section,
        ),
        block(1, "Birinci cümle. İkinci cümle.", section=section),
    )

    chunks = ParagraphChunker(max_characters=20).chunks(document)

    assert tuple(chunk.text for chunk in chunks) == (
        "Konu\n\nBirinci cümle.",
        "İkinci cümle.",
    )
    assert chunks[1].embedding_text == "Konu\n\nİkinci cümle."


def test_paragraph_chunker_treats_a_slide_title_as_included_context() -> None:
    section = ("Sunum Başlığı",)
    document = parsed_document(
        block(
            0,
            "Sunum Başlığı",
            block_type=TextBlockType.SLIDE_TITLE,
            slide=1,
            section=section,
        ),
        block(1, "Slayt içeriği.", slide=1, section=section),
    )

    chunk = ParagraphChunker(max_characters=100).chunks(document)[0]

    assert chunk.text == "Sunum Başlığı\n\nSlayt içeriği."
    assert chunk.embedding_text == chunk.text


def test_paragraph_chunker_handles_empty_documents() -> None:
    assert ParagraphChunker().chunks(parsed_document()) == ()


def test_unicode_sentence_splitter_avoids_abbreviations_and_decimals() -> None:
    sentences = UnicodeSentenceSplitter().split(
        "Dr. Örnek 3.14 değerini yazdı. 日本語です。العربية؟ Son."
    )

    assert sentences == (
        "Dr. Örnek 3.14 değerini yazdı.",
        "日本語です。",
        "العربية؟",
        "Son.",
    )


def test_paragraph_chunker_splits_a_single_oversized_word_safely() -> None:
    document = parsed_document(block(0, "çğışüöabcdef"))

    chunks = ParagraphChunker(max_characters=5).chunks(document)

    assert tuple(chunk.text for chunk in chunks) == ("çğışü", "öabcd", "ef")
    assert all(len(chunk.text) <= 5 for chunk in chunks)


@pytest.mark.parametrize("limit", (0, -1))
def test_paragraph_chunker_rejects_non_positive_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        ParagraphChunker(max_characters=limit)
