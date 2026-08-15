from __future__ import annotations

from pathlib import Path

import pytest
import reportlab
from pypdf import PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from filelore.documents import PdfDocumentParser, TextBlockType
from filelore.metadata import DocumentFormat


_FONT_NAME = "FileLoreTestUnicode"


def create_text_pdf(path: Path) -> None:
    if _FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        font_path = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
        pdfmetrics.registerFont(TTFont(_FONT_NAME, font_path))

    canvas = Canvas(str(path), pagesize=A4)
    canvas.setTitle("Türkçe PDF Belgesi")
    canvas.setAuthor("Örnek Yazar")
    canvas.setSubject("Çok dilli metin çıkarma")
    canvas.setCreator("FileLore test suite")
    canvas.setKeywords("Türkçe, arama, İstanbul")

    canvas.bookmarkPage("introduction")
    canvas.addOutlineEntry("Giriş", "introduction", level=0)
    canvas.setFont(_FONT_NAME, 16)
    canvas.drawString(72, 780, "Türkçe PDF Belgesi")
    canvas.setFont(_FONT_NAME, 11)
    canvas.drawString(72, 720, "İstanbul, Türkçe karakterleri korur: ğüşiöç.")
    canvas.drawString(72, 650, "İkinci paragraf doğal bir sınır oluşturur.")
    canvas.showPage()

    canvas.bookmarkPage("details")
    canvas.addOutlineEntry("Ayrıntılar", "details", level=0)
    canvas.setFont(_FONT_NAME, 11)
    canvas.drawString(72, 760, "İkinci sayfadaki içerik ayrı konum taşır.")
    canvas.save()


def test_pdf_parser_extracts_pages_bookmarks_metadata_and_turkish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "belge.PDF"
    create_text_pdf(path)

    document = PdfDocumentParser().parse(path)

    assert document.metadata.document_format is DocumentFormat.PDF
    assert document.metadata.extension == ".pdf"
    assert document.metadata.mime_type == "application/pdf"
    assert document.metadata.title == "Türkçe PDF Belgesi"
    assert document.metadata.authors == ("Örnek Yazar",)
    assert document.metadata.page_count == 2
    assert document.metadata.properties["subject"] == "Çok dilli metin çıkarma"
    assert document.metadata.properties["creator"] == "FileLore test suite"
    assert document.metadata.properties["keywords"] == (
        "Türkçe",
        "arama",
        "İstanbul",
    )
    assert document.metadata.properties["pdf_version"]
    assert all(
        block.block_type is TextBlockType.PARAGRAPH
        for block in document.blocks
    )
    assert [block.index for block in document.blocks] == list(
        range(len(document.blocks))
    )
    assert [block.location.page_number for block in document.blocks] == [1, 1, 1, 2]
    assert [block.location.section_path for block in document.blocks] == [
        ("Giriş",),
        ("Giriş",),
        ("Giriş",),
        ("Ayrıntılar",),
    ]
    assert document.blocks[0].text == "Türkçe PDF Belgesi"
    assert "ğüşiöç" in document.blocks[1].text
    assert document.blocks[-1].text == (
        "İkinci sayfadaki içerik ayrı konum taşır."
    )


def test_pdf_parser_allows_pages_without_extractable_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)

    document = PdfDocumentParser().parse(path)

    assert document.metadata.page_count == 1
    assert document.blocks == ()


def test_pdf_parser_enforces_the_content_stream_limit(tmp_path: Path) -> None:
    path = tmp_path / "limited.pdf"
    create_text_pdf(path)

    with pytest.raises(ValueError, match="content stream exceeds"):
        PdfDocumentParser(max_content_stream_bytes=1).parse(path)


def test_pdf_parser_rejects_encrypted_documents(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pdf"
    encrypted_path = tmp_path / "encrypted.pdf"
    create_text_pdf(source_path)
    writer = PdfWriter(clone_from=source_path)
    writer.encrypt("secret")
    with encrypted_path.open("wb") as output:
        writer.write(output)

    with pytest.raises(ValueError, match="Encrypted PDF"):
        PdfDocumentParser().parse(encrypted_path)


def test_pdf_parser_rejects_corrupt_unsupported_and_missing_files(
    tmp_path: Path,
) -> None:
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_text("not a PDF", encoding="utf-8")
    parser = PdfDocumentParser()

    with pytest.raises(ValueError, match="Could not parse PDF"):
        parser.parse(corrupt_path)
    with pytest.raises(ValueError, match="Unsupported document extension"):
        parser.parse(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.pdf")


@pytest.mark.parametrize("limit", (0, -1))
def test_pdf_parser_rejects_invalid_content_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        PdfDocumentParser(max_content_stream_bytes=limit)
