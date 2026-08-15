from __future__ import annotations

from pathlib import Path

import pytest

from filelore.documents import MarkdownDocumentParser, TextBlockType
from filelore.metadata import DocumentFormat


def test_markdown_parser_preserves_structure_locations_and_unicode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "çok-dilli.md"
    path.write_text(
        "\ufeff# Türkçe Başlık\r\n"
        "\r\n"
        "Girişte **Türkçe**, 日本語 ve العربية bulunur.\r\n"
        "\r\n"
        "## Ayrıntılar\r\n"
        "\r\n"
        "- Bir öğe\r\n"
        "- İkinci [bağlantı](https://example.com)\r\n"
        "\r\n"
        "> Alıntı metni\r\n"
        "\r\n"
        "```python\r\n"
        "print(\"İstanbul\")\r\n"
        "```\r\n"
        "\r\n"
        "| Dil | Şehir |\r\n"
        "| --- | --- |\r\n"
        "| Türkçe | İstanbul |\r\n"
        "\r\n"
        "<div>HTML içeriği &amp; metin</div>\r\n"
        "<script>görünmeyen_kod()</script>\r\n",
        encoding="utf-8",
        newline="",
    )

    document = MarkdownDocumentParser().parse(path)

    assert document.metadata.document_format is DocumentFormat.MARKDOWN
    assert document.metadata.path == path.resolve()
    assert document.metadata.extension == ".md"
    assert document.metadata.mime_type == "text/markdown"
    assert document.metadata.title == "Türkçe Başlık"
    assert [block.block_type for block in document.blocks] == [
        TextBlockType.HEADING,
        TextBlockType.PARAGRAPH,
        TextBlockType.HEADING,
        TextBlockType.LIST_ITEM,
        TextBlockType.LIST_ITEM,
        TextBlockType.QUOTE,
        TextBlockType.CODE,
        TextBlockType.TABLE,
        TextBlockType.OTHER,
    ]
    assert [block.index for block in document.blocks] == list(
        range(len(document.blocks))
    )
    assert document.blocks[1].text == (
        "Girişte Türkçe, 日本語 ve العربية bulunur."
    )
    assert document.blocks[1].location.section_path == ("Türkçe Başlık",)
    assert document.blocks[1].location.source_line_start == 3
    assert document.blocks[3].location.section_path == (
        "Türkçe Başlık",
        "Ayrıntılar",
    )
    assert document.blocks[4].text == "İkinci bağlantı"
    assert document.blocks[6].text == 'print("İstanbul")'
    assert document.blocks[6].attributes == {"language": "python"}
    assert document.blocks[7].text == (
        "Dil | Şehir\nTürkçe | İstanbul"
    )
    assert document.blocks[8].text == "HTML içeriği & metin"
    assert all("görünmeyen" not in block.text for block in document.blocks)


def test_markdown_parser_tracks_heading_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "sections.md"
    path.write_text(
        "# Ana\n\n### Derin\n\nMetin\n\n## Yeni\n\nSon",
        encoding="utf-8",
    )

    document = MarkdownDocumentParser().parse(path)

    assert [block.location.section_path for block in document.blocks] == [
        ("Ana",),
        ("Ana", "Derin"),
        ("Ana", "Derin"),
        ("Ana", "Yeni"),
        ("Ana", "Yeni"),
    ]
    assert document.blocks[1].attributes == {"level": 3}
    assert document.blocks[3].attributes == {"level": 2}


def test_markdown_parser_allows_an_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("\n\x00\n", encoding="utf-8")

    document = MarkdownDocumentParser().parse(path)

    assert document.metadata.title is None
    assert document.blocks == ()


def test_markdown_parser_requires_utf8(tmp_path: Path) -> None:
    path = tmp_path / "legacy.md"
    path.write_bytes("Türkçe".encode("cp1254"))

    with pytest.raises(UnicodeDecodeError):
        MarkdownDocumentParser().parse(path)


def test_markdown_parser_rejects_unsupported_and_missing_files(
    tmp_path: Path,
) -> None:
    parser = MarkdownDocumentParser()

    with pytest.raises(ValueError, match="Unsupported document extension"):
        parser.parse(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.md")
