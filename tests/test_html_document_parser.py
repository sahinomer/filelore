from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from filelore.documents import HtmlDocumentParser, TextBlockType
from filelore.metadata import DocumentFormat


def test_html_parser_extracts_structure_metadata_and_unicode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "çok-dilli.HTML"
    path.write_text(
        """<!doctype html>
<html lang="tr-TR">
<head>
  <meta charset="utf-8">
  <title>Çok Dilli Belge</title>
  <meta name="author" content="Şahin">
  <meta property="article:author" content="Ayşe">
  <meta name="description" content="Türkçe açıklama">
  <meta name="keywords" content="arama, İstanbul, 日本語">
  <meta name="created" content="2026-08-14T10:30:00+03:00">
  <meta property="article:modified_time" content="2026-08-15T07:00:00Z">
  <script>görünmeyen_kod()</script>
</head>
<body>
  <h1>Ana Başlık</h1>
  <p>Türkçe, <strong>日本語</strong> ve العربية metin.</p>
  <h2>Ayrıntılar</h2>
  <ul>
    <li>İlk öğe<ul><li>İç öğe</li></ul></li>
    <li>İkinci öğe</li>
  </ul>
  <blockquote><p>Alıntı metni</p></blockquote>
  <pre><code class="language-python">print("İstanbul")
</code></pre>
  <table>
    <tr><th>Dil</th><th>Şehir</th></tr>
    <tr><td>Türkçe</td><td>İstanbul</td></tr>
  </table>
  <template>görünmeyen şablon</template>
</body>
</html>
""",
        encoding="utf-8",
    )

    document = HtmlDocumentParser().parse(path)

    assert document.metadata.document_format is DocumentFormat.HTML
    assert document.metadata.extension == ".html"
    assert document.metadata.mime_type == "text/html"
    assert document.metadata.title == "Çok Dilli Belge"
    assert document.metadata.authors == ("Şahin", "Ayşe")
    assert document.metadata.language == "tr-TR"
    assert document.metadata.created_at == datetime(
        2026, 8, 14, 10, 30, tzinfo=timezone(timedelta(hours=3))
    )
    assert document.metadata.content_modified_at == datetime(
        2026, 8, 15, 7, 0, tzinfo=timezone.utc
    )
    assert document.metadata.properties["description"] == "Türkçe açıklama"
    assert document.metadata.properties["keywords"] == (
        "arama",
        "İstanbul",
        "日本語",
    )
    assert [block.block_type for block in document.blocks] == [
        TextBlockType.HEADING,
        TextBlockType.PARAGRAPH,
        TextBlockType.HEADING,
        TextBlockType.LIST_ITEM,
        TextBlockType.LIST_ITEM,
        TextBlockType.LIST_ITEM,
        TextBlockType.QUOTE,
        TextBlockType.CODE,
        TextBlockType.TABLE,
    ]
    assert document.blocks[1].text == "Türkçe, 日本語 ve العربية metin."
    assert document.blocks[1].location.section_path == ("Ana Başlık",)
    assert document.blocks[1].location.source_line_start == 16
    assert [block.text for block in document.blocks[3:6]] == [
        "İlk öğe",
        "İç öğe",
        "İkinci öğe",
    ]
    assert document.blocks[6].text == "Alıntı metni"
    assert document.blocks[7].attributes == {"language": "python"}
    assert document.blocks[8].text == "Dil | Şehir\nTürkçe | İstanbul"
    assert all("görünmeyen" not in block.text for block in document.blocks)


def test_html_parser_uses_declared_legacy_encoding_and_title_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.htm"
    markup = (
        '<html lang="tr"><head><meta charset="windows-1254"></head>'
        "<body><h1>İstanbul ve Türkçe</h1><p>Şehir içeriği</p></body></html>"
    )
    path.write_bytes(markup.encode("cp1254"))

    document = HtmlDocumentParser().parse(path)

    assert document.metadata.title == "İstanbul ve Türkçe"
    assert document.metadata.language == "tr"
    assert document.metadata.properties["encoding"] == "windows-1254"
    assert document.blocks[1].text == "Şehir içeriği"


def test_html_parser_tolerates_malformed_markup(tmp_path: Path) -> None:
    path = tmp_path / "malformed.html"
    path.write_text(
        "<html><body><h1>Başlık<p>Kapanmamış metin<ul><li>Öğe",
        encoding="utf-8",
    )

    document = HtmlDocumentParser().parse(path)

    assert document.metadata.title == "Başlık Kapanmamış metin Öğe"
    assert any("Kapanmamış metin" in block.text for block in document.blocks)


def test_html_parser_preserves_bare_blockquotes_with_inline_code(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quote.html"
    path.write_text(
        "<html><body><blockquote>Ham <code>değer</code> korunur."
        "</blockquote></body></html>",
        encoding="utf-8",
    )

    document = HtmlDocumentParser().parse(path)

    assert [(block.block_type, block.text) for block in document.blocks] == [
        (TextBlockType.QUOTE, "Ham değer korunur."),
    ]


def test_html_parser_allows_a_document_without_visible_blocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.html"
    path.write_text(
        "<html><head><title>Metadata only</title></head>"
        "<body><script>x</script></body></html>",
        encoding="utf-8",
    )

    document = HtmlDocumentParser().parse(path)

    assert document.metadata.title == "Metadata only"
    assert document.blocks == ()


def test_html_parser_rejects_unsupported_and_missing_files(
    tmp_path: Path,
) -> None:
    parser = HtmlDocumentParser()

    with pytest.raises(ValueError, match="Unsupported document extension"):
        parser.parse(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.html")
