"""Structure-aware HTML document parser backed by Beautiful Soup."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
)
from filelore.documents.parsers.base import DocumentParser
from filelore.metadata import DocumentFormat, DocumentMetadata


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = tuple(
    sorted(_HEADING_TAGS | {"blockquote", "code", "li", "p", "pre", "table"})
)
_REMOVED_TAGS = ("iframe", "noscript", "script", "style", "template")


class HtmlDocumentParser(DocumentParser):
    """Extract visible structural blocks and normalized HTML metadata."""

    supported_extensions = frozenset({".htm", ".html"})

    def parse(self, path: str | Path) -> ParsedDocument:
        document_path = self.prepare_path(path)
        soup = BeautifulSoup(document_path.read_bytes(), "html.parser")
        metadata_values = _metadata_values(soup)
        title = _document_title(soup, metadata_values)
        authors = _authors(metadata_values)
        created_at = _first_datetime(
            metadata_values,
            ("article:published_time", "created", "date", "dc.date", "dcterms.created"),
        )
        content_modified_at = _first_datetime(
            metadata_values,
            ("article:modified_time", "dcterms.modified", "modified"),
        )
        language = _language(soup, metadata_values)
        properties = _document_properties(soup, metadata_values)

        for tag in soup.find_all(_REMOVED_TAGS):
            tag.decompose()

        stat = document_path.stat()
        return ParsedDocument(
            metadata=DocumentMetadata(
                path=document_path,
                extension=document_path.suffix.casefold(),
                mime_type="text/html",
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
                document_format=DocumentFormat.HTML,
                title=title,
                authors=authors,
                created_at=created_at,
                content_modified_at=content_modified_at,
                language=language,
                properties=properties,
            ),
            blocks=_extract_blocks(soup),
        )


def _extract_blocks(soup: BeautifulSoup) -> tuple[TextBlock, ...]:
    root = soup.body or soup
    blocks: list[TextBlock] = []
    section_stack: list[tuple[int, str]] = []

    for tag in root.find_all(_BLOCK_TAGS):
        block_type: TextBlockType
        attributes: dict[str, object] = {}

        if tag.name in _HEADING_TAGS:
            if _inside(tag, {"li", "pre", "table"}):
                continue
            text = _text(tag)
            if not text:
                continue
            level = int(tag.name.removeprefix("h"))
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, text))
            block_type = TextBlockType.HEADING
            attributes["level"] = level
        elif tag.name == "p":
            if _inside(tag, {"li", "pre", "table"}):
                continue
            text = _text(tag)
            block_type = (
                TextBlockType.QUOTE
                if tag.find_parent("blockquote") is not None
                else TextBlockType.PARAGRAPH
            )
        elif tag.name == "li":
            if _inside(tag, {"pre", "table"}):
                continue
            text = _list_item_text(tag)
            block_type = TextBlockType.LIST_ITEM
            if tag.find_parent("blockquote") is not None:
                attributes["quoted"] = True
        elif tag.name == "blockquote":
            text = _blockquote_text(tag)
            block_type = TextBlockType.QUOTE
        elif tag.name == "pre":
            if _inside(tag, {"table"}):
                continue
            text = _code_text(tag)
            block_type = TextBlockType.CODE
            language = _code_language(tag)
            if language:
                attributes["language"] = language
        elif tag.name == "code":
            if _inside(
                tag,
                _HEADING_TAGS | {"blockquote", "li", "p", "pre", "table"},
            ):
                continue
            text = _code_text(tag)
            block_type = TextBlockType.CODE
            language = _code_language(tag)
            if language:
                attributes["language"] = language
        elif tag.name == "table":
            if _inside(tag, {"table"}):
                continue
            text = _table_text(tag)
            block_type = TextBlockType.TABLE
        else:
            continue

        if not text:
            continue
        blocks.append(
            TextBlock(
                index=len(blocks),
                block_type=block_type,
                text=text,
                location=_location(tag, section_stack),
                attributes=attributes,
            )
        )

    return tuple(blocks)


def _metadata_values(soup: BeautifulSoup) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for tag in soup.find_all("meta"):
        key_value = tag.get("name") or tag.get("property") or tag.get("http-equiv")
        content_value = tag.get("content")
        if not isinstance(key_value, str) or not isinstance(content_value, str):
            continue
        key = key_value.strip().casefold()
        content = _normalize_text(content_value)
        if key and content:
            collected.setdefault(key, []).append(content)
    return {key: tuple(values) for key, values in collected.items()}


def _document_title(
    soup: BeautifulSoup,
    metadata_values: dict[str, tuple[str, ...]],
) -> str | None:
    if soup.title is not None:
        title = _text(soup.title)
        if title:
            return title
    for key in ("og:title", "twitter:title"):
        values = metadata_values.get(key, ())
        if values:
            return values[0]
    heading = soup.find("h1")
    if isinstance(heading, Tag):
        return _text(heading) or None
    return None


def _authors(metadata_values: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    authors: list[str] = []
    seen: set[str] = set()
    for key in ("author", "article:author", "dc.creator", "dcterms.creator"):
        for author in metadata_values.get(key, ()):
            normalized = author.casefold()
            if normalized not in seen:
                seen.add(normalized)
                authors.append(author)
    return tuple(authors)


def _first_datetime(
    metadata_values: dict[str, tuple[str, ...]],
    keys: Iterable[str],
) -> datetime | None:
    for key in keys:
        for value in metadata_values.get(key, ()):
            prepared = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
            try:
                return datetime.fromisoformat(prepared)
            except ValueError:
                continue
    return None


def _language(
    soup: BeautifulSoup,
    metadata_values: dict[str, tuple[str, ...]],
) -> str | None:
    html = soup.find("html")
    if isinstance(html, Tag):
        value = html.get("lang") or html.get("xml:lang")
        if isinstance(value, str) and value.strip():
            return value.strip()
    values = metadata_values.get("content-language", ())
    return values[0] if values else None


def _document_properties(
    soup: BeautifulSoup,
    metadata_values: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    properties: dict[str, object] = {}
    encoding = soup.original_encoding
    if encoding:
        properties["encoding"] = encoding

    description = _first_value(
        metadata_values,
        ("description", "og:description", "twitter:description"),
    )
    if description:
        properties["description"] = description

    keyword_values = metadata_values.get("keywords", ())
    keywords = tuple(
        keyword.strip()
        for value in keyword_values
        for keyword in value.split(",")
        if keyword.strip()
    )
    if keywords:
        properties["keywords"] = keywords

    generator = _first_value(metadata_values, ("generator",))
    if generator:
        properties["generator"] = generator
    return properties


def _first_value(
    metadata_values: dict[str, tuple[str, ...]],
    keys: Iterable[str],
) -> str | None:
    for key in keys:
        values = metadata_values.get(key, ())
        if values:
            return values[0]
    return None


def _inside(tag: Tag, ancestor_names: set[str] | frozenset[str]) -> bool:
    return any(
        isinstance(parent, Tag) and parent.name in ancestor_names
        for parent in tag.parents
    )


def _text(tag: Tag) -> str:
    return _normalize_text(tag.get_text(" ", strip=True))


def _list_item_text(tag: Tag) -> str:
    parts: list[str] = []
    for value in tag.descendants:
        if isinstance(value, NavigableString) and value.find_parent("li") is tag:
            text = _normalize_text(str(value))
            if text:
                parts.append(text)
    return _normalize_text(" ".join(parts))


def _blockquote_text(tag: Tag) -> str:
    nested_block_tags = _HEADING_TAGS | {"blockquote", "li", "p", "pre", "table"}
    parts: list[str] = []
    for value in tag.descendants:
        if not isinstance(value, NavigableString):
            continue
        parent = value.parent
        nested = False
        while isinstance(parent, Tag) and parent is not tag:
            if parent.name in nested_block_tags:
                nested = True
                break
            parent = parent.parent
        if not nested:
            text = _normalize_text(str(value))
            if text:
                parts.append(text)
    return _normalize_text(" ".join(parts))


def _code_text(tag: Tag) -> str:
    return (
        tag.get_text("", strip=False)
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip("\n")
    )


def _code_language(tag: Tag) -> str | None:
    code_tag = tag.find("code") if tag.name == "pre" else tag
    if not isinstance(code_tag, Tag):
        return None
    classes = code_tag.get("class", ())
    if not isinstance(classes, list):
        return None
    for class_name in classes:
        if not isinstance(class_name, str):
            continue
        for prefix in ("lang-", "language-"):
            if class_name.startswith(prefix) and class_name.removeprefix(prefix):
                return class_name.removeprefix(prefix)
    return None


def _table_text(tag: Tag) -> str:
    rows: list[str] = []
    for row in tag.find_all("tr"):
        cells = [
            _text(cell)
            for cell in row.find_all(("th", "td"), recursive=False)
        ]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _location(
    tag: Tag,
    section_stack: list[tuple[int, str]],
) -> SourceLocation:
    source_line = tag.sourceline
    if not isinstance(source_line, int) or source_line <= 0:
        source_line = None
    return SourceLocation(
        section_path=tuple(title for _, title in section_stack),
        source_line_start=source_line,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\x00", "").split())
