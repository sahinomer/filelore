"""Markdown parser backed by markdown-it-py tokens."""

from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
)
from filelore.documents.parsers.base import DocumentParser
from filelore.metadata import DocumentFormat, DocumentMetadata


class MarkdownDocumentParser(DocumentParser):
    """Extract structural text blocks from UTF-8 Markdown documents."""

    supported_extensions = frozenset({".md"})

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark").enable("table")

    def parse(self, path: str | Path) -> ParsedDocument:
        document_path = self.prepare_path(path)
        source = document_path.read_text(encoding="utf-8-sig")
        tokens = self._markdown.parse(source.replace("\x00", ""))
        blocks, title = _extract_blocks(tokens)
        stat = document_path.stat()

        return ParsedDocument(
            metadata=DocumentMetadata(
                path=document_path,
                extension=document_path.suffix.casefold(),
                mime_type="text/markdown",
                size_bytes=stat.st_size,
                modified_at=_modified_at(stat.st_mtime),
                document_format=DocumentFormat.MARKDOWN,
                title=title,
            ),
            blocks=blocks,
        )


def _extract_blocks(
    tokens: list[Token],
) -> tuple[tuple[TextBlock, ...], str | None]:
    blocks: list[TextBlock] = []
    section_stack: list[tuple[int, str]] = []
    title: str | None = None
    list_item_depth = 0
    quote_depth = 0
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token.type == "list_item_open":
            list_item_depth += 1
        elif token.type == "list_item_close":
            list_item_depth -= 1
        elif token.type == "blockquote_open":
            quote_depth += 1
        elif token.type == "blockquote_close":
            quote_depth -= 1
        elif token.type == "heading_open":
            text = _following_inline_text(tokens, index)
            if text:
                level = int(token.tag.removeprefix("h"))
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, text))
                if title is None and level == 1:
                    title = text
                blocks.append(
                    TextBlock(
                        index=len(blocks),
                        block_type=TextBlockType.HEADING,
                        text=text,
                        location=_location(token, section_stack),
                        attributes={"level": level},
                    )
                )
        elif token.type == "paragraph_open":
            text = _following_inline_text(tokens, index)
            if text:
                if list_item_depth:
                    block_type = TextBlockType.LIST_ITEM
                elif quote_depth:
                    block_type = TextBlockType.QUOTE
                else:
                    block_type = TextBlockType.PARAGRAPH
                blocks.append(
                    TextBlock(
                        index=len(blocks),
                        block_type=block_type,
                        text=text,
                        location=_location(token, section_stack),
                    )
                )
        elif token.type in {"fence", "code_block"}:
            text = _normalize_text(token.content)
            if text:
                attributes: dict[str, Any] = {}
                language = token.info.strip()
                if language:
                    attributes["language"] = language
                blocks.append(
                    TextBlock(
                        index=len(blocks),
                        block_type=TextBlockType.CODE,
                        text=text,
                        location=_location(token, section_stack),
                        attributes=attributes,
                    )
                )
        elif token.type == "html_block":
            text = _visible_html_text(token.content)
            if text:
                blocks.append(
                    TextBlock(
                        index=len(blocks),
                        block_type=TextBlockType.OTHER,
                        text=text,
                        location=_location(token, section_stack),
                        attributes={"source": "html"},
                    )
                )
        elif token.type == "table_open":
            text, closing_index = _table_text(tokens, index)
            if text:
                blocks.append(
                    TextBlock(
                        index=len(blocks),
                        block_type=TextBlockType.TABLE,
                        text=text,
                        location=_location(token, section_stack),
                    )
                )
            index = closing_index

        index += 1

    return tuple(blocks), title


def _following_inline_text(tokens: list[Token], index: int) -> str:
    following_index = index + 1
    if following_index >= len(tokens):
        return ""
    following = tokens[following_index]
    if following.type != "inline":
        return ""
    return _inline_text(following)


def _inline_text(token: Token) -> str:
    if not token.children:
        return _normalize_text(token.content)

    parts: list[str] = []
    for child in token.children:
        if child.type in {"text", "code_inline", "image"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif child.type == "html_inline":
            parts.append(_visible_html_text(child.content))
    return _normalize_text("".join(parts))


def _table_text(tokens: list[Token], opening_index: int) -> tuple[str, int]:
    rows: list[str] = []
    cells: list[str] | None = None
    index = opening_index + 1

    while index < len(tokens):
        token = tokens[index]
        if token.type == "tr_open":
            cells = []
        elif token.type == "inline" and cells is not None:
            text = _inline_text(token)
            cells.append(text)
        elif token.type == "tr_close" and cells is not None:
            if cells:
                rows.append(" | ".join(cells))
            cells = None
        elif token.type == "table_close":
            return "\n".join(rows), index
        index += 1

    return "\n".join(rows), index - 1


def _location(
    token: Token,
    section_stack: list[tuple[int, str]],
) -> SourceLocation:
    line_start: int | None = None
    line_end: int | None = None
    if token.map is not None:
        line_start = token.map[0] + 1
        line_end = token.map[1]
    return SourceLocation(
        section_path=tuple(title for _, title in section_stack),
        source_line_start=line_start,
        source_line_end=line_end,
    )


def _normalize_text(text: str) -> str:
    return (
        text.replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


class _VisibleHtmlParser(HTMLParser):
    _BREAK_TAGS = frozenset(
        {"br", "div", "li", "p", "pre", "table", "td", "th", "tr"}
    )
    _SKIPPED_TAGS = frozenset({"noscript", "script", "style", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipped_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in self._SKIPPED_TAGS:
            self._skipped_depth += 1
            return
        if self._skipped_depth:
            return
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS:
            if self._skipped_depth:
                self._skipped_depth -= 1
            return
        if self._skipped_depth:
            return
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipped_depth:
            self.parts.append(data)


def _visible_html_text(value: str) -> str:
    parser = _VisibleHtmlParser()
    parser.feed(value)
    parser.close()
    lines = (
        " ".join(line.split())
        for line in "".join(parser.parts).splitlines()
    )
    return "\n".join(line for line in lines if line)


def _modified_at(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp).astimezone()
