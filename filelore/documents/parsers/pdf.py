"""Page-aware PDF document parser backed by pypdf."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
)
from filelore.documents.parsers.base import DocumentParser
from filelore.metadata import DocumentFormat, DocumentMetadata


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")


class PdfDocumentParser(DocumentParser):
    """Extract page-local text and normalized metadata from text PDFs."""

    supported_extensions = frozenset({".pdf"})
    default_max_content_stream_bytes = 50 * 1024 * 1024

    def __init__(
        self,
        *,
        max_content_stream_bytes: int | None = default_max_content_stream_bytes,
    ) -> None:
        if max_content_stream_bytes is not None and max_content_stream_bytes <= 0:
            raise ValueError("PDF content stream limit must be positive")
        self.max_content_stream_bytes = max_content_stream_bytes

    def parse(self, path: str | Path) -> ParsedDocument:
        document_path = self.prepare_path(path)
        reader: PdfReader | None = None
        try:
            reader = PdfReader(document_path, strict=False)
            if reader.is_encrypted:
                raise ValueError(
                    f"Encrypted PDF documents are not supported: {document_path}"
                )

            page_count = len(reader.pages)
            if page_count == 0:
                raise ValueError(f"PDF document contains no pages: {document_path}")

            section_paths = _section_paths(reader, page_count)
            blocks: list[TextBlock] = []
            for page_index, page in enumerate(reader.pages):
                page_number = page_index + 1
                text = _extract_page_text(
                    page,
                    page_number=page_number,
                    maximum=self.max_content_stream_bytes,
                )
                location = SourceLocation(
                    page_number=page_number,
                    section_path=section_paths[page_index],
                )
                for paragraph in _paragraphs(text or ""):
                    blocks.append(
                        TextBlock(
                            index=len(blocks),
                            block_type=TextBlockType.PARAGRAPH,
                            text=paragraph,
                            location=location,
                        )
                    )

            metadata = _document_metadata(reader, document_path, page_count)
            return ParsedDocument(metadata=metadata, blocks=tuple(blocks))
        except ValueError:
            raise
        except (PdfReadError, KeyError, TypeError, IndexError) as error:
            raise ValueError(
                f"Could not parse PDF document: {document_path}"
            ) from error
        finally:
            if reader is not None:
                reader.close()


def _document_metadata(
    reader: PdfReader,
    path: Path,
    page_count: int,
) -> DocumentMetadata:
    regular = _safe_reader_property(reader, "metadata")
    xmp = _safe_reader_property(reader, "xmp_metadata")

    title = _first_text(
        _safe_attribute(regular, "title"),
        _language_alternative(_safe_attribute(xmp, "dc_title")),
    )
    authors = _unique_texts(
        _iter_values(_safe_attribute(xmp, "dc_creator")),
        _iter_values(_safe_attribute(regular, "author")),
    )
    created_at = _first_datetime(
        _safe_attribute(regular, "creation_date"),
        _safe_attribute(xmp, "xmp_create_date"),
    )
    content_modified_at = _first_datetime(
        _safe_attribute(regular, "modification_date"),
        _safe_attribute(xmp, "xmp_modify_date"),
    )
    language_values = _unique_texts(
        _iter_values(_safe_attribute(xmp, "dc_language"))
    )
    stat = path.stat()

    return DocumentMetadata(
        path=path,
        extension=path.suffix.casefold(),
        mime_type="application/pdf",
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
        document_format=DocumentFormat.PDF,
        title=title,
        authors=authors,
        created_at=created_at,
        content_modified_at=content_modified_at,
        page_count=page_count,
        language=language_values[0] if language_values else None,
        properties=_document_properties(reader, regular, xmp),
    )


def _document_properties(reader: PdfReader, regular: Any, xmp: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    pdf_header = _first_text(_safe_reader_property(reader, "pdf_header"))
    if pdf_header:
        properties["pdf_version"] = pdf_header.removeprefix("%PDF-")

    subject = _first_text(
        _safe_attribute(regular, "subject"),
        _language_alternative(_safe_attribute(xmp, "dc_description")),
    )
    if subject:
        properties["subject"] = subject

    creator = _first_text(_safe_attribute(regular, "creator"))
    if creator:
        properties["creator"] = creator

    producer = _first_text(
        _safe_attribute(regular, "producer"),
        _safe_attribute(xmp, "pdf_producer"),
    )
    if producer:
        properties["producer"] = producer

    keyword_values = list(_iter_values(_safe_mapping_value(regular, "/Keywords")))
    keyword_values.extend(_iter_values(_safe_attribute(xmp, "dc_subject")))
    keyword_values.extend(_iter_values(_safe_attribute(xmp, "pdf_keywords")))
    keywords = _keywords(keyword_values)
    if keywords:
        properties["keywords"] = keywords
    return properties


def _section_paths(
    reader: PdfReader,
    page_count: int,
) -> tuple[tuple[str, ...], ...]:
    try:
        outline = reader.outline
    except (PdfReadError, KeyError, TypeError, ValueError):
        return ((),) * page_count

    entries: list[tuple[int, int, tuple[str, ...]]] = []
    order = 0

    def visit(items: Iterable[Any], parent_path: tuple[str, ...]) -> None:
        nonlocal order
        previous_path = parent_path
        for item in items:
            if isinstance(item, list):
                visit(item, previous_path)
                continue
            title = _first_text(_safe_attribute(item, "title"))
            if not title:
                previous_path = parent_path
                continue
            try:
                page_index = reader.get_destination_page_number(item)
            except (PdfReadError, KeyError, TypeError, ValueError):
                previous_path = parent_path
                continue
            if not 0 <= page_index < page_count:
                previous_path = parent_path
                continue
            path = parent_path + (title,)
            entries.append((page_index, order, path))
            order += 1
            previous_path = path

    visit(outline, ())
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    paths: list[tuple[str, ...]] = []
    current: tuple[str, ...] = ()
    entry_index = 0
    for page_index in range(page_count):
        while entry_index < len(entries) and entries[entry_index][0] <= page_index:
            current = entries[entry_index][2]
            entry_index += 1
        paths.append(current)
    return tuple(paths)


def _extract_page_text(
    page: Any,
    *,
    page_number: int,
    maximum: int | None,
) -> str:
    contents = page.get_contents()
    if contents is None:
        return ""
    size = len(contents.get_data()) if maximum is not None else None
    if maximum is not None and size is not None and size > maximum:
        raise ValueError(
            f"PDF page {page_number} content stream exceeds the "
            f"{maximum}-byte safety limit"
        )
    return (
        page.extract_text(
            extraction_mode="layout",
            layout_mode_space_vertically=True,
            layout_mode_strip_rotated=True,
        )
        or ""
    )


def _paragraphs(value: str) -> tuple[str, ...]:
    normalized = (
        value.replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    paragraphs: list[str] = []
    for raw_paragraph in _PARAGRAPH_BREAK.split(normalized):
        lines = (" ".join(line.split()) for line in raw_paragraph.splitlines())
        paragraph = " ".join(line for line in lines if line)
        if paragraph:
            paragraphs.append(paragraph)
    return tuple(paragraphs)


def _safe_reader_property(reader: PdfReader, name: str) -> Any:
    try:
        return getattr(reader, name, None)
    except (PdfReadError, KeyError, TypeError, ValueError):
        return None


def _safe_attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, name, None)
    except (PdfReadError, KeyError, TypeError, ValueError):
        return None


def _safe_mapping_value(value: Any, key: str) -> Any:
    if value is None or not hasattr(value, "get"):
        return None
    try:
        return value.get(key)
    except (PdfReadError, KeyError, TypeError, ValueError):
        return None


def _language_alternative(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "x-default" in value:
        return value["x-default"]
    return next(iter(value.values()), None)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = " ".join(str(value).replace("\x00", "").split())
        if text:
            return text
    return None


def _iter_values(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return (value,)


def _unique_texts(*value_groups: Iterable[Any]) -> tuple[str, ...]:
    prepared: list[str] = []
    seen: set[str] = set()
    for values in value_groups:
        for value in values:
            text = _first_text(value)
            if text is None or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            prepared.append(text)
    return tuple(prepared)


def _first_datetime(*values: Any) -> datetime | None:
    for value in values:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            prepared = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
            try:
                return datetime.fromisoformat(prepared)
            except ValueError:
                continue
    return None


def _keywords(values: Iterable[Any]) -> tuple[str, ...]:
    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        for candidate in re.split(r"[,;]", str(value)):
            keyword = _first_text(candidate)
            if keyword is None or keyword.casefold() in seen:
                continue
            seen.add(keyword.casefold())
            keywords.append(keyword)
    return tuple(keywords)
