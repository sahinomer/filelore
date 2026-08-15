"""Metadata shared by parsed text documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from filelore.metadata.base import BaseMetadata


class DocumentFormat(StrEnum):
    """Document formats supported by the initial text pipeline."""

    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    PPTX = "pptx"
    MARKDOWN = "markdown"


@dataclass(frozen=True, slots=True)
class DocumentMetadata(BaseMetadata):
    """Normalized metadata common to every parsed document format."""

    file_type: ClassVar[str] = "text"

    document_format: DocumentFormat
    title: str | None = None
    authors: tuple[str, ...] = ()
    created_at: datetime | None = None
    content_modified_at: datetime | None = None
    page_count: int | None = None
    slide_count: int | None = None
    language: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_count is not None and self.page_count <= 0:
            raise ValueError("Document page count must be positive")
        if self.slide_count is not None and self.slide_count <= 0:
            raise ValueError("Document slide count must be positive")
