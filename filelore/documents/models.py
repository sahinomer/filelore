"""Format-neutral value objects for parsed and chunked documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from filelore.metadata.document import DocumentMetadata


class TextBlockType(StrEnum):
    """Structural roles a parser can assign to extracted text."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    SLIDE_TITLE = "slide_title"
    SPEAKER_NOTE = "speaker_note"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Optional source coordinates shared across document formats."""

    page_number: int | None = None
    slide_number: int | None = None
    section_path: tuple[str, ...] = ()
    source_line_start: int | None = None
    source_line_end: int | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("Source page number must be positive")
        if self.slide_number is not None and self.slide_number <= 0:
            raise ValueError("Source slide number must be positive")
        if self.source_line_start is not None and self.source_line_start <= 0:
            raise ValueError("Source line start must be positive")
        if self.source_line_end is not None:
            if self.source_line_start is None:
                raise ValueError("Source line end requires a line start")
            if self.source_line_end < self.source_line_start:
                raise ValueError("Source line end must not precede its start")
        if any(not heading.strip() for heading in self.section_path):
            raise ValueError("Source section headings must not be empty")


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One natural structural unit extracted from a document."""

    index: int
    block_type: TextBlockType
    text: str
    location: SourceLocation = SourceLocation()
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Text block index must be non-negative")
        if not self.text.strip():
            raise ValueError("Text block must contain non-whitespace text")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A parser result independent of the source document format."""

    metadata: DocumentMetadata
    blocks: tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        indices = tuple(block.index for block in self.blocks)
        if indices != tuple(range(len(self.blocks))):
            raise ValueError(
                "Parsed document block indices must be contiguous from zero"
            )


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk ready for embedding with its source context intact."""

    index: int
    text: str
    embedding_text: str
    first_block_index: int
    last_block_index: int
    location: SourceLocation = SourceLocation()

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Text chunk index must be non-negative")
        if not self.text.strip():
            raise ValueError("Text chunk must contain non-whitespace text")
        if not self.embedding_text.strip():
            raise ValueError("Text chunk embedding text must not be empty")
        if self.first_block_index < 0:
            raise ValueError("Text chunk block indices must be non-negative")
        if self.last_block_index < self.first_block_index:
            raise ValueError(
                "Text chunk last block must not precede its first block"
            )
