"""Structure-aware paragraph chunking with sentence fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar

from filelore.documents.chunking.base import SentenceSplitter
from filelore.documents.chunking.sentence import UnicodeSentenceSplitter
from filelore.documents.models import (
    ParsedDocument,
    SourceLocation,
    TextBlock,
    TextBlockType,
    TextChunk,
)


@dataclass(frozen=True, slots=True)
class _ChunkUnit:
    text: str
    block_index: int
    block_type: TextBlockType
    location: SourceLocation
    separator_before: str = ""


@dataclass(frozen=True, slots=True)
class ParagraphChunker:
    """Combine structural blocks and split oversized prose naturally."""

    default_max_characters: ClassVar[int] = 1_600

    max_characters: int = default_max_characters
    sentence_splitter: SentenceSplitter = field(
        default_factory=UnicodeSentenceSplitter
    )

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("Chunk character limit must be positive")

    def chunks(self, document: ParsedDocument) -> tuple[TextChunk, ...]:
        """Return chunks that never cross page, slide, or section scopes."""

        chunks: list[TextChunk] = []
        pending: list[_ChunkUnit] = []
        pending_length = 0

        def flush() -> None:
            nonlocal pending_length
            if not pending:
                return
            chunks.append(_text_chunk(len(chunks), pending))
            pending.clear()
            pending_length = 0

        for block in document.blocks:
            for unit in _block_units(
                block,
                maximum=self.max_characters,
                sentence_splitter=self.sentence_splitter,
            ):
                if pending and _scope(pending[-1].location) != _scope(unit.location):
                    flush()

                separator = _separator(pending[-1], unit) if pending else ""
                projected_length = pending_length + len(separator) + len(unit.text)
                if pending and projected_length > self.max_characters:
                    flush()
                    separator = ""
                    projected_length = len(unit.text)

                pending.append(unit)
                pending_length = projected_length

        flush()
        return tuple(chunks)


def _block_units(
    block: TextBlock,
    *,
    maximum: int,
    sentence_splitter: SentenceSplitter,
) -> tuple[_ChunkUnit, ...]:
    if block.block_type in {TextBlockType.CODE, TextBlockType.TABLE}:
        pieces = _line_pieces(block.text, maximum)
    elif block.block_type is TextBlockType.HEADING:
        pieces = _bounded_pieces(block.text.strip(), maximum, " ")
    else:
        pieces = _sentence_pieces(block.text, maximum, sentence_splitter)

    return tuple(
        _ChunkUnit(
            text=text,
            block_index=block.index,
            block_type=block.block_type,
            location=block.location,
            separator_before=separator,
        )
        for text, separator in pieces
        if text
    )


def _sentence_pieces(
    text: str,
    maximum: int,
    sentence_splitter: SentenceSplitter,
) -> tuple[tuple[str, str], ...]:
    sentences = sentence_splitter.split(text)
    if not sentences:
        normalized = " ".join(text.replace("\x00", "").split())
        sentences = (normalized,) if normalized else ()

    pieces: list[tuple[str, str]] = []
    for sentence in sentences:
        separator = "" if not pieces else " "
        for piece_index, piece in enumerate(
            _split_oversized_text(sentence, maximum)
        ):
            pieces.append((piece, separator if piece_index == 0 else " "))
    return tuple(pieces)


def _line_pieces(text: str, maximum: int) -> tuple[tuple[str, str], ...]:
    pieces: list[tuple[str, str]] = []
    blank_lines = 0
    for line in text.replace("\x00", "").splitlines():
        if not line.strip():
            blank_lines += 1
            continue

        separator = "" if not pieces else "\n" * (blank_lines + 1)
        blank_lines = 0
        for piece_index, piece in enumerate(_split_oversized_text(line, maximum)):
            pieces.append((piece, separator if piece_index == 0 else " "))
    return tuple(pieces)


def _bounded_pieces(
    text: str,
    maximum: int,
    separator: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (piece, "" if index == 0 else separator)
        for index, piece in enumerate(_split_oversized_text(text, maximum))
    )


def _split_oversized_text(text: str, maximum: int) -> tuple[str, ...]:
    if len(text) <= maximum:
        return (text,) if text else ()

    words = text.split()
    if not words:
        return ()

    pieces: list[str] = []
    pending = ""
    for word in words:
        if len(word) > maximum:
            if pending:
                pieces.append(pending)
                pending = ""
            pieces.extend(
                word[start : start + maximum]
                for start in range(0, len(word), maximum)
            )
            continue

        candidate = f"{pending} {word}" if pending else word
        if len(candidate) <= maximum:
            pending = candidate
        else:
            pieces.append(pending)
            pending = word

    if pending:
        pieces.append(pending)
    return tuple(pieces)


def _separator(previous: _ChunkUnit, current: _ChunkUnit) -> str:
    if previous.block_index == current.block_index:
        return current.separator_before
    if (
        previous.block_type is TextBlockType.LIST_ITEM
        and current.block_type is TextBlockType.LIST_ITEM
    ):
        return "\n"
    return "\n\n"


def _scope(location: SourceLocation) -> tuple[object, ...]:
    return (
        location.page_number,
        location.slide_number,
        location.section_path,
    )


def _text_chunk(index: int, units: list[_ChunkUnit]) -> TextChunk:
    text = units[0].text
    for previous, current in pairwise(units):
        text += _separator(previous, current) + current.text

    location = _combined_location(units)
    return TextChunk(
        index=index,
        text=text,
        embedding_text=_embedding_text(text, location, units),
        first_block_index=units[0].block_index,
        last_block_index=units[-1].block_index,
        location=location,
    )


def _combined_location(units: list[_ChunkUnit]) -> SourceLocation:
    first = units[0].location
    starts = [
        unit.location.source_line_start
        for unit in units
        if unit.location.source_line_start is not None
    ]
    ends = [
        unit.location.source_line_end
        for unit in units
        if unit.location.source_line_end is not None
    ]
    line_start = min(starts) if len(starts) == len(units) else None
    line_end = (
        max(ends)
        if line_start is not None and len(ends) == len(units)
        else None
    )
    return SourceLocation(
        page_number=first.page_number,
        slide_number=first.slide_number,
        section_path=first.section_path,
        source_line_start=line_start,
        source_line_end=line_end,
    )


def _embedding_text(
    text: str,
    location: SourceLocation,
    units: list[_ChunkUnit],
) -> str:
    included_headings = {
        unit.text.casefold()
        for unit in units
        if unit.block_type is TextBlockType.HEADING
    }
    missing_context = tuple(
        heading
        for heading in location.section_path
        if heading.casefold() not in included_headings
    )
    if not missing_context:
        return text
    return "\n".join((*missing_context, "", text))
