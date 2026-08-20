from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from filelore.documents import ParagraphChunker
from filelore.embedding import DocumentEmbedding, EmbeddingVector
from filelore.processors import DocumentProcessor


class RecordingDocumentEmbedding(DocumentEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="test-document-model",
            vector_name="text_test",
            dimensions=3,
        )
        self.batches: list[tuple[str, ...]] = []

    def predict_batch(
        self, items: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        self.batches.append(tuple(items))
        return tuple(
            (float(index + 1), 0.0, 0.0)
            for index, _ in enumerate(items)
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)


def test_document_processor_parses_chunks_and_embeds_successful_files(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "city-guide.md"
    second_path = tmp_path / "museum-guide.md"
    invalid_path = tmp_path / "invalid.pdf"
    first_path.write_text(
        "# City Guide\n\n"
        "## Public Transport\n\n"
        "Metro services connect the central districts.\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "# Museum Guide\n\n"
        "The history museum opens every weekday.\n",
        encoding="utf-8",
    )
    invalid_path.write_text("not a PDF", encoding="utf-8")
    embedding = RecordingDocumentEmbedding()

    batch = DocumentProcessor(embedding=embedding).process_batch(
        (first_path, invalid_path, second_path)
    )

    assert [item.metadata.path for item in batch.files] == [
        first_path.resolve(),
        second_path.resolve(),
    ]
    assert len(batch.failures) == 1
    assert batch.failures[0].path == invalid_path
    assert len(embedding.batches) == 1
    assert len(embedding.batches[0]) == sum(
        len(item.segments) for item in batch.files
    )

    first_segments = batch.files[0].segments
    assert [segment.index for segment in first_segments] == list(
        range(len(first_segments))
    )
    assert first_segments[0].vectors == {"text_test": (1.0, 0.0, 0.0)}
    assert all(segment.start_seconds is None for segment in first_segments)
    assert all(segment.end_seconds is None for segment in first_segments)
    assert all(
        segment.payload["segment_kind"] == "document_chunk"
        for segment in first_segments
    )
    assert first_segments[0].payload == {
        "segment_kind": "document_chunk",
        "chunk_index": 0,
        "chunk_text": "City Guide",
        "character_count": 10,
        "first_block_index": 0,
        "last_block_index": 0,
        "section_path": ["City Guide"],
        "heading": "City Guide",
        "source_line_start": 1,
        "source_line_end": 1,
    }
    assert first_segments[1].payload["heading"] == "Public Transport"
    assert "Metro services" in str(first_segments[1].payload["chunk_text"])


def test_document_processor_can_prepare_metadata_without_embeddings(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "notes.md"
    document_path.write_text("# Notes\n\nA short paragraph.\n", encoding="utf-8")

    class UnexpectedChunker:
        def chunks(self, document: object) -> tuple[object, ...]:
            raise AssertionError("Metadata-only processing must not chunk text")

    batch = DocumentProcessor(
        chunker=UnexpectedChunker(),  # type: ignore[arg-type]
    ).process_batch((document_path,))

    assert len(batch.files) == 1
    assert batch.files[0].metadata.path == document_path.resolve()
    assert batch.files[0].vectors == {}
    assert batch.files[0].segments == ()
    assert batch.failures == ()


def test_document_processor_discovers_registered_formats(tmp_path: Path) -> None:
    expected = (
        tmp_path / "guide.MD",
        tmp_path / "page.html",
        tmp_path / "report.pdf",
        tmp_path / "slides.pptx",
    )
    nested = tmp_path / "nested" / "letter.docx"
    ignored = tmp_path / "notes.txt"
    nested.parent.mkdir()
    for path in (*expected, nested, ignored):
        path.write_bytes(b"placeholder")
    processor = DocumentProcessor()

    recursive = tuple(processor.discover(tmp_path))
    direct = tuple(processor.discover(tmp_path, recursive=False))

    assert recursive == tuple(sorted((*expected, nested)))
    assert direct == tuple(sorted(expected))


def test_document_processor_uses_replaceable_chunker(tmp_path: Path) -> None:
    document_path = tmp_path / "long.md"
    document_path.write_text(
        "# Notes\n\nThis paragraph contains several separate words.\n",
        encoding="utf-8",
    )
    embedding = RecordingDocumentEmbedding()

    batch = DocumentProcessor(
        embedding=embedding,
        chunker=ParagraphChunker(max_characters=12),
    ).process_batch((document_path,))

    assert len(batch.files[0].segments) > 2
    assert all(
        int(segment.payload["character_count"]) <= 12
        for segment in batch.files[0].segments
    )


def test_document_processor_validates_embedding_count(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.md"
    document_path.write_text("# Notes\n\nA paragraph.\n", encoding="utf-8")
    embedding = RecordingDocumentEmbedding()
    embedding.predict_batch = lambda items: ()  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="embedding count"):
        DocumentProcessor(embedding=embedding).process_batch((document_path,))


def test_document_processor_does_not_embed_an_empty_batch() -> None:
    embedding = RecordingDocumentEmbedding()

    batch = DocumentProcessor(embedding=embedding).process_batch(())

    assert batch.files == ()
    assert batch.failures == ()
    assert embedding.batches == []
