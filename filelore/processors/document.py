"""Indexing adapter for parsed, chunked, and vectorized text documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from filelore.documents import (
    DocxDocumentParser,
    DocumentChunker,
    DocumentParserRegistry,
    HtmlDocumentParser,
    MarkdownDocumentParser,
    ParagraphChunker,
    PdfDocumentParser,
    PptxDocumentParser,
    TextChunk,
)
from filelore.embedding import DocumentEmbedding, EmbeddingVector
from filelore.metadata import DocumentMetadata
from filelore.processors.models import (
    PreparedFile,
    PreparedSegment,
    ProcessingBatch,
    ProcessingFailure,
)


def default_document_parser_registry() -> DocumentParserRegistry:
    """Return parsers for every initially supported document format."""
    return DocumentParserRegistry(
        (
            PdfDocumentParser(),
            HtmlDocumentParser(),
            MarkdownDocumentParser(),
            DocxDocumentParser(),
            PptxDocumentParser(),
        )
    )


class DocumentProcessor:
    """Prepare document metadata and chunk vectors for indexing."""

    def __init__(
        self,
        *,
        parser_registry: DocumentParserRegistry | None = None,
        chunker: DocumentChunker | None = None,
        embedding: DocumentEmbedding | None = None,
    ) -> None:
        self.parser_registry = parser_registry or default_document_parser_registry()
        self.chunker = chunker or ParagraphChunker()
        self.embedding = embedding

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]:
        """Yield document paths recognized by the configured parser registry."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {root}")

        pattern = "**/*" if recursive else "*"
        for path in sorted(root.glob(pattern)):
            if (
                path.is_file()
                and path.suffix.casefold()
                in self.parser_registry.supported_extensions
            ):
                yield path

    def process_batch(
        self, paths: Sequence[str | Path]
    ) -> ProcessingBatch[DocumentMetadata]:
        """Parse and chunk valid documents, then embed all chunks as one batch."""
        parsed: list[tuple[DocumentMetadata, tuple[TextChunk, ...]]] = []
        failures: list[ProcessingFailure] = []
        for path in paths:
            document_path = Path(path).expanduser()
            try:
                document = self.parser_registry.parse(document_path)
                chunks = (
                    self.chunker.chunks(document)
                    if self.embedding is not None
                    else ()
                )
            except (OSError, ValueError) as error:
                failures.append(ProcessingFailure(path=document_path, error=error))
                continue
            parsed.append((document.metadata, chunks))

        if not parsed:
            return ProcessingBatch(files=(), failures=tuple(failures))
        if self.embedding is None:
            return ProcessingBatch(
                files=tuple(
                    PreparedFile(metadata=metadata, vectors={})
                    for metadata, _ in parsed
                ),
                failures=tuple(failures),
            )

        flattened_chunks = tuple(
            chunk for _, chunks in parsed for chunk in chunks
        )
        vectors = self.embedding.predict_batch(
            tuple(chunk.embedding_text for chunk in flattened_chunks)
        )
        if len(vectors) != len(flattened_chunks):
            raise ValueError(
                "Document embedding count must match generated chunk count"
            )

        vector_iterator = iter(vectors)
        files = tuple(
            PreparedFile(
                metadata=metadata,
                vectors={},
                segments=tuple(
                    _prepared_chunk(
                        chunk,
                        next(vector_iterator),
                        vector_name=self.embedding.vector_name,
                    )
                    for chunk in chunks
                ),
            )
            for metadata, chunks in parsed
        )
        return ProcessingBatch(files=files, failures=tuple(failures))


def _prepared_chunk(
    chunk: TextChunk,
    vector: EmbeddingVector,
    *,
    vector_name: str,
) -> PreparedSegment:
    location = chunk.location
    payload: dict[str, object] = {
        "segment_kind": "document_chunk",
        "chunk_index": chunk.index,
        "chunk_text": chunk.text,
        "character_count": len(chunk.text),
        "first_block_index": chunk.first_block_index,
        "last_block_index": chunk.last_block_index,
    }
    if location.page_number is not None:
        payload["page_number"] = location.page_number
    if location.slide_number is not None:
        payload["slide_number"] = location.slide_number
    if location.section_path:
        payload["section_path"] = list(location.section_path)
        payload["heading"] = location.section_path[-1]
    if location.source_line_start is not None:
        payload["source_line_start"] = location.source_line_start
    if location.source_line_end is not None:
        payload["source_line_end"] = location.source_line_end

    return PreparedSegment(
        index=chunk.index,
        start_seconds=None,
        end_seconds=None,
        vectors={vector_name: vector},
        payload=payload,
    )
