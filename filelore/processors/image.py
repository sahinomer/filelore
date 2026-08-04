"""Image metadata and embedding processor."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from PIL import UnidentifiedImageError

from filelore.embedding import EmbeddingVector, ImageEmbedding
from filelore.metadata import ImageMetadata, ImageMetadataParser
from filelore.processors.models import (
    PreparedFile,
    ProcessingBatch,
    ProcessingFailure,
)


class ImageProcessor:
    """Prepare image metadata and optional vectors in storage-ready batches."""

    def __init__(
        self,
        *,
        metadata_parser: ImageMetadataParser | None = None,
        embedding: ImageEmbedding | None = None,
    ) -> None:
        self.metadata_parser = metadata_parser or ImageMetadataParser()
        self.embedding = embedding

    def discover(
        self, directory: str | Path, *, recursive: bool = True
    ) -> Iterator[Path]:
        """Yield images supported by the configured metadata parser."""
        return self.metadata_parser.discover(directory, recursive=recursive)

    def process_batch(
        self, paths: Sequence[str | Path]
    ) -> ProcessingBatch[ImageMetadata]:
        """Parse valid images and embed successful items as one model batch."""
        metadata_items: list[ImageMetadata] = []
        failures: list[ProcessingFailure] = []
        for path in paths:
            image_path = Path(path).expanduser()
            try:
                metadata_items.append(self.metadata_parser.parse(image_path))
            except (OSError, ValueError, UnidentifiedImageError) as error:
                failures.append(ProcessingFailure(path=image_path, error=error))

        if not metadata_items:
            return ProcessingBatch(files=(), failures=tuple(failures))

        vector_sets: list[dict[str, EmbeddingVector]]
        if self.embedding is None:
            vector_sets = [{} for _ in metadata_items]
        else:
            vectors = self.embedding.predict_batch(
                tuple(metadata.path for metadata in metadata_items)
            )
            if len(vectors) != len(metadata_items):
                raise ValueError(
                    "Image embedding count must match parsed image count"
                )
            vector_sets = [
                {self.embedding.vector_name: vector} for vector in vectors
            ]

        files = tuple(
            PreparedFile(metadata=metadata, vectors=vectors)
            for metadata, vectors in zip(metadata_items, vector_sets)
        )
        return ProcessingBatch(files=files, failures=tuple(failures))
