from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from filelore.embedding import EmbeddingVector, ImageEmbedding
from filelore.processors import ImageProcessor


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 6), color=color).save(path)


class RecordingImageEmbedding(ImageEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="test-image-model",
            vector_name="image_test",
            dimensions=3,
        )
        self.batches: list[tuple[str | Path | Image.Image, ...]] = []

    def predict_batch(
        self, items: Sequence[str | Path | Image.Image]
    ) -> tuple[EmbeddingVector, ...]:
        self.batches.append(tuple(items))
        return tuple((1.0, 0.0, 0.0) for _ in items)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)


def test_image_processor_batches_successes_and_isolates_invalid_files(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    invalid_path = tmp_path / "invalid.png"
    second_path = tmp_path / "second.jpg"
    create_image(first_path, (255, 0, 0))
    invalid_path.write_text("not an image", encoding="utf-8")
    create_image(second_path, (0, 0, 255))
    embedding = RecordingImageEmbedding()

    batch = ImageProcessor(embedding=embedding).process_batch(
        (first_path, invalid_path, second_path)
    )

    assert [file.metadata.path for file in batch.files] == [
        first_path.resolve(),
        second_path.resolve(),
    ]
    assert [file.vectors for file in batch.files] == [
        {"image_test": (1.0, 0.0, 0.0)},
        {"image_test": (1.0, 0.0, 0.0)},
    ]
    assert embedding.batches == [
        (first_path.resolve(), second_path.resolve())
    ]
    assert len(batch.failures) == 1
    assert batch.failures[0].path == invalid_path


def test_image_processor_can_prepare_metadata_without_embeddings(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.png"
    create_image(image_path, (25, 50, 75))

    batch = ImageProcessor().process_batch((image_path,))

    assert len(batch.files) == 1
    assert batch.files[0].metadata.path == image_path.resolve()
    assert batch.files[0].vectors == {}
    assert batch.failures == ()


def test_image_processor_does_not_embed_an_empty_batch() -> None:
    embedding = RecordingImageEmbedding()

    batch = ImageProcessor(embedding=embedding).process_batch(())

    assert batch.files == ()
    assert batch.failures == ()
    assert embedding.batches == []
