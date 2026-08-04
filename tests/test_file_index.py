from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence

from PIL import Image

from filelore.embedding import EmbeddingVector, ImageEmbedding
from filelore.index import (
    FileIndexRepository,
    FileMetadataQuery,
    calculate_file_hash,
    file_point_id,
)
from filelore.metadata import ImageMetadataParser
from filelore.storage import (
    ConditionOperator,
    DistanceMetric,
    MetadataCondition,
    MetadataFilter,
    VectorConfig,
    VectorDatabase,
)
from filelore.storage.qdrant import QdrantVectorDatabase


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 6), color=color).save(path)


def test_global_index_persists_and_upserts_by_absolute_path(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    database_path = tmp_path / "database"
    create_image(image_path, (10, 20, 30))
    metadata = ImageMetadataParser().parse(image_path)

    with QdrantVectorDatabase(database_path) as database:
        assert isinstance(database, VectorDatabase)
        index = FileIndexRepository(database)
        first = index.store(metadata)
        second = index.store(metadata)

        assert first.id == second.id == file_point_id(image_path)
        assert first.content_hash == calculate_file_hash(image_path)
        assert index.count() == 1

    with QdrantVectorDatabase(database_path) as database:
        index = FileIndexRepository(database)
        restored = index.get_by_path(image_path)

        assert restored is not None
        assert restored.path == image_path.resolve()
        assert restored.metadata["width"] == 10
        assert restored.metadata["height"] == 6
        assert restored.content_hash == first.content_hash


def test_metadata_filters_and_duplicate_detection(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    duplicate_path = tmp_path / "copy.png"
    other_path = tmp_path / "other.jpg"
    create_image(first_path, (10, 20, 30))
    shutil.copyfile(first_path, duplicate_path)
    create_image(other_path, (30, 20, 10))
    parser = ImageMetadataParser()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        index = FileIndexRepository(database)
        for path in (first_path, duplicate_path, other_path):
            index.store(parser.parse(path))

        png_files = index.search_metadata(
            MetadataFilter(
                all_of=(MetadataCondition("extension", ".png"),)
            )
        )
        large_files = index.search_metadata(
            MetadataFilter(
                all_of=(
                    MetadataCondition(
                        "metadata.width",
                        5,
                        operator=ConditionOperator.GREATER_THAN,
                    ),
                )
            )
        )
        duplicates = index.find_duplicate_groups()

        assert {file.path for file in png_files} == {
            first_path.resolve(),
            duplicate_path.resolve(),
        }
        assert len(large_files) == 3
        assert len(duplicates) == 1
        assert {file.path for file in duplicates[0].files} == {
            first_path.resolve(),
            duplicate_path.resolve(),
        }


def test_named_vector_search_can_be_combined_with_metadata_filter(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.jpg"
    create_image(first_path, (10, 20, 30))
    create_image(second_path, (30, 20, 10))
    parser = ImageMetadataParser()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        index = FileIndexRepository(
            database,
            vector_configs={
                "image": VectorConfig(3, distance=DistanceMetric.COSINE)
            },
        )
        index.store(parser.parse(first_path), vectors={"image": [1.0, 0.0, 0.0]})
        index.store(parser.parse(second_path), vectors={"image": [0.0, 1.0, 0.0]})

        results = index.semantic_search(
            [0.9, 0.1, 0.0],
            vector_name="image",
            metadata_filter=MetadataFilter(
                all_of=(MetadataCondition("file_type", "image"),)
            ),
        )

        assert [result.file.path for result in results] == [
            first_path.resolve(),
            second_path.resolve(),
        ]
        assert results[0].score > results[1].score


def test_existing_metadata_collection_can_add_a_named_vector(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.png"
    database_path = tmp_path / "database"
    create_image(image_path, (10, 20, 30))
    metadata = ImageMetadataParser().parse(image_path)

    with QdrantVectorDatabase(database_path) as database:
        FileIndexRepository(database).store(metadata)

    with QdrantVectorDatabase(database_path) as database:
        index = FileIndexRepository(
            database,
            vector_configs={
                "image": VectorConfig(3, distance=DistanceMetric.COSINE)
            },
        )
        index.store(metadata, vectors={"image": [1.0, 0.0, 0.0]})
        results = index.semantic_search([1.0, 0.0, 0.0], vector_name="image")

        assert [result.file.path for result in results] == [image_path.resolve()]


def test_image_embedding_supports_semantic_and_similarity_search(
    tmp_path: Path,
) -> None:
    class ColorImageEmbedding(ImageEmbedding):
        def __init__(self) -> None:
            super().__init__(
                model_id="test-color-model",
                vector_name="image_color",
                dimensions=3,
            )

        def predict_batch(
            self, items: Sequence[str | Path | Image.Image]
        ) -> tuple[EmbeddingVector, ...]:
            vectors: list[tuple[float, float, float]] = []
            for item in items:
                if isinstance(item, Image.Image):
                    color = item.convert("RGB").getpixel((0, 0))
                else:
                    with Image.open(item) as image:
                        color = image.convert("RGB").getpixel((0, 0))
                vectors.append(tuple(float(value) for value in color))
            return self._prepare_vectors(
                vectors, expected_count=len(items), normalize=True
            )

        def predict_text_batch(
            self, texts: Sequence[str]
        ) -> tuple[EmbeddingVector, ...]:
            colors = {
                "red": (1.0, 0.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
            }
            return self._prepare_vectors(
                [colors[text] for text in texts],
                expected_count=len(texts),
                normalize=True,
            )

    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    create_image(red_path, (255, 0, 0))
    create_image(blue_path, (0, 0, 255))
    parser = ImageMetadataParser()
    embedding = ColorImageEmbedding()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        index = FileIndexRepository(
            database,
            vector_configs={
                embedding.vector_name: VectorConfig(
                    embedding.dimensions,
                    distance=DistanceMetric.COSINE,
                )
            },
        )
        for path in (red_path, blue_path):
            index.store(
                parser.parse(path),
                vectors={embedding.vector_name: embedding.predict(path)},
            )

        semantic_results = index.semantic_search(
            embedding.predict_text("red"),
            vector_name=embedding.vector_name,
        )
        similarity_results = index.semantic_search(
            embedding.predict(blue_path),
            vector_name=embedding.vector_name,
        )

        assert semantic_results[0].file.path == red_path.resolve()
        assert similarity_results[0].file.path == blue_path.resolve()


def test_remove_deletes_a_path_record(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    create_image(image_path, (10, 20, 30))

    with QdrantVectorDatabase(tmp_path / "database") as database:
        index = FileIndexRepository(database)
        index.store(ImageMetadataParser().parse(image_path))
        index.remove([image_path])

        assert index.get_by_path(image_path) is None
        assert index.count() == 0


def test_basic_search_supports_name_format_resolution_and_dates(
    tmp_path: Path,
) -> None:
    matching_path = tmp_path / "HolidayPhoto.JPG"
    other_path = tmp_path / "work.png"
    create_image(matching_path, (10, 20, 30))
    create_image(other_path, (30, 20, 10))
    matching_timestamp = datetime(2022, 6, 15, 12, 0).timestamp()
    other_timestamp = datetime(2024, 6, 15, 12, 0).timestamp()
    os.utime(matching_path, (matching_timestamp, matching_timestamp))
    os.utime(other_path, (other_timestamp, other_timestamp))
    parser = ImageMetadataParser()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        index = FileIndexRepository(database)
        index.store_many([parser.parse(matching_path), parser.parse(other_path)])

        results = index.search_files(
            FileMetadataQuery(
                name_contains="holiday",
                file_format="jpg",
                min_width=5,
                min_height=5,
                max_width=20,
                max_height=10,
                modified_after=datetime(2022, 1, 1).astimezone(),
                modified_before=datetime(2022, 12, 31, 23, 59).astimezone(),
            )
        )

        assert [result.path for result in results] == [matching_path.resolve()]
        assert len(index.search_files(FileMetadataQuery())) == 2
