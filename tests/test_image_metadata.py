from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from filelore.metadata import (
    BaseMetadata,
    ImageMetadata,
    ImageMetadataParser,
    MetadataParser,
)


def create_image(path: Path, *, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(25, 50, 75)).save(path)


def test_image_parser_extracts_file_and_image_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    create_image(image_path)

    parser = ImageMetadataParser()
    metadata = parser.parse(image_path)

    assert isinstance(parser, MetadataParser)
    assert isinstance(metadata, ImageMetadata)
    assert isinstance(metadata, BaseMetadata)
    assert metadata.path == image_path.resolve()
    assert metadata.extension == ".png"
    assert metadata.mime_type == "image/png"
    assert metadata.size_bytes == image_path.stat().st_size
    assert metadata.width == 12
    assert metadata.height == 8
    assert metadata.image_format == "PNG"
    assert metadata.color_mode == "RGB"
    assert metadata.frame_count == 1
    assert metadata.is_animated is False

    serialized = metadata.to_dict()
    assert serialized["path"] == str(image_path.resolve())
    assert isinstance(serialized["modified_at"], str)


def test_image_parser_extracts_named_exif_tags(tmp_path: Path) -> None:
    image_path = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[271] = "FileLore Camera"
    Image.new("RGB", (4, 3)).save(image_path, exif=exif)

    metadata = ImageMetadataParser().parse(image_path)

    assert metadata.exif["Make"] == "FileLore Camera"
    json.dumps(metadata.to_dict())


def test_discover_filters_extensions_and_can_recurse(tmp_path: Path) -> None:
    top_image = tmp_path / "top.PNG"
    nested_image = tmp_path / "nested" / "child.jpg"
    create_image(top_image)
    create_image(nested_image)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    parser = ImageMetadataParser()

    assert list(parser.discover(tmp_path, recursive=False)) == [top_image]
    assert list(parser.discover(tmp_path)) == [nested_image, top_image]


def test_parser_rejects_an_unsupported_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image extension"):
        ImageMetadataParser().parse(text_path)
