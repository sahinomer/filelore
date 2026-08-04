"""Image metadata record and Pillow-backed parser."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from PIL import ExifTags, Image

from filelore.metadata.base import BaseMetadata, MetadataParser


@dataclass(frozen=True, slots=True)
class ImageMetadata(BaseMetadata):
    """Metadata specific to raster image files."""

    file_type: ClassVar[str] = "image"

    width: int
    height: int
    image_format: str | None
    color_mode: str
    frame_count: int = 1
    is_animated: bool = False
    exif: dict[str, Any] = field(default_factory=dict)


class ImageMetadataParser(MetadataParser[ImageMetadata]):
    """Extract common image properties and EXIF tags with Pillow."""

    supported_extensions = frozenset(
        {
            ".bmp",
            ".dib",
            ".gif",
            ".ico",
            ".jfif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }
    )

    def parse(self, path: str | Path) -> ImageMetadata:
        image_path = Path(path).expanduser()
        if not self.supports(image_path):
            extension = image_path.suffix or "<none>"
            raise ValueError(f"Unsupported image extension: {extension}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        stat = image_path.stat()
        with Image.open(image_path) as image:
            image_format = image.format
            mime_type = Image.MIME.get(image_format or "")
            if mime_type is None:
                mime_type = mimetypes.guess_type(image_path.name)[0]

            exif = {
                str(ExifTags.TAGS.get(tag_id, tag_id)): value
                for tag_id, value in image.getexif().items()
            }
            frame_count = int(getattr(image, "n_frames", 1))

            return ImageMetadata(
                path=image_path.resolve(),
                extension=image_path.suffix.lower(),
                mime_type=mime_type,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
                width=image.width,
                height=image.height,
                image_format=image_format,
                color_mode=image.mode,
                frame_count=frame_count,
                is_animated=bool(getattr(image, "is_animated", frame_count > 1)),
                exif=exif,
            )
