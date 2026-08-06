"""Contracts shared by image-text embedding models."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from PIL import Image

from filelore.embedding.text import TextEmbedding


ImageInput: TypeAlias = str | Path | Image.Image


class ImageEmbedding(TextEmbedding[ImageInput]):
    """Embed images and text into the same comparable vector space."""
