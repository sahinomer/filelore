"""Metadata records and parsers for supported file types."""

from filelore.metadata.base import BaseMetadata, MetadataParser
from filelore.metadata.image import ImageMetadata, ImageMetadataParser

__all__ = [
    "BaseMetadata",
    "ImageMetadata",
    "ImageMetadataParser",
    "MetadataParser",
]
