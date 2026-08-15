"""Metadata records and parsers for supported file types."""

from filelore.metadata.audio import AudioMetadata, AudioMetadataParser
from filelore.metadata.base import BaseMetadata, MetadataParser
from filelore.metadata.document import DocumentFormat, DocumentMetadata
from filelore.metadata.image import ImageMetadata, ImageMetadataParser

__all__ = [
    "AudioMetadata",
    "AudioMetadataParser",
    "BaseMetadata",
    "DocumentFormat",
    "DocumentMetadata",
    "ImageMetadata",
    "ImageMetadataParser",
    "MetadataParser",
]
