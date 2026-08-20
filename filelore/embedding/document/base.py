"""Embedding contract for document passages and text queries."""

from filelore.embedding.text import TextEmbedding


class DocumentEmbedding(TextEmbedding[str]):
    """Embed document passages and queries into one retrieval vector space."""
