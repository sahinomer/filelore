"""Default file-query adapter registry."""

from filelore.search.audio import AudioFileQueryVectorizer
from filelore.search.image import ImageFileQueryVectorizer
from filelore.search.protocols import FileQueryVectorizer


def default_file_query_vectorizers() -> dict[str, FileQueryVectorizer]:
    """Return the built-in file-query adapter for each supported target."""
    return {
        "image": ImageFileQueryVectorizer(),
        "audio": AudioFileQueryVectorizer(),
    }
