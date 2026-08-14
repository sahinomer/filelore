"""Shared audio decoding, segmentation, and vectorization services."""

from filelore.audio.decoding import AudioDecoder, SoundFileAudioDecoder
from filelore.audio.models import AudioRange
from filelore.audio.segmenting import (
    AudioSegmenter,
    SlidingWindowChunker,
    WholeFileSegmenter,
)
from filelore.audio.vectorization import (
    AudioChunkVectorizer,
    AudioVectorizationBatch,
    AudioVectorizationFailure,
    AudioVectorizationSource,
    AudioVectorizedFile,
    AudioVectorizedSegment,
)

__all__ = [
    "AudioChunkVectorizer",
    "AudioDecoder",
    "AudioRange",
    "AudioSegmenter",
    "AudioVectorizationBatch",
    "AudioVectorizationFailure",
    "AudioVectorizationSource",
    "AudioVectorizedFile",
    "AudioVectorizedSegment",
    "SlidingWindowChunker",
    "SoundFileAudioDecoder",
    "WholeFileSegmenter",
]
