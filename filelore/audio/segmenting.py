"""Interchangeable strategies for selecting audio embedding ranges."""

from __future__ import annotations

from typing import Protocol

from filelore.audio.models import AudioRange
from filelore.metadata import AudioMetadata


class AudioSegmenter(Protocol):
    """Select the ranges of an audio file that should be embedded."""

    def segments(self, metadata: AudioMetadata) -> tuple[AudioRange, ...]: ...


class SlidingWindowChunker:
    """Split audio into overlapping, fixed-size windows."""

    def __init__(self, *, window_seconds: float, hop_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("Audio chunk window must be positive")
        if hop_seconds <= 0:
            raise ValueError("Audio chunk hop must be positive")
        if hop_seconds > window_seconds:
            raise ValueError("Audio chunk hop must not exceed its window")
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)

    def segments(self, metadata: AudioMetadata) -> tuple[AudioRange, ...]:
        duration = metadata.duration_seconds
        if duration <= 0:
            return ()

        ranges: list[AudioRange] = []
        start = 0.0
        while start < duration:
            end = min(start + self.window_seconds, duration)
            ranges.append(AudioRange(start_seconds=start, end_seconds=end))
            if end >= duration:
                break
            start += self.hop_seconds
        return tuple(ranges)


class WholeFileSegmenter:
    """Treat the entire audio file as one embedding input."""

    def segments(self, metadata: AudioMetadata) -> tuple[AudioRange, ...]:
        if metadata.duration_seconds <= 0:
            return ()
        return (
            AudioRange(
                start_seconds=0.0,
                end_seconds=metadata.duration_seconds,
            ),
        )
