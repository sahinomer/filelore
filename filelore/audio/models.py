"""Value objects shared by audio pipeline components."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioRange:
    """A half-open time range within an audio file."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("Audio range start must be finite and non-negative")
        if (
            not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("Audio range end must be finite and after its start")

    @property
    def duration_seconds(self) -> float:
        """Return the length of this range in seconds."""
        return self.end_seconds - self.start_seconds
