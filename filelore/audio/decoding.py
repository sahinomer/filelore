"""Bounded audio decoding and resampling services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from filelore.audio.models import AudioRange
from filelore.embedding import AudioInput


class AudioDecoder(Protocol):
    """Decode one time range into model-ready mono samples."""

    def decode(
        self,
        path: str | Path,
        audio_range: AudioRange,
        *,
        target_sampling_rate: int,
    ) -> AudioInput: ...


def _load_audio_backend() -> tuple[Any, Any, Any]:
    try:
        import numpy
        import soundfile
        import soxr
    except ImportError as error:
        raise ImportError(
            "Audio decoding requires numpy, soundfile, and soxr"
        ) from error
    return numpy, soundfile, soxr


class SoundFileAudioDecoder:
    """Read bounded ranges through libsndfile and resample them with SoXR."""

    def __init__(self) -> None:
        self._numpy, self._soundfile, self._soxr = _load_audio_backend()

    def decode(
        self,
        path: str | Path,
        audio_range: AudioRange,
        *,
        target_sampling_rate: int,
    ) -> AudioInput:
        if target_sampling_rate < 1:
            raise ValueError("Target sampling rate must be positive")

        with self._soundfile.SoundFile(Path(path)) as audio_file:
            source_sampling_rate = int(audio_file.samplerate)
            start_frame = min(
                round(audio_range.start_seconds * source_sampling_rate),
                len(audio_file),
            )
            end_frame = min(
                round(audio_range.end_seconds * source_sampling_rate),
                len(audio_file),
            )
            if end_frame <= start_frame:
                raise ValueError("Audio range contains no decodable samples")
            audio_file.seek(start_frame)
            samples = audio_file.read(
                end_frame - start_frame,
                dtype="float32",
                always_2d=True,
            )

        mono = self._numpy.mean(samples, axis=1, dtype=self._numpy.float32)
        if source_sampling_rate != target_sampling_rate:
            mono = self._soxr.resample(
                mono,
                source_sampling_rate,
                target_sampling_rate,
                quality="HQ",
            )
        mono = self._numpy.asarray(mono, dtype=self._numpy.float32)
        if mono.size < 1:
            raise ValueError("Audio range contains no decodable samples")
        return AudioInput(samples=mono, sampling_rate=target_sampling_rate)
