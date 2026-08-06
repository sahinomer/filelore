"""Audio metadata record and Mutagen-backed parser."""

from __future__ import annotations

import math
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Iterable

from mutagen import File as MutagenFile
from mutagen import MutagenError

from filelore.metadata.base import BaseMetadata, MetadataParser


_FORMAT_NAMES = {
    "AIFF": "AIFF",
    "EasyMP3": "MP3",
    "FLAC": "FLAC",
    "MP3": "MP3",
    "OggFLAC": "OGG FLAC",
    "OggOpus": "OPUS",
    "OggSpeex": "SPEEX",
    "OggVorbis": "OGG VORBIS",
    "WAVE": "WAV",
}


@dataclass(frozen=True, slots=True)
class AudioMetadata(BaseMetadata):
    """Metadata specific to sampled audio files."""

    file_type: ClassVar[str] = "audio"

    duration_seconds: float
    sample_rate_hz: int | None
    channels: int | None
    bitrate_bps: int | None
    bits_per_sample: int | None
    audio_format: str
    codec: str | None = None
    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)


class AudioMetadataParser(MetadataParser[AudioMetadata]):
    """Extract stream properties and normalized text tags with Mutagen."""

    supported_extensions = frozenset(
        {
            ".aif",
            ".aiff",
            ".flac",
            ".mp3",
            ".oga",
            ".ogg",
            ".opus",
            ".wav",
            ".wave",
        }
    )

    def parse(self, path: str | Path) -> AudioMetadata:
        audio_path = Path(path).expanduser()
        if not self.supports(audio_path):
            extension = audio_path.suffix or "<none>"
            raise ValueError(f"Unsupported audio extension: {extension}")
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        try:
            audio = MutagenFile(audio_path, easy=True)
        except MutagenError as error:
            raise ValueError(f"Could not read audio metadata: {audio_path}") from error
        if audio is None or getattr(audio, "info", None) is None:
            raise ValueError(f"Unrecognized or corrupt audio file: {audio_path}")

        info = audio.info
        duration_seconds = _duration(getattr(info, "length", None))
        stat = audio_path.stat()
        mime_type = _mime_type(audio, audio_path)

        return AudioMetadata(
            path=audio_path.resolve(),
            extension=audio_path.suffix.lower(),
            mime_type=mime_type,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
            duration_seconds=duration_seconds,
            sample_rate_hz=_positive_int(getattr(info, "sample_rate", None)),
            channels=_positive_int(getattr(info, "channels", None)),
            bitrate_bps=_positive_int(getattr(info, "bitrate", None)),
            bits_per_sample=_positive_int(
                getattr(info, "bits_per_sample", None)
            ),
            audio_format=_audio_format(audio),
            codec=_codec(info),
            tags=_normalize_tags(getattr(audio, "tags", None)),
        )


def _duration(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Audio metadata has no valid duration") from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Audio duration must be a finite non-negative value")
    return duration


def _positive_int(value: Any) -> int | None:
    try:
        prepared = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return prepared if prepared > 0 else None


def _mime_type(audio: Any, path: Path) -> str | None:
    mime_types = getattr(audio, "mime", None)
    if isinstance(mime_types, Iterable) and not isinstance(
        mime_types, (str, bytes)
    ):
        for mime_type in mime_types:
            if isinstance(mime_type, str) and mime_type:
                return mime_type
    return mimetypes.guess_type(path.name)[0]


def _audio_format(audio: Any) -> str:
    class_name = type(audio).__name__
    return _FORMAT_NAMES.get(class_name, class_name)


def _codec(info: Any) -> str | None:
    for attribute in (
        "codec",
        "codec_name",
        "codec_description",
        "compression",
        "encoder_info",
    ):
        value = getattr(info, attribute, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _normalize_tags(tags: Any) -> dict[str, tuple[str, ...]]:
    if tags is None or not hasattr(tags, "items"):
        return {}

    normalized: dict[str, tuple[str, ...]] = {}
    for key, raw_value in tags.items():
        values = tuple(_text_values(raw_value))
        if values:
            normalized[str(key).casefold()] = values
    return normalized


def _text_values(value: Any) -> Iterable[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        if isinstance(item, bytes):
            continue
        if isinstance(item, (str, int, float, bool)):
            text = str(item).strip()
            if text:
                yield text
