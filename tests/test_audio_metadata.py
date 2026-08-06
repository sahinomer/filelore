from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from filelore.metadata import (
    AudioMetadata,
    AudioMetadataParser,
    BaseMetadata,
    MetadataParser,
)
from filelore.metadata import audio as audio_module


def create_wave(
    path: Path,
    *,
    duration_seconds: float = 0.25,
    sample_rate: int = 8_000,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count * channels)


def test_audio_parser_extracts_file_and_stream_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    create_wave(audio_path)

    parser = AudioMetadataParser()
    metadata = parser.parse(audio_path)

    assert isinstance(parser, MetadataParser)
    assert isinstance(metadata, AudioMetadata)
    assert isinstance(metadata, BaseMetadata)
    assert metadata.path == audio_path.resolve()
    assert metadata.extension == ".wav"
    assert metadata.mime_type in {"audio/wav", "audio/x-wav"}
    assert metadata.size_bytes == audio_path.stat().st_size
    assert metadata.duration_seconds == pytest.approx(0.25)
    assert metadata.sample_rate_hz == 8_000
    assert metadata.channels == 1
    assert metadata.bitrate_bps == 128_000
    assert metadata.bits_per_sample == 16
    assert metadata.audio_format == "WAV"
    assert metadata.tags == {}

    serialized = metadata.to_dict()
    assert serialized["path"] == str(audio_path.resolve())
    assert isinstance(serialized["modified_at"], str)
    json.dumps(serialized)


def test_audio_parser_normalizes_text_tags_and_ignores_binary_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "tagged.mp3"
    audio_path.write_bytes(b"placeholder")

    class EasyMP3:
        info = SimpleNamespace(
            length=12.5,
            sample_rate=44_100,
            channels=2,
            bitrate=192_000,
            encoder_info="LAME",
        )
        mime = ["audio/mpeg"]
        tags = {
            "TITLE": ["  Example Song  "],
            "artist": ["First Artist", "Second Artist"],
            "tracknumber": 3,
            "cover": b"binary artwork",
        }

    monkeypatch.setattr(
        audio_module,
        "MutagenFile",
        lambda _path, *, easy: EasyMP3(),
    )

    metadata = AudioMetadataParser().parse(audio_path)

    assert metadata.audio_format == "MP3"
    assert metadata.codec == "LAME"
    assert metadata.tags == {
        "title": ("Example Song",),
        "artist": ("First Artist", "Second Artist"),
        "tracknumber": ("3",),
    }
    assert metadata.to_dict()["tags"]["artist"] == [
        "First Artist",
        "Second Artist",
    ]


def test_audio_discover_filters_extensions_and_can_recurse(
    tmp_path: Path,
) -> None:
    top_audio = tmp_path / "top.WAV"
    nested_audio = tmp_path / "nested" / "child.flac"
    create_wave(top_audio)
    nested_audio.parent.mkdir()
    nested_audio.write_bytes(b"not parsed during discovery")
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

    parser = AudioMetadataParser()

    assert list(parser.discover(tmp_path, recursive=False)) == [top_audio]
    assert list(parser.discover(tmp_path)) == [nested_audio, top_audio]


def test_audio_parser_rejects_unsupported_and_corrupt_files(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "notes.txt"
    corrupt_path = tmp_path / "corrupt.wav"
    text_path.write_text("hello", encoding="utf-8")
    corrupt_path.write_text("not audio", encoding="utf-8")

    parser = AudioMetadataParser()
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        parser.parse(text_path)
    with pytest.raises(ValueError, match="audio|Audio"):
        parser.parse(corrupt_path)


def test_audio_parser_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AudioMetadataParser().parse(tmp_path / "missing.wav")
