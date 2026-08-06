from __future__ import annotations

from pathlib import Path

import pytest

from filelore.index import IndexCoordinator, IndexHandler


def handler(file_type: str, *extensions: str) -> IndexHandler:
    def unexpected_model_load() -> object:
        raise AssertionError("Discovery must not load embedding models")

    return IndexHandler(
        file_type=file_type,
        extensions=frozenset(extensions),
        embedding_factory=unexpected_model_load,  # type: ignore[arg-type]
        processor_factory=lambda embedding: None,  # type: ignore[arg-type]
        vector_scope="file",
    )


def test_coordinator_discovers_once_and_groups_paths_by_registered_type(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.PNG"
    audio_path = tmp_path / "nested" / "effect.wav"
    ignored_path = tmp_path / "notes.txt"
    image_path.write_bytes(b"image placeholder")
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"audio placeholder")
    ignored_path.write_text("ignored", encoding="utf-8")
    coordinator = IndexCoordinator(
        (handler("image", ".png", ".jpg"), handler("audio", ".wav"))
    )

    plan = coordinator.discover(tmp_path)

    assert plan.total_files == 2
    assert [queue.file_type for queue in plan.queues] == ["image", "audio"]
    assert plan.queues[0].paths == (image_path,)
    assert plan.queues[1].paths == (audio_path,)


def test_coordinator_type_filter_skips_other_queues(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    audio_path = tmp_path / "effect.wav"
    image_path.write_bytes(b"image placeholder")
    audio_path.write_bytes(b"audio placeholder")
    coordinator = IndexCoordinator(
        (handler("image", "png"), handler("audio", "wav"))
    )

    plan = coordinator.discover(tmp_path, allowed_types=("audio",))

    assert [queue.file_type for queue in plan.queues] == ["audio"]
    assert plan.queues[0].paths == (audio_path,)


def test_coordinator_rejects_ambiguous_extension_registration() -> None:
    with pytest.raises(ValueError, match="registered for both"):
        IndexCoordinator(
            (handler("first", ".wav"), handler("second", "WAV"))
        )
