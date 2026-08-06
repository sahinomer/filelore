from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_coordinator_hashes_and_classifies_incremental_work(
    tmp_path: Path,
) -> None:
    unchanged_path = tmp_path / "photo.png"
    updated_path = tmp_path / "effect.wav"
    new_path = tmp_path / "new.mp3"
    for path in (unchanged_path, updated_path, new_path):
        path.write_bytes(path.name.encode())
    coordinator = IndexCoordinator(
        (handler("image", ".png"), handler("audio", ".wav", ".mp3"))
    )
    plan = coordinator.discover(tmp_path)
    existing_hashes = {
        unchanged_path: "same",
        updated_path: "old",
    }

    class Repository:
        def get_by_paths(self, paths: tuple[Path, ...]) -> tuple[object, ...]:
            return tuple(
                SimpleNamespace(content_hash=existing_hashes[path])
                if path in existing_hashes
                else None
                for path in paths
            )

    current_hashes = {
        unchanged_path: "same",
        updated_path: "changed",
        new_path: "new",
    }
    progress: list[int] = []

    work_plan = coordinator.classify_changes(
        plan,
        Repository(),  # type: ignore[arg-type]
        hash_file=current_hashes.__getitem__,
        on_progress=progress.append,
    )

    image_queue, audio_queue = work_plan.queues
    assert image_queue.discovered_count == 1
    assert image_queue.unchanged_count == 1
    assert image_queue.candidates == ()
    assert audio_queue.new_count == 1
    assert audio_queue.updated_count == 1
    assert [candidate.path for candidate in audio_queue.candidates] == [
        updated_path,
        new_path,
    ]
    assert [candidate.change for candidate in audio_queue.candidates] == [
        "updated",
        "new",
    ]
    assert progress == [1, 1, 1]
