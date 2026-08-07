from __future__ import annotations

import json
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from PIL import Image

from filelore.embedding import (
    AudioEmbedding,
    ImageEmbedding,
)
from filelore.embedding.audio import ClapAudioEmbedding
from filelore.embedding.image import ClipImageEmbedding
from profiling.index_pipeline import ProfileConfiguration, run_profile
from profiling.instrumentation import validate_instrumentation_targets
from profiling.metrics import StageRecorder, aggregate_stages


class FakeDeviceTensor:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self.rows = rows

    def to(self, device: str) -> FakeDeviceTensor:
        return self


class FakeFeatureTensor:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self.rows = rows

    def detach(self) -> FakeFeatureTensor:
        return self

    def cpu(self) -> FakeFeatureTensor:
        return self

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self.rows]


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=8_000, max_length_s=1)

    def __call__(self, **kwargs: Any) -> dict[str, FakeDeviceTensor]:
        inputs = kwargs.get("images") or kwargs.get("audio")
        assert isinstance(inputs, list)
        rows = [(1.0, 0.0, 0.0) for _ in inputs]
        name = "pixel_values" if "images" in kwargs else "input_features"
        return {name: FakeDeviceTensor(rows)}


class FakeModel:
    def get_image_features(self, **inputs: FakeDeviceTensor) -> FakeFeatureTensor:
        return FakeFeatureTensor(next(iter(inputs.values())).rows)

    def get_audio_features(self, **inputs: FakeDeviceTensor) -> FakeFeatureTensor:
        return FakeFeatureTensor(next(iter(inputs.values())).rows)


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class ProfileImageEmbedding(ClipImageEmbedding):
    def __init__(self) -> None:
        self._torch = FakeTorch()
        self._processor = FakeProcessor()
        self._model = FakeModel()
        self.device = "cpu"
        self.batch_size = 2
        ImageEmbedding.__init__(
            self,
            model_id="profile-image",
            vector_name="profile_image",
            dimensions=3,
        )

class ProfileAudioEmbedding(ClapAudioEmbedding):
    sampling_rate = 8_000
    max_length_seconds = 1.0
    batch_size = 2

    def __init__(self) -> None:
        self._torch = FakeTorch()
        self._processor = FakeProcessor()
        self._model = FakeModel()
        self.device = "cpu"
        self.batch_size = 2
        self.sampling_rate = 8_000
        self.max_length_seconds = 1.0
        AudioEmbedding.__init__(
            self,
            model_id="profile-audio",
            vector_name="profile_audio",
            dimensions=3,
        )


def create_wave(path: Path, *, frames: int = 800) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\x00\x00" * frames)


def test_stage_recorder_tracks_nested_events_and_aggregates() -> None:
    recorder = StageRecorder()

    with recorder.span("parent", items=2):
        with recorder.span("child", items=1, input_bytes=10):
            pass

    parent = next(event for event in recorder.events if event.stage == "parent")
    child = next(event for event in recorder.events if event.stage == "child")
    aggregates = {item["stage"]: item for item in aggregate_stages(recorder.events)}

    assert child.parent_id == parent.event_id
    assert aggregates["parent"]["items"] == 2
    assert aggregates["child"]["input_bytes"] == 10


def test_profiler_targets_match_the_indexing_pipeline() -> None:
    validate_instrumentation_targets()


def test_profile_runs_real_image_and_audio_pipeline_and_writes_reports(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    audios = tmp_path / "audios"
    images.mkdir()
    audios.mkdir()
    Image.new("RGB", (8, 8), "red").save(images / "sample.jpg")
    create_wave(audios / "sample.wav")
    output = tmp_path / "results"

    result = run_profile(
        ProfileConfiguration(
            output_directory=output,
            image_directory=images,
            audio_directory=audios,
            batch_size=1,
            resource_sampling=False,
        ),
        image_embedding_factory=ProfileImageEmbedding,
        audio_embedding_factory=ProfileAudioEmbedding,
    )

    assert result.successful
    assert result.exit_codes == {"image": 0, "audio": 0}
    stages = {event.stage for event in result.events}
    assert {
        "planning.discovery",
        "planning.hash",
        "image.metadata",
        "image.decode_convert",
        "image.model_preprocessing",
        "image.gpu_forward",
        "image.embedding",
        "audio.metadata",
        "audio.segment_planning",
        "audio.decode_downmix_resample",
        "audio.model_preprocessing",
        "audio.gpu_forward",
        "audio.embedding",
        "storage.prepare_and_write",
        "storage.upsert",
    }.issubset(stages)
    assert (output / "summary.md").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "events.csv").is_file()
    assert (output / "resources.csv").is_file()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["exit_codes"] == {"image": 0, "audio": 0}
    assert summary["configuration"]["discovered_files"] == {
        "audio": 1,
        "image": 1,
    }
    assert summary["resources"] == {}
    assert {item["stage"] for item in summary["stages"]}.issuperset(stages)


def test_profile_refuses_nonempty_index_and_output_directories(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory"):
        run_profile(
            ProfileConfiguration(
                output_directory=output,
                image_directory=images,
                resource_sampling=False,
            ),
            image_embedding_factory=ProfileImageEmbedding,
        )
