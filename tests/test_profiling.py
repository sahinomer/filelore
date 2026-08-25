from __future__ import annotations

import json
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from filelore.embedding import (
    AudioEmbedding,
    DocumentEmbedding,
    ImageEmbedding,
)
from filelore.embedding.audio import ClapAudioEmbedding
from filelore.embedding.document import SentenceTransformerTextEmbedding
from filelore.embedding.image import ClipImageEmbedding
from profiling.index_pipeline import (
    ProfileConfiguration,
    _run_dataset,
    _validate_empty_qdrant_service,
    run_profile,
    validate_configuration,
)
from profiling.instrumentation import validate_instrumentation_targets
from profiling.metrics import (
    ResourceSample,
    StageRecorder,
    aggregate_stages,
    summarize_resources,
)


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

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        features = self.preprocess(texts)
        self.forward(features)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def preprocess(self, texts: list[str]) -> dict[str, Any]:
        return {
            "profile_document": {
                "input_ids": FakeDeviceTensor(
                    [(1.0, 0.0, 0.0) for _ in texts]
                )
            }
        }

    def forward(self, features: dict[str, Any]) -> dict[str, Any]:
        return features


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


class ProfileDocumentEmbedding(SentenceTransformerTextEmbedding):
    def __init__(self) -> None:
        self._torch = FakeTorch()
        self._model = FakeModel()
        self.device = "cpu"
        self.batch_size = 2
        self.query_prompt_name = None
        self.document_prompt_name = None
        self.model_kwargs = {}
        self.trust_remote_code = False
        DocumentEmbedding.__init__(
            self,
            model_id="profile-document",
            vector_name="profile_document",
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


def test_resource_summary_reports_average_p95_and_peak_usage() -> None:
    samples = tuple(
        ResourceSample(
            timestamp_ms=float(index),
            process_cpu_percent=process_cpu,
            system_cpu_percent=system_cpu,
            rss_bytes=rss_mb * 1024 * 1024,
            read_bytes=index * 1024 * 1024,
            write_bytes=index * 2 * 1024 * 1024,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_mb=gpu_memory,
            gpu_power_watts=gpu_power,
        )
        for index, (
            process_cpu,
            system_cpu,
            rss_mb,
            gpu_utilization,
            gpu_memory,
            gpu_power,
        ) in enumerate(
            (
                (10.0, 1.0, 100, 0.0, 500.0, 20.0),
                (20.0, 2.0, 200, 50.0, 600.0, 40.0),
                (30.0, 3.0, 300, 100.0, 700.0, 60.0),
            )
        )
    )

    summary = summarize_resources(samples)

    assert summary["process_cpu_max_percent"] == 30.0
    assert summary["system_cpu_p95_percent"] == 2.9
    assert summary["rss_average_mb"] == 200.0
    assert summary["rss_p95_mb"] == 290.0
    assert summary["peak_rss_mb"] == 300.0
    assert summary["gpu_memory_average_mb"] == 600.0
    assert summary["gpu_memory_p95_mb"] == 690.0
    assert summary["peak_gpu_memory_mb"] == 700.0
    assert summary["gpu_power_p95_watts"] == 58.0
    assert summary["gpu_power_max_watts"] == 60.0


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


def test_profile_runs_real_document_pipeline_and_writes_reports(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "sample.md").write_text(
        "# Profile document\n\nDocument indexing profile content.\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"

    result = run_profile(
        ProfileConfiguration(
            output_directory=output,
            document_directory=documents,
            batch_size=1,
            resource_sampling=False,
        ),
        document_embedding_factory=ProfileDocumentEmbedding,
    )

    assert result.successful
    assert result.exit_codes == {"text": 0}
    stages = {event.stage for event in result.events}
    assert {
        "planning.discovery",
        "planning.hash",
        "text.parse",
        "text.chunking",
        "text.processing",
        "text.model_encode",
        "text.model_preprocessing",
        "text.gpu_forward",
        "text.vector_postprocessing",
        "text.embedding",
        "storage.prepare_and_write",
        "storage.upsert",
    }.issubset(stages)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_codes"] == {"text": 0}
    assert summary["configuration"]["discovered_files"] == {"text": 1}
    assert summary["configuration"]["discovered_extensions"] == {".md": 1}
    assert {item["stage"] for item in summary["stages"]}.issuperset(stages)
    gpu_forward = next(
        item for item in summary["stages"] if item["stage"] == "text.gpu_forward"
    )
    assert gpu_forward["items"] == 1
    rendered = (output / "summary.md").read_text(encoding="utf-8")
    assert "Discovered documents: `1`" in rendered
    assert "Discovered extensions: `.md=1`" in rendered


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


def test_profile_refuses_local_path_with_service_url(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_configuration(
            ProfileConfiguration(
                output_directory=tmp_path / "results",
                image_directory=images,
                index_path=tmp_path / "index",
                qdrant_url="http://localhost:6333",
                resource_sampling=False,
            )
        )


def test_run_dataset_forwards_service_url() -> None:
    recorder = StageRecorder()

    with patch("profiling.index_pipeline.filelore_main", return_value=0) as main:
        exit_code = _run_dataset(
            Path("dataset"),
            "image",
            None,
            100,
            recorder,
            qdrant_url="http://localhost:6333",
            image_factory=ProfileImageEmbedding,
            audio_factory=ProfileAudioEmbedding,
            document_factory=ProfileDocumentEmbedding,
        )

    arguments = main.call_args.args[0]
    assert exit_code == 0
    assert arguments[arguments.index("--qdrant-url") + 1] == (
        "http://localhost:6333"
    )
    assert "--index-path" not in arguments


def test_profile_refuses_nonempty_service_collections() -> None:
    database = MagicMock()
    database.__enter__.return_value = database
    database.collection_exists.return_value = True
    database.count.side_effect = (3, 8)

    with (
        patch(
            "profiling.index_pipeline.QdrantVectorDatabase",
            return_value=database,
        ),
        pytest.raises(ValueError, match="files=3, files_segments=8"),
    ):
        _validate_empty_qdrant_service("http://localhost:6333")
