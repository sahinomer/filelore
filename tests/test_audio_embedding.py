from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Sequence

import pytest

from filelore.embedding import (
    AudioEmbedding,
    AudioInput,
    ClapAudioEmbedding,
    EmbeddingVector,
    TextEmbedding,
)
from filelore.embedding.audio import clap


class FakeDeviceTensor:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self.rows = rows
        self.device: str | None = None

    def to(self, device: str) -> FakeDeviceTensor:
        self.device = device
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
    feature_extractor = SimpleNamespace(
        sampling_rate=48_000,
        max_length_s=10,
    )

    def __init__(self) -> None:
        self.audio_batches: list[list[Sequence[float]]] = []
        self.audio_arguments: list[dict[str, object]] = []
        self.text_batches: list[list[str]] = []

    def __call__(self, **kwargs: object) -> dict[str, FakeDeviceTensor]:
        if "audio" in kwargs:
            audio_items = kwargs["audio"]
            assert isinstance(audio_items, list)
            self.audio_batches.append(audio_items)
            self.audio_arguments.append(kwargs)
            rows = [tuple(float(value) for value in item[:3]) for item in audio_items]
            return {"input_features": FakeDeviceTensor(rows)}

        texts = kwargs["text"]
        assert isinstance(texts, list)
        self.text_batches.append(texts)
        rows = [
            {
                "dog barking": (1.0, 0.0, 0.0),
                "rain": (0.0, 1.0, 0.0),
                "thunder": (0.0, 0.0, 1.0),
            }.get(text, (1.0, 1.0, 1.0))
            for text in texts
        ]
        return {"input_ids": FakeDeviceTensor(rows)}


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(projection_dim=3)
        self.device: str | None = None
        self.evaluation_mode = False

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.evaluation_mode = True
        return self

    def get_audio_features(
        self, *, input_features: FakeDeviceTensor
    ) -> FakeFeatureTensor:
        return FakeFeatureTensor(input_features.rows)

    def get_text_features(
        self, *, input_ids: FakeDeviceTensor
    ) -> FakeFeatureTensor:
        return FakeFeatureTensor(input_ids.rows)


class FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


def test_clap_backend_configures_processor_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = object()
    model = object()
    processor_calls: list[str] = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str) -> object:
            processor_calls.append(model_id)
            return processor

    class FakeClapModel:
        @staticmethod
        def from_pretrained(model_id: str) -> object:
            assert model_id == "example/clap"
            return model

    torch_module = ModuleType("torch")
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.ClapModel = FakeClapModel
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    loaded_torch, loaded_processor, loaded_model = clap._load_clap_backend(
        "example/clap"
    )

    assert loaded_torch is torch_module
    assert loaded_processor is processor
    assert loaded_model is model
    assert processor_calls == ["example/clap"]


def test_clap_embeds_audio_and_text_in_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor()
    model = FakeModel()
    monkeypatch.setattr(
        clap,
        "_load_clap_backend",
        lambda model_id: (FakeTorch(), processor, model),
    )

    embedding = ClapAudioEmbedding(batch_size=1)
    audio_vectors = embedding.predict_batch(
        (
            AudioInput((3.0, 0.0, 0.0), 48_000),
            AudioInput((0.0, 0.0, 5.0), 48_000),
        )
    )
    text_vector = embedding.predict_text("dog barking")
    text_vectors = embedding.predict_text_batch(("rain", "thunder"))

    assert isinstance(embedding, AudioEmbedding)
    assert isinstance(embedding, TextEmbedding)
    assert embedding.dimensions == 3
    assert embedding.vector_name == "audio_clap_laion_larger_general"
    assert embedding.sampling_rate == 48_000
    assert embedding.max_length_seconds == 10.0
    assert embedding.device == "cpu"
    assert model.device == "cpu"
    assert model.evaluation_mode is True
    assert audio_vectors == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert text_vector == (1.0, 0.0, 0.0)
    assert text_vectors == ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert [len(batch) for batch in processor.audio_batches] == [1, 1]
    assert processor.text_batches == [["dog barking"], ["rain"], ["thunder"]]
    assert all(
        arguments["sampling_rate"] == 48_000
        for arguments in processor.audio_arguments
    )


def test_clap_supports_custom_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clap,
        "_load_clap_backend",
        lambda model_id: (FakeTorch(), FakeProcessor(), FakeModel()),
    )

    embedding = ClapAudioEmbedding(model_id="Example/My CLAP")
    overridden = ClapAudioEmbedding(
        model_id="Example/My CLAP",
        vector_name="custom_audio",
    )

    assert embedding.vector_name == "audio_clap_example_my_clap"
    assert overridden.vector_name == "custom_audio"


def test_clap_close_releases_model_and_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clap,
        "_load_clap_backend",
        lambda model_id: (FakeTorch(), FakeProcessor(), FakeModel()),
    )
    embedding = ClapAudioEmbedding()

    embedding.close()
    embedding.close()

    assert embedding._model is None
    assert embedding._processor is None


def test_clap_allows_unchunked_audio_longer_than_the_model_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor()
    monkeypatch.setattr(
        clap,
        "_load_clap_backend",
        lambda model_id: (FakeTorch(), processor, FakeModel()),
    )
    embedding = ClapAudioEmbedding()
    samples = (1.0, 0.0, 0.0) + (0.0,) * 480_000

    vector = embedding.predict(AudioInput(samples, 48_000))

    assert vector == (1.0, 0.0, 0.0)
    assert len(processor.audio_batches[0][0]) == 480_003


def test_clap_rejects_invalid_audio_and_text_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clap,
        "_load_clap_backend",
        lambda model_id: (FakeTorch(), FakeProcessor(), FakeModel()),
    )
    embedding = ClapAudioEmbedding()

    with pytest.raises(TypeError, match="sequence"):
        embedding.predict_batch(  # type: ignore[arg-type]
            AudioInput((1.0, 0.0, 0.0), 48_000)
        )
    with pytest.raises(TypeError, match="AudioInput"):
        embedding.predict_batch(((1.0, 0.0, 0.0),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        embedding.predict(AudioInput((), 48_000))
    with pytest.raises(ValueError, match="mono"):
        embedding.predict(  # type: ignore[arg-type]
            AudioInput(((1.0, 0.0), (0.0, 1.0)), 48_000)
        )
    with pytest.raises(ValueError, match="48000 Hz"):
        embedding.predict(AudioInput((1.0, 0.0, 0.0), 44_100))
    with pytest.raises(TypeError, match="sequence"):
        embedding.predict_text_batch("query")
    with pytest.raises(ValueError, match="must not be empty"):
        embedding.predict_text(" ")
    assert embedding.predict_batch(()) == ()
    assert embedding.predict_text_batch(()) == ()
