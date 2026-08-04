from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence

import pytest
from PIL import Image

from filelore.embedding import BaseEmbedding, ClipImageEmbedding, EmbeddingVector
from filelore.embedding.image import clip


class EchoEmbedding(BaseEmbedding[str]):
    def __init__(self) -> None:
        super().__init__(model_id="echo", vector_name="echo", dimensions=2)
        self.batches: list[tuple[str, ...]] = []

    def predict_batch(self, items: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.batches.append(tuple(items))
        vectors = [(float(len(item)), 1.0) for item in items]
        return self._prepare_vectors(vectors, expected_count=len(items))


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
    def __init__(self) -> None:
        self.image_batches: list[list[str]] = []
        self.text_batches: list[list[str]] = []

    def __call__(self, **kwargs: object) -> dict[str, FakeDeviceTensor]:
        if "images" in kwargs:
            images = kwargs["images"]
            assert isinstance(images, list)
            self.image_batches.append([image.mode for image in images])
            rows = [
                tuple(float(value) for value in image.getpixel((0, 0)))
                for image in images
            ]
            return {"pixel_values": FakeDeviceTensor(rows)}

        texts = kwargs["text"]
        assert isinstance(texts, list)
        self.text_batches.append(texts)
        rows = [
            {
                "red": (1.0, 0.0, 0.0),
                "green": (0.0, 1.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
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

    def get_image_features(
        self, *, pixel_values: FakeDeviceTensor
    ) -> FakeFeatureTensor:
        return FakeFeatureTensor(pixel_values.rows)

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


@pytest.mark.parametrize("use_fast_processor", [True, False])
def test_clip_backend_configures_processor(
    monkeypatch: pytest.MonkeyPatch,
    use_fast_processor: bool,
) -> None:
    processor = object()
    model = object()
    processor_calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            processor_calls.append((model_id, kwargs))
            return processor

    class FakeClipModel:
        @staticmethod
        def from_pretrained(model_id: str) -> object:
            assert model_id == "example/clip"
            return model

    torch_module = ModuleType("torch")
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.CLIPModel = FakeClipModel
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        transformers_module,
    )

    loaded_torch, loaded_processor, loaded_model = clip._load_clip_backend(
        "example/clip", use_fast_processor
    )

    assert loaded_torch is torch_module
    assert loaded_processor is processor
    assert loaded_model is model
    assert processor_calls == [
        ("example/clip", {"use_fast": use_fast_processor})
    ]


def test_base_embedding_single_prediction_uses_batch_path() -> None:
    embedding = EchoEmbedding()

    assert embedding.predict("hello") == (5.0, 1.0)
    assert embedding.batches == [("hello",)]


def test_base_embedding_rejects_invalid_model_vectors() -> None:
    embedding = EchoEmbedding()

    with pytest.raises(ValueError, match="dimensions"):
        embedding._prepare_vectors([(1.0,)], expected_count=1)
    with pytest.raises(ValueError, match="finite"):
        embedding._prepare_vectors([(float("nan"), 1.0)], expected_count=1)
    with pytest.raises(ValueError, match="zero-length"):
        embedding._prepare_vectors([(0.0, 0.0)], expected_count=1, normalize=True)


def test_clip_embeds_single_and_batch_images_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = FakeProcessor()
    model = FakeModel()

    def load_backend(
        model_id: str, use_fast_processor: bool
    ) -> tuple[FakeTorch, FakeProcessor, FakeModel]:
        assert model_id == "openai/clip-vit-base-patch32"
        assert use_fast_processor is False
        return FakeTorch(), processor, model

    monkeypatch.setattr(
        clip,
        "_load_clip_backend",
        load_backend,
    )
    red_path = tmp_path / "red.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 100)).save(red_path)
    blue_image = Image.new("RGB", (2, 2), (0, 0, 255))

    embedding = ClipImageEmbedding(batch_size=1, use_fast_processor=False)
    image_vectors = embedding.predict_batch((red_path, blue_image))
    text_vector = embedding.predict_text("red")
    text_vectors = embedding.predict_text_batch(("green", "blue"))

    assert embedding.dimensions == 3
    assert embedding.vector_name == "image_clip_openai_vit_b32"
    assert embedding.device == "cpu"
    assert model.device == "cpu"
    assert model.evaluation_mode is True
    assert image_vectors == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert text_vector == (1.0, 0.0, 0.0)
    assert text_vectors == ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert processor.image_batches == [["RGB"], ["RGB"]]
    assert processor.text_batches == [["red"], ["green"], ["blue"]]
    assert blue_image.getpixel((0, 0)) == (0, 0, 255)
    blue_image.close()


def test_clip_rejects_ambiguous_batches_and_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clip,
        "_load_clip_backend",
        lambda _model_id, _use_fast: (
            FakeTorch(),
            FakeProcessor(),
            FakeModel(),
        ),
    )
    embedding = ClipImageEmbedding()

    with pytest.raises(TypeError, match="sequence"):
        embedding.predict_batch("image.png")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        embedding.predict_text_batch("query")
    with pytest.raises(ValueError, match="must not be empty"):
        embedding.predict_text(" ")
    assert embedding.predict_batch(()) == ()
    assert embedding.predict_text_batch(()) == ()
