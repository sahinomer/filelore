"""Hugging Face CLIP implementation for image and text embeddings."""

from __future__ import annotations

import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from filelore.embedding.base import EmbeddingVector
from filelore.embedding.image.base import ImageEmbedding, ImageInput


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_VECTOR_NAME = "image_clip_openai_vit_b32"


def _load_clip_backend(
    model_id: str, use_fast_processor: bool
) -> tuple[Any, Any, Any]:
    """Load optional ML dependencies only when a CLIP embedder is created."""
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except ImportError as error:
        raise ImportError(
            "ClipImageEmbedding requires the embedding dependencies; "
            "run 'uv sync --extra embedding' from the repository root"
        ) from error

    processor = AutoProcessor.from_pretrained(
        model_id, use_fast=use_fast_processor
    )
    model = CLIPModel.from_pretrained(model_id)
    return torch, processor, model


def _model_vector_name(model_id: str) -> str:
    if model_id == DEFAULT_CLIP_MODEL:
        return DEFAULT_CLIP_VECTOR_NAME
    normalized = re.sub(r"[^a-z0-9]+", "_", model_id.casefold()).strip("_")
    return f"image_clip_{normalized}"


class ClipImageEmbedding(ImageEmbedding):
    """Generate normalized CLIP vectors for image and text retrieval."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CLIP_MODEL,
        vector_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        use_fast_processor: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        torch_module, processor, model = _load_clip_backend(
            model_id, use_fast_processor
        )
        projection_dimensions = getattr(model.config, "projection_dim", None)
        if not isinstance(projection_dimensions, int):
            raise ValueError(f"CLIP model {model_id!r} has no projection dimension")

        self._torch = torch_module
        self._processor = processor
        self._model = model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self._model.to(self.device)
        self._model.eval()

        super().__init__(
            model_id=model_id,
            vector_name=vector_name or _model_vector_name(model_id),
            dimensions=projection_dimensions,
        )

    def predict_batch(
        self, items: Sequence[ImageInput]
    ) -> tuple[EmbeddingVector, ...]:
        if isinstance(items, (str, bytes, Path, Image.Image)):
            raise TypeError("predict_batch expects a sequence of image inputs")
        if not items:
            return ()

        vectors: list[EmbeddingVector] = []
        for batch in self._batches(items):
            with ExitStack() as stack:
                images = [self._prepare_image(item, stack) for item in batch]
                inputs = self._processor(images=images, return_tensors="pt")
                rows = self._image_features(self._move_inputs(inputs))
                vectors.extend(
                    self._prepare_vectors(
                        self._feature_rows(rows),
                        expected_count=len(batch),
                        normalize=True,
                    )
                )
        return tuple(vectors)

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, str):
            raise TypeError("predict_text_batch expects a sequence of strings")
        if not texts:
            return ()
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every text query must be a string")
        if any(not text.strip() for text in texts):
            raise ValueError("Text queries must not be empty")

        vectors: list[EmbeddingVector] = []
        for batch in self._batches(texts):
            inputs = self._processor(
                text=list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            rows = self._text_features(self._move_inputs(inputs))
            vectors.extend(
                self._prepare_vectors(
                    self._feature_rows(rows),
                    expected_count=len(batch),
                    normalize=True,
                )
            )
        return tuple(vectors)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(self._torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        return {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in dict(inputs).items()
        }

    def _image_features(self, inputs: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._model.get_image_features(**inputs)

    def _text_features(self, inputs: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._model.get_text_features(**inputs)

    @staticmethod
    def _feature_rows(features: Any) -> Sequence[Sequence[float]]:
        if not hasattr(features, "detach"):
            raise TypeError("CLIP feature output must be a tensor")
        return features.detach().cpu().tolist()

    def _batches(self, items: Sequence[Any]) -> Iterator[Sequence[Any]]:
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]

    @staticmethod
    def _prepare_image(item: ImageInput, stack: ExitStack) -> Image.Image:
        if isinstance(item, Image.Image):
            prepared = item.convert("RGB")
            stack.callback(prepared.close)
            return prepared
        if not isinstance(item, (str, Path)):
            raise TypeError("Image inputs must be paths or PIL images")

        opened = stack.enter_context(Image.open(Path(item).expanduser()))
        prepared = opened.convert("RGB")
        stack.callback(prepared.close)
        return prepared
