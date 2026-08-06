"""Hugging Face CLAP implementation for audio and text embeddings."""

from __future__ import annotations

import re
from typing import Any, Iterator, Sequence

from filelore.embedding.audio.base import AudioEmbedding, AudioInput
from filelore.embedding.base import EmbeddingVector


DEFAULT_CLAP_MODEL = "laion/larger_clap_general"
DEFAULT_CLAP_VECTOR_NAME = "audio_clap_laion_larger_general"


def _load_clap_backend(model_id: str) -> tuple[Any, Any, Any]:
    """Load optional ML dependencies only when a CLAP embedder is created."""
    try:
        import torch
        from transformers import AutoProcessor, ClapModel
    except ImportError as error:
        raise ImportError(
            "ClapAudioEmbedding requires the embedding dependencies; "
            "run 'uv sync --extra embedding' from the repository root"
        ) from error

    processor = AutoProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    return torch, processor, model


def _model_vector_name(model_id: str) -> str:
    if model_id == DEFAULT_CLAP_MODEL:
        return DEFAULT_CLAP_VECTOR_NAME
    normalized = re.sub(r"[^a-z0-9]+", "_", model_id.casefold()).strip("_")
    return f"audio_clap_{normalized}"


class ClapAudioEmbedding(AudioEmbedding):
    """Generate normalized CLAP vectors for decoded audio and text retrieval."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CLAP_MODEL,
        vector_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        torch_module, processor, model = _load_clap_backend(model_id)
        projection_dimensions = getattr(model.config, "projection_dim", None)
        if not isinstance(projection_dimensions, int):
            raise ValueError(f"CLAP model {model_id!r} has no projection dimension")

        feature_extractor = getattr(processor, "feature_extractor", None)
        sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if not isinstance(sampling_rate, int) or sampling_rate < 1:
            raise ValueError(f"CLAP model {model_id!r} has no valid sampling rate")
        max_length_seconds = getattr(feature_extractor, "max_length_s", None)
        if (
            not isinstance(max_length_seconds, (int, float))
            or max_length_seconds <= 0
        ):
            raise ValueError(f"CLAP model {model_id!r} has no maximum audio length")

        self._torch = torch_module
        self._processor = processor
        self._model = model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.max_length_seconds = float(max_length_seconds)
        self._model.to(self.device)
        self._model.eval()

        super().__init__(
            model_id=model_id,
            vector_name=vector_name or _model_vector_name(model_id),
            dimensions=projection_dimensions,
        )

    def predict_batch(
        self, items: Sequence[AudioInput]
    ) -> tuple[EmbeddingVector, ...]:
        if isinstance(items, AudioInput):
            raise TypeError("predict_batch expects a sequence of audio inputs")
        if not items:
            return ()

        prepared = tuple(self._prepare_audio(item) for item in items)
        vectors: list[EmbeddingVector] = []
        for batch in self._batches(prepared):
            inputs = self._processor(
                audio=[item.samples for item in batch],
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            rows = self._audio_features(self._move_inputs(inputs))
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

    def _prepare_audio(self, item: AudioInput) -> AudioInput:
        if not isinstance(item, AudioInput):
            raise TypeError("Every audio input must be an AudioInput")
        if item.sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Audio inputs must use the model sampling rate of "
                f"{self.sampling_rate} Hz"
            )
        if isinstance(item.samples, (str, bytes)):
            raise TypeError("Audio samples must be a one-dimensional numeric sequence")
        try:
            sample_count = len(item.samples)
        except TypeError as error:
            raise TypeError("Audio samples must be a sized sequence") from error
        if sample_count < 1:
            raise ValueError("Audio samples must not be empty")
        dimensions = getattr(item.samples, "ndim", 1)
        if dimensions != 1:
            raise ValueError("Audio samples must be mono and one-dimensional")
        first_sample = item.samples[0]
        if isinstance(first_sample, (str, bytes)):
            raise TypeError("Audio samples must contain numeric values")
        if hasattr(first_sample, "__len__"):
            raise ValueError("Audio samples must be mono and one-dimensional")
        return item

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

    def _audio_features(self, inputs: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._model.get_audio_features(**inputs)

    def _text_features(self, inputs: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._model.get_text_features(**inputs)

    @staticmethod
    def _feature_rows(features: Any) -> Sequence[Sequence[float]]:
        if not hasattr(features, "detach"):
            raise TypeError("CLAP feature output must be a tensor")
        return features.detach().cpu().tolist()

    def _batches(self, items: Sequence[Any]) -> Iterator[Sequence[Any]]:
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]
