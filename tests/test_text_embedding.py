from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence

import pytest

from filelore.embedding import (
    DEFAULT_HARRIER_MODEL,
    DEFAULT_HARRIER_QUERY_PROMPT,
    DEFAULT_HARRIER_VECTOR_NAME,
    HarrierTextEmbedding,
    SentenceTransformerTextEmbedding,
    TextEmbedding,
)
from filelore.embedding.document import sentence_transformer


class FakeModel:
    def __init__(
        self,
        *,
        dimensions: int | None = 3,
        device: str = "cpu",
        rows: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.device = device
        self.rows = rows
        self.evaluation_mode = False
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimensions

    def eval(self) -> FakeModel:
        self.evaluation_mode = True
        return self

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append((texts, kwargs))
        if self.rows is not None:
            return [list(row) for row in self.rows]
        return [
            {
                "Public transport in the city": [1.0, 0.0, 0.0],
                "Museums in the capital": [0.0, 1.0, 0.0],
                "urban transportation": [1.0, 0.0, 0.0],
            }.get(text, [0.0, 0.0, 1.0])
            for text in texts
        ]


class FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def load(model_id: str, **kwargs: Any) -> tuple[FakeTorch, FakeModel]:
        calls.append((model_id, kwargs))
        return FakeTorch(), model

    monkeypatch.setattr(
        sentence_transformer,
        "_load_sentence_transformer_backend",
        load,
    )
    return calls


def test_sentence_transformer_backend_configures_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: Any) -> None:
            constructor_calls.append((model_id, kwargs))

    torch_module = ModuleType("torch")
    sentence_transformers_module = ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        sentence_transformers_module,
    )

    loaded_torch, loaded_model = (
        sentence_transformer._load_sentence_transformer_backend(
            "example/model",
            device="auto",
            model_kwargs={"dtype": "auto"},
            trust_remote_code=False,
        )
    )

    assert loaded_torch is torch_module
    assert isinstance(loaded_model, FakeSentenceTransformer)
    assert constructor_calls == [
        (
            "example/model",
            {
                "device": None,
                "model_kwargs": {"dtype": "auto"},
                "trust_remote_code": False,
            },
        )
    ]


def test_sentence_transformer_embeds_documents_and_prompted_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    backend_calls = _install_fake_backend(monkeypatch, model)
    embedding = SentenceTransformerTextEmbedding(
        model_id="Example/Multilingual Model",
        batch_size=2,
        query_prompt_name="search_query",
        document_prompt_name="search_document",
        model_kwargs={"dtype": "auto"},
    )

    document_vectors = embedding.predict_batch(
        ("Public transport in the city", "Museums in the capital")
    )
    query_vector = embedding.predict_text("urban transportation")

    assert isinstance(embedding, TextEmbedding)
    assert embedding.dimensions == 3
    assert embedding.device == "cpu"
    assert embedding.vector_name == (
        "text_sentence_transformer_example_multilingual_model"
    )
    assert model.evaluation_mode is True
    assert document_vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert query_vector == (1.0, 0.0, 0.0)
    assert backend_calls == [
        (
            "Example/Multilingual Model",
            {
                "device": "auto",
                "model_kwargs": {"dtype": "auto"},
                "trust_remote_code": False,
            },
        )
    ]
    assert model.encode_calls == [
        (
            ["Public transport in the city", "Museums in the capital"],
            {
                "batch_size": 2,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "prompt_name": "search_document",
            },
        ),
        (
            ["urban transportation"],
            {
                "batch_size": 2,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "prompt_name": "search_query",
            },
        ),
    ]


def test_sentence_transformer_omits_unconfigured_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    _install_fake_backend(monkeypatch, model)
    embedding = SentenceTransformerTextEmbedding(model_id="example/model")

    embedding.predict("Public transport in the city")
    embedding.predict_text("urban transportation")

    assert all("prompt_name" not in kwargs for _, kwargs in model.encode_calls)


def test_harrier_supplies_only_its_model_specific_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    backend_calls = _install_fake_backend(monkeypatch, model)
    embedding = HarrierTextEmbedding()

    embedding.predict("Public transport in the city")
    embedding.predict_text("urban transportation")

    assert embedding.model_id == DEFAULT_HARRIER_MODEL
    assert embedding.vector_name == DEFAULT_HARRIER_VECTOR_NAME
    assert embedding.query_prompt_name == DEFAULT_HARRIER_QUERY_PROMPT
    assert embedding.document_prompt_name is None
    assert backend_calls[0][1]["model_kwargs"] == {"dtype": "auto"}
    assert "prompt_name" not in model.encode_calls[0][1]
    assert model.encode_calls[1][1]["prompt_name"] == "web_search_query"


def test_harrier_allows_compatible_model_and_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(dimensions=1_024, device="cuda:1")
    backend_calls = _install_fake_backend(monkeypatch, model)
    embedding = HarrierTextEmbedding(
        model_id="microsoft/harrier-oss-v1-0.6b",
        vector_name="custom_text",
        device="cuda:1",
        batch_size=8,
        model_kwargs={},
    )

    assert embedding.dimensions == 1_024
    assert embedding.device == "cuda:1"
    assert embedding.vector_name == "custom_text"
    assert backend_calls[0][1] == {
        "device": "cuda:1",
        "model_kwargs": {},
        "trust_remote_code": False,
    }


def test_sentence_transformer_rejects_invalid_configuration_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_backend(monkeypatch, FakeModel())

    with pytest.raises(ValueError, match="model_id"):
        SentenceTransformerTextEmbedding(model_id=" ")
    with pytest.raises(ValueError, match="device"):
        SentenceTransformerTextEmbedding(model_id="example/model", device=" ")
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerTextEmbedding(model_id="example/model", batch_size=0)
    with pytest.raises(ValueError, match="query_prompt_name"):
        SentenceTransformerTextEmbedding(
            model_id="example/model", query_prompt_name=" "
        )

    embedding = SentenceTransformerTextEmbedding(model_id="example/model")
    with pytest.raises(TypeError, match="sequence"):
        embedding.predict_batch("document")
    with pytest.raises(TypeError, match="must be a string"):
        embedding.predict_batch((123,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        embedding.predict_text(" ")
    assert embedding.predict_batch(()) == ()
    assert embedding.predict_text_batch(()) == ()


@pytest.mark.parametrize("dimensions", [None, 0])
def test_sentence_transformer_requires_a_valid_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
    dimensions: int | None,
) -> None:
    _install_fake_backend(monkeypatch, FakeModel(dimensions=dimensions))

    with pytest.raises(ValueError, match="embedding dimension"):
        SentenceTransformerTextEmbedding(model_id="example/model")


def test_sentence_transformer_validates_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_backend(
        monkeypatch,
        FakeModel(rows=((1.0, 0.0),)),
    )
    embedding = SentenceTransformerTextEmbedding(model_id="example/model")

    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        embedding.predict("document")


def test_sentence_transformer_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    _install_fake_backend(monkeypatch, model)
    embedding = SentenceTransformerTextEmbedding(model_id="example/model")

    embedding.close()
    embedding.close()

    assert embedding._model is None
    with pytest.raises(RuntimeError, match="closed"):
        embedding.predict_text("query")
