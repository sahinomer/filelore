from __future__ import annotations

from pathlib import Path

import pytest

from filelore.storage import QdrantVectorDatabase
from filelore.storage import qdrant as qdrant_module


class FakeQdrantClient:
    instances: list[FakeQdrantClient] = []

    def __init__(self, **configuration: object) -> None:
        self.configuration = configuration
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def test_qdrant_url_takes_precedence_over_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qdrant_module, "QdrantClient", FakeQdrantClient)
    local_path = tmp_path / "unused-local-index"

    database = QdrantVectorDatabase(
        local_path,
        url="http://qdrant.test:6333",
    )

    assert database.url == "http://qdrant.test:6333"
    assert database.path is None
    assert not local_path.exists()
    assert database._client.configuration == {
        "url": "http://qdrant.test:6333"
    }
    database.close()
    assert database._client.closed is True


def test_qdrant_preconfigured_client_is_exclusive(tmp_path: Path) -> None:
    client = FakeQdrantClient()

    with pytest.raises(ValueError, match="cannot be combined"):
        QdrantVectorDatabase(
            tmp_path / "index",
            url="http://qdrant.test:6333",
            client=client,  # type: ignore[arg-type]
        )


def test_qdrant_connection_requires_a_source() -> None:
    with pytest.raises(ValueError, match="Provide at least one"):
        QdrantVectorDatabase()
