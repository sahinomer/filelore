from __future__ import annotations

from pathlib import Path
from typing import Sequence

from filelore.embedding import DocumentEmbedding, EmbeddingVector
from filelore.index import (
    FileIndexer,
    FileIndexRepository,
    file_point_id,
    file_segment_point_id,
)
from filelore.processors import DocumentProcessor
from filelore.storage import (
    DistanceMetric,
    MetadataCondition,
    MetadataFilter,
    VectorConfig,
)
from filelore.storage.qdrant import QdrantVectorDatabase


class KeywordDocumentEmbedding(DocumentEmbedding):
    def __init__(self) -> None:
        super().__init__(
            model_id="test-keyword-model",
            vector_name="text_test",
            dimensions=3,
        )

    def predict_batch(
        self, items: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(
            (1.0, 0.0, 0.0)
            if "train" in item.casefold()
            else (0.0, 1.0, 0.0)
            for item in items
        )

    def predict_text_batch(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(
            (1.0, 0.0, 0.0)
            if "rail" in text.casefold()
            else (0.0, 1.0, 0.0)
            for text in texts
        )


def test_file_indexer_stores_searchable_document_chunk_payloads(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "travel.md"
    document_path.write_text(
        "# Travel Guide\n\n"
        "## Rail Travel\n\n"
        "Regional trains connect the main cities.\n\n"
        "## Air Travel\n\n"
        "Direct flights serve the largest airports.\n",
        encoding="utf-8",
    )
    embedding = KeywordDocumentEmbedding()

    with QdrantVectorDatabase(tmp_path / "database") as database:
        repository = FileIndexRepository(
            database,
            segment_vector_configs={
                embedding.vector_name: VectorConfig(
                    embedding.dimensions,
                    distance=DistanceMetric.COSINE,
                )
            },
        )
        indexer = FileIndexer(
            repository,
            DocumentProcessor(embedding=embedding),
        )

        result = indexer.index_batch(tuple(indexer.discover(tmp_path)))

        assert result.failures == ()
        assert [entry.path for entry in result.entries] == [
            document_path.resolve()
        ]
        parent = database.retrieve(
            repository.collection_name,
            (file_point_id(document_path),),
        )[0]
        segment_count = int(parent.payload["segment_count"])
        chunks = database.retrieve(
            repository.segment_collection_name,
            tuple(
                file_segment_point_id(document_path, index)
                for index in range(segment_count)
            ),
            with_vectors=True,
        )

        assert parent.payload["file_type"] == "text"
        assert parent.payload["metadata"]["document_format"] == "markdown"
        assert segment_count == 3
        assert len(chunks) == 3
        assert all(
            chunk.payload["segment_kind"] == "document_chunk"
            for chunk in chunks
        )
        assert all("chunk_text" in chunk.payload for chunk in chunks)
        assert all(
            "segment_start_seconds" not in chunk.payload for chunk in chunks
        )
        assert [chunk.payload["chunk_index"] for chunk in chunks] == [0, 1, 2]

        matches = database.search(
            repository.segment_collection_name,
            embedding.predict_text("rail connections"),
            vector_name=embedding.vector_name,
            metadata_filter=MetadataFilter(
                all_of=(MetadataCondition("file_type", "text"),)
            ),
        )

        assert "Regional trains" in str(matches[0].record.payload["chunk_text"])
        assert matches[0].record.payload["heading"] == "Rail Travel"
        assert matches[0].record.payload["absolute_path"] == str(
            document_path.resolve()
        )

        search_results = repository.semantic_segment_search(
            embedding.predict_text("rail connections"),
            vector_name=embedding.vector_name,
            metadata_filter=MetadataFilter(
                all_of=(MetadataCondition("file_type", "text"),)
            ),
        )

        best_segment = search_results[0].segment
        assert best_segment is not None
        assert not best_segment.is_timed
        assert best_segment.kind == "document_chunk"
        assert best_segment.heading == "Rail Travel"
        assert best_segment.section_path == ("Travel Guide", "Rail Travel")
        assert best_segment.text is not None
        assert "Regional trains" in best_segment.text
