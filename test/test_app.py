from __future__ import annotations

import importlib.util
from pathlib import Path

from langchain_core.embeddings import Embeddings

from research_lite.app import ingest_and_embed
from research_lite.embedding import EmbeddingConfig, EmbeddingService
from research_lite.preprocessing import SplitConfig


class LengthEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_ingest_and_embed_runs_pipeline(tmp_path: Path) -> None:
    file = tmp_path / "sample.txt"
    file.write_text("Hello KnowledgeLiteRAG", encoding="utf-8")

    embedding_service = EmbeddingService(LengthEmbeddings(), EmbeddingConfig(batch_size=4))
    embedded, vector_store, service = ingest_and_embed(
        str(tmp_path),
        extensions=["txt"],
        split_config=SplitConfig(chunk_size=50, chunk_overlap=0),
        embedding_service=embedding_service,
    )

    assert len(embedded) == 1
    chunk = embedded[0]
    assert chunk.vector == [float(len(chunk.document.page_content))]
    metadata = chunk.document.metadata
    assert metadata["chunk_index"] == 0
    assert metadata["num_chunks"] == 1
    assert metadata["chunk_id"].endswith(":0")
    if importlib.util.find_spec("faiss") is not None:
        assert vector_store is not None
    else:
        assert vector_store is None
    assert service is embedding_service
