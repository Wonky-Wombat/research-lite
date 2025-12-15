from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from knowledge_lite.app import ingest_and_embed
from knowledge_lite.embedding import EmbeddingConfig, EmbeddingService
from knowledge_lite.preprocessing import SplitConfig


class LengthEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_ingest_and_embed_runs_pipeline(tmp_path: Path) -> None:
    file = tmp_path / "sample.txt"
    file.write_text("Hello KnowledgeLiteRAG", encoding="utf-8")

    service = EmbeddingService(LengthEmbeddings(), EmbeddingConfig(batch_size=4))
    embedded = ingest_and_embed(
        str(tmp_path),
        extensions=["txt"],
        split_config=SplitConfig(chunk_size=50, chunk_overlap=0),
        embedding_service=service,
    )

    assert len(embedded) == 1
    chunk = embedded[0]
    assert chunk.vector == [float(len(chunk.document.page_content))]
    metadata = chunk.document.metadata
    assert metadata["chunk_index"] == 0
    assert metadata["num_chunks"] == 1
    assert metadata["chunk_id"].endswith(":0")
