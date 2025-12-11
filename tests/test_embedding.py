from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_lite.embedding import EmbeddingConfig, EmbeddingService


class StubEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_embed_documents_batches_and_preserves_metadata() -> None:
    documents = [
        Document(page_content="first", metadata={"chunk_id": "1"}),
        Document(page_content="second", metadata={"chunk_id": "2"}),
        Document(page_content="third", metadata={"chunk_id": "3"}),
    ]
    backend = StubEmbeddings()
    service = EmbeddingService(backend, EmbeddingConfig(batch_size=2))

    embedded = service.embed_documents(documents)
    assert [len(call) for call in backend.calls] == [2, 1]
    assert [doc.vector for doc in embedded] == [[5.0], [6.0], [5.0]]
    assert [doc.document.metadata["chunk_id"] for doc in embedded] == ["1", "2", "3"]


def test_embed_documents_empty_input_returns_empty() -> None:
    backend = StubEmbeddings()
    service = EmbeddingService(backend)
    assert service.embed_documents([]) == []


def test_embed_documents_raises_on_mismatched_vectors() -> None:
    class BrokenEmbeddings(StubEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(texts[:-1])

    backend = BrokenEmbeddings()
    service = EmbeddingService(backend)
    docs = [Document(page_content="a"), Document(page_content="b")]

    with pytest.raises(ValueError, match="returned 1 vectors for 2 documents"):
        service.embed_documents(docs)


def test_embed_query_delegates_to_backend() -> None:
    backend = StubEmbeddings()
    service = EmbeddingService(backend)
    assert service.embed_query("hello") == [5.0]
