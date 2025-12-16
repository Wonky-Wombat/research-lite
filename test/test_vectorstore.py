from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_lite.embedding import EmbeddedDocument
from knowledge_lite.vectorstore import FaissVectorStore

pytest.importorskip("faiss")


class StubEmbeddings(Embeddings):
    """Simple deterministic embedding backend for tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def make_embedded_docs() -> list[EmbeddedDocument]:
    doc1 = Document(page_content="knowledgelite", metadata={"id": "doc1"})
    doc2 = Document(page_content="rag", metadata={"id": "doc2"})
    return [
        EmbeddedDocument(doc1, [1.0, 0.0]),
        EmbeddedDocument(doc2, [0.0, 1.0]),
    ]


def test_from_documents_indexes_data_and_supports_similarity_search() -> None:
    backend = StubEmbeddings()
    store = FaissVectorStore.from_documents(make_embedded_docs(), embedding_backend=backend)

    results = store.similarity_search([1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0].metadata["id"] == "doc1"

    scored = store.similarity_search_with_score([0.0, 1.0], k=1)
    assert len(scored) == 1
    doc, score = scored[0]
    assert doc.metadata["id"] == "doc2"
    assert isinstance(score, float)


def test_add_appends_documents_to_existing_index() -> None:
    backend = StubEmbeddings()
    initial = make_embedded_docs()
    store = FaissVectorStore.from_documents(initial[:1], embedding_backend=backend)

    new_doc = EmbeddedDocument(
        Document(page_content="charlie", metadata={"id": "doc3"}), [0.5, 0.5]
    )
    store.add([initial[1], new_doc])

    ids = sorted(doc.metadata["id"] for doc in store.similarity_search([0.5, 0.5], k=3))
    assert ids == ["doc1", "doc2", "doc3"]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    backend = StubEmbeddings()
    store = FaissVectorStore.from_documents(make_embedded_docs(), embedding_backend=backend)

    store.save(tmp_path, index_name="demo")
    reloaded = FaissVectorStore.load(
        tmp_path,
        embedding_backend=backend,
        index_name="demo",
        allow_dangerous_deserialization=True,
    )

    docs = reloaded.similarity_search([1.0, 0.0], k=1)
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "doc1"


def test_similarity_search_returns_empty_for_empty_vector() -> None:
    backend = StubEmbeddings()
    store = FaissVectorStore.from_documents(make_embedded_docs(), embedding_backend=backend)
    assert store.similarity_search([], k=1) == []
