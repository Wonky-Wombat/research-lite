#
# __init__.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass(slots=True)
class EmbeddedDocument:
    """Pair a document with its embedding vector."""

    document: Document
    vector: list[float]


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for batching behaviour of the embedding service."""

    batch_size: int = 32


class EmbeddingService:
    """Batch documents through a langchain Embeddings backend."""

    def __init__(
        self, embedding_backend: Embeddings, config: EmbeddingConfig | None = None
    ) -> None:
        self._embedding_backend = embedding_backend
        self._config = config or EmbeddingConfig()

    @property
    def backend(self) -> Embeddings:
        return self._embedding_backend

    def embed_documents(self, documents: Iterable[Document]) -> list[EmbeddedDocument]:
        docs = [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in documents
        ]
        if not docs:
            return []

        embedded: list[EmbeddedDocument] = []
        batch_size = max(1, self._config.batch_size)
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            vectors = self._embedding_backend.embed_documents([doc.page_content for doc in batch])
            if len(vectors) != len(batch):
                raise ValueError(
                    f"Embedding backend returned {len(vectors)} vectors for {len(batch)} documents"
                )
            for doc, vector in zip(batch, vectors, strict=True):
                embedded.append(EmbeddedDocument(document=doc, vector=list(vector)))

        return embedded

    def embed_query(self, query: str) -> list[float]:
        return list(self._embedding_backend.embed_query(query))


__all__ = ["EmbeddedDocument", "EmbeddingConfig", "EmbeddingService"]
