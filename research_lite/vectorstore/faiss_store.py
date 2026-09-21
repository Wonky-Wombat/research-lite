from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_lite.embedding import EmbeddedDocument


class FaissVectorStore:
    """Thin convenience wrapper around LangChain's FAISS vector store."""

    def __init__(self, index: FAISS) -> None:
        self._index = index

    @property
    def index(self) -> FAISS:
        return self._index

    @staticmethod
    def _as_text_embeddings(
        embedded_documents: Sequence[EmbeddedDocument],
    ) -> tuple[list[tuple[str, list[float]]], list[dict]]:
        text_embeddings: list[tuple[str, list[float]]] = []
        metadatas: list[dict] = []
        for embedded in embedded_documents:
            text_embeddings.append((embedded.document.page_content, list(embedded.vector)))
            metadata = dict(embedded.document.metadata) if embedded.document.metadata else {}
            metadatas.append(metadata)
        return text_embeddings, metadatas

    @staticmethod
    def _coerce_kwargs(kwargs: dict[str, object]) -> dict[str, Any]:
        if not kwargs:
            return {}
        return {key: cast(Any, value) for key, value in kwargs.items()}

    @classmethod
    def from_documents(
        cls,
        embedded_documents: Iterable[EmbeddedDocument],
        *,
        embedding_backend: Embeddings,
        **kwargs: object,
    ) -> FaissVectorStore:
        docs = list(embedded_documents)
        if not docs:
            msg = "Cannot build a FAISS index from an empty document collection."
            raise ValueError(msg)
        text_embeddings, metadatas = cls._as_text_embeddings(docs)
        extra_kwargs = cls._coerce_kwargs(kwargs)
        index = FAISS.from_embeddings(
            text_embeddings,
            embedding_backend,
            metadatas=metadatas,
            **extra_kwargs,
        )
        return cls(index)

    def add(self, embedded_documents: Iterable[EmbeddedDocument]) -> list[str]:
        docs = list(embedded_documents)
        if not docs:
            return []
        text_embeddings, metadatas = self._as_text_embeddings(docs)
        ids: list[str] = self._index.add_embeddings(text_embeddings, metadatas=metadatas)
        return ids

    def documents(self) -> list[Document]:
        """Return the documents persisted in this FAISS store in index order."""
        documents: list[Document] = []
        for index in sorted(self._index.index_to_docstore_id):
            document_id = self._index.index_to_docstore_id[index]
            document = self._index.docstore.search(document_id)
            if isinstance(document, Document):
                documents.append(document)
            else:
                msg = f"FAISS docstore entry {document_id!r} is not a document."
                raise ValueError(msg)
        return documents

    def similarity_search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 4,
        **kwargs: object,
    ) -> list[Document]:
        vector = list(query_vector)
        if not vector:
            return []
        extra_kwargs = self._coerce_kwargs(kwargs)
        docs: list[Document] = self._index.similarity_search_by_vector(vector, k=k, **extra_kwargs)
        return docs

    def similarity_search_with_score(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 4,
        **kwargs: object,
    ) -> list[tuple[Document, float]]:
        vector = list(query_vector)
        if not vector:
            return []
        extra_kwargs = self._coerce_kwargs(kwargs)
        scored: list[tuple[Document, float]] = self._index.similarity_search_with_score_by_vector(
            vector,
            k=k,
            **extra_kwargs,
        )
        return scored

    def save(self, path: Path, *, index_name: str = "index") -> None:
        self._index.save_local(str(path), index_name=index_name)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        embedding_backend: Embeddings,
        allow_dangerous_deserialization: bool = False,
        index_name: str = "index",
        **kwargs: object,
    ) -> FaissVectorStore:
        extra_kwargs = cls._coerce_kwargs(kwargs)
        index = FAISS.load_local(
            str(path),
            embedding_backend,
            index_name=index_name,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
            **extra_kwargs,
        )
        return cls(index)
