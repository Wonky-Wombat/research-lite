"""Reusable, read-only query sessions for persisted ResearchLite libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from research_lite.embedding import EmbeddingService
from research_lite.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    RerankerUnavailableError,
    RetrievalMode,
)
from research_lite.vectorstore import FaissVectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class QueryOutcome:
    """Documents returned from a query and the retrieval mode actually used."""

    documents: list[Document]
    retrieval_mode: RetrievalMode
    reranker_unavailable: bool = False


class LibraryQuerySession:
    """Search a loaded library without modifying its manifest or FAISS index."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_service: EmbeddingService,
        *,
        retrieval_documents: Sequence[Document],
        retrieval_mode: RetrievalMode,
        top_k: int,
        candidate_k: int,
        rrf_constant: int,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._retrieval_mode = retrieval_mode
        self._top_k = max(1, top_k)
        self._candidate_k = max(1, candidate_k)
        self._retriever = HybridRetriever(
            vector_store,
            retrieval_documents,
            rrf_constant=rrf_constant,
            reranker=reranker,
        )

    def query(self, question: str) -> QueryOutcome:
        """Retrieve evidence for one independent user question."""
        query_vector = self._embedding_service.embed_query(question)
        try:
            documents = self._retriever.search(
                question,
                query_vector,
                mode=self._retrieval_mode,
                k=self._top_k,
                candidate_k=self._candidate_k,
            )
        except RerankerUnavailableError:
            if self._retrieval_mode != "hybrid-rerank":
                raise
            documents = self._retriever.search(
                question,
                query_vector,
                mode="hybrid",
                k=self._top_k,
                candidate_k=self._candidate_k,
            )
            return QueryOutcome(documents, retrieval_mode="hybrid", reranker_unavailable=True)
        return QueryOutcome(documents, retrieval_mode=self._retrieval_mode)


def format_evidence(documents: Sequence[Document]) -> list[str]:
    """Return human-readable evidence lines for terminal output."""
    output: list[str] = []
    for index, document in enumerate(documents, start=1):
        preview = document.page_content[:240].replace("\n", " ")
        output.append(
            f"[{index}] title={document.metadata.get('title')!r} "
            f"source={document.metadata.get('source_path')!r} "
            f"page_or_unit={document.metadata.get('source_unit')!r} "
            f"chunk_id={document.metadata.get('chunk_id')!r}\n"
            f"  excerpt={preview!r}\n"
            f"  metadata={document.metadata}"
        )
    return output
