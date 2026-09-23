#
# retrieval.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

"""Composable dense, hybrid, and reranked retrieval for ResearchLite."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Literal, Protocol, cast

from langchain_core.documents import Document

from research_lite.vectorstore import FaissVectorStore

RetrievalMode = Literal["dense", "hybrid", "hybrid-rerank"]
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class _CrossEncoder(Protocol):
    """Minimal interface used from sentence-transformers' cross-encoder."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


class RerankerUnavailableError(RuntimeError):
    """Raised when the optional cross-encoder cannot be loaded locally."""


def _tokenize(text: str) -> list[str]:
    """Return deterministic word tokens for lexical retrieval."""
    return re.findall(r"\w+", text.lower())


def _document_key(document: Document) -> str:
    """Return the stable chunk identity used when fusing rankings."""
    metadata = document.metadata
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id is not None:
        return str(chunk_id)
    return "\x1f".join(
        str(metadata.get(field, "")) for field in ("source_path", "source_unit", "chunk_index")
    ) or str(id(document))


class BM25Retriever:
    """Dependency-free BM25 retrieval over an in-memory chunk collection."""

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            msg = "Cannot build a BM25 index from an empty document collection."
            raise ValueError(msg)
        if k1 <= 0:
            msg = "BM25 k1 must be positive."
            raise ValueError(msg)
        if not 0 <= b <= 1:
            msg = "BM25 b must be between zero and one."
            raise ValueError(msg)

        self._documents = list(documents)
        self._k1 = k1
        self._b = b
        self._term_frequencies = [
            Counter(_tokenize(document.page_content)) for document in self._documents
        ]
        self._document_lengths = [sum(terms.values()) for terms in self._term_frequencies]
        self._average_document_length = max(
            sum(self._document_lengths) / len(self._document_lengths), 1.0
        )
        self._document_frequencies: Counter[str] = Counter(
            term for terms in self._term_frequencies for term in terms
        )

    def search(self, query: str, *, k: int = 4) -> list[Document]:
        """Return up to ``k`` chunks ranked by BM25 score."""
        if k < 1:
            return []
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        document_count = len(self._documents)
        scores: list[tuple[int, float]] = []
        for index, (terms, document_length) in enumerate(
            zip(self._term_frequencies, self._document_lengths, strict=True)
        ):
            normalization = self._k1 * (
                1 - self._b + self._b * document_length / self._average_document_length
            )
            score = 0.0
            for term in query_terms:
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (self._k1 + 1)
                    / (frequency + normalization)
                )
            if score > 0:
                scores.append((index, score))

        return [
            self._documents[index]
            for index, _ in sorted(scores, key=lambda item: (-item[1], item[0]))[:k]
        ]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Document]], *, k: int = 4, constant: int = 60
) -> list[Document]:
    """Fuse ranked lists with RRF, deduplicating chunks by stable identity."""
    if k < 1:
        return []
    if constant < 1:
        msg = "RRF constant must be positive."
        raise ValueError(msg)

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = _document_key(document)
            documents[key] = document
            scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    return [documents[key] for key in sorted(scores, key=lambda item: (-scores[item], item))[:k]]


class CrossEncoderReranker:
    """Lazy cross-encoder wrapper for reranking a bounded candidate set."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            msg = "Reranker batch size must be positive."
            raise ValueError(msg)
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._local_files_only = local_files_only
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    local_files_only=self._local_files_only,
                )
            except Exception as exc:
                msg = f"Could not load reranker model {self._model_name!r}."
                raise RerankerUnavailableError(msg) from exc
        return self._model

    def rerank(self, query: str, documents: Sequence[Document]) -> list[Document]:
        """Return the candidate chunks ordered by cross-encoder relevance."""
        if not documents:
            return []
        model = cast(_CrossEncoder, self._get_model())
        scores = model.predict(
            [(query, document.page_content) for document in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(documents, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )
        return [document for document, _ in ranked]


class HybridRetriever:
    """Retrieve FAISS and BM25 candidates, then optionally rerank the fusion."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        documents: Sequence[Document],
        *,
        rrf_constant: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if rrf_constant < 1:
            msg = "RRF constant must be positive."
            raise ValueError(msg)
        self._vector_store = vector_store
        self._bm25 = BM25Retriever(documents)
        self._rrf_constant = rrf_constant
        self._reranker = reranker

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        mode: RetrievalMode = "dense",
        k: int = 4,
        candidate_k: int = 20,
    ) -> list[Document]:
        """Retrieve top chunks using dense, hybrid, or hybrid-rerank search."""
        if k < 1 or candidate_k < 1:
            return []
        if mode not in {"dense", "hybrid", "hybrid-rerank"}:
            msg = f"Unsupported retrieval mode: {mode}."
            raise ValueError(msg)

        fetch_k = max(k, candidate_k)
        dense_results = self._vector_store.similarity_search(query_vector, k=fetch_k)
        if mode == "dense":
            return dense_results[:k]

        lexical_results = self._bm25.search(query, k=fetch_k)
        fused_results = reciprocal_rank_fusion(
            (dense_results, lexical_results), k=fetch_k, constant=self._rrf_constant
        )
        if mode == "hybrid-rerank":
            if self._reranker is None:
                msg = "hybrid-rerank mode requires a configured cross-encoder reranker."
                raise ValueError(msg)
            fused_results = self._reranker.rerank(query, fused_results)
        return fused_results[:k]
