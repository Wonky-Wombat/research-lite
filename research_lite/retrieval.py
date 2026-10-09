#
# retrieval.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

"""Composable dense, hybrid, and reranked retrieval for ResearchLite."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, Protocol, cast

from langchain_core.documents import Document

from research_lite.defaults import DEFAULT_RERANKER_MODEL
from research_lite.model_loading import model_cache_dir, silence_model_downloads, use_cuda
from research_lite.vectorstore import LibraryIndex

RetrievalMode = Literal["dense", "hybrid", "hybrid-rerank"]


class _CrossEncoder(Protocol):
    """Minimal interface used from fastembed's cross-encoder."""

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = 64
    ) -> Iterable[float]: ...


class RerankerUnavailableError(RuntimeError):
    """Raised when the optional cross-encoder cannot be loaded locally."""


def _document_key(document: Document) -> str:
    """Return the stable chunk identity used when fusing rankings."""
    metadata = document.metadata
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id is not None:
        return str(chunk_id)
    return "\x1f".join(
        str(metadata.get(field, "")) for field in ("source_path", "source_unit", "chunk_index")
    ) or str(id(document))


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
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                silence_model_downloads()
                self._model = TextCrossEncoder(
                    self._model_name,
                    cache_dir=model_cache_dir(),
                    cuda=use_cuda(self._device),
                    local_files_only=self._local_files_only,
                )
            except Exception as exc:
                msg = f"Could not load reranker model {self._model_name!r}."
                raise RerankerUnavailableError(msg) from exc
        return self._model

    def warm_up(self) -> None:
        """Load the reranker before an interactive session starts."""
        self._get_model()

    def rerank(self, query: str, documents: Sequence[Document]) -> list[Document]:
        """Return the candidate chunks ordered by cross-encoder relevance."""
        if not documents:
            return []
        model = cast(_CrossEncoder, self._get_model())
        scores = list(
            model.rerank(
                query,
                [document.page_content for document in documents],
                batch_size=self._batch_size,
            )
        )
        ranked = sorted(
            zip(documents, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )
        return [document for document, _ in ranked]


class HybridRetriever:
    """Retrieve dense and keyword candidates, then optionally rerank the fusion."""

    def __init__(
        self,
        index: LibraryIndex,
        *,
        rrf_constant: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if rrf_constant < 1:
            msg = "RRF constant must be positive."
            raise ValueError(msg)
        self._index = index
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
        dense_results = self._index.similarity_search(query_vector, k=fetch_k)
        if mode == "dense":
            return dense_results[:k]

        lexical_results = self._index.keyword_search(query, k=fetch_k)
        fused_results = reciprocal_rank_fusion(
            (dense_results, lexical_results), k=fetch_k, constant=self._rrf_constant
        )
        if mode == "hybrid-rerank":
            if self._reranker is None:
                msg = "hybrid-rerank mode requires a configured cross-encoder reranker."
                raise ValueError(msg)
            fused_results = self._reranker.rerank(query, fused_results)
        return fused_results[:k]
