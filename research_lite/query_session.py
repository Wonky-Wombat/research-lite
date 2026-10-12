#
# query_session.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Reusable, read-only query sessions for persisted ResearchLite libraries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from research_lite import Document
from research_lite.citations import citation_number_by_source, page_label, page_number
from research_lite.embedding import EmbeddingService
from research_lite.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    RerankerUnavailableError,
    RetrievalMode,
)
from research_lite.vectorstore import LibraryIndex

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class QueryOutcome:
    """Documents returned from a query and the retrieval mode actually used."""

    documents: list[Document]
    retrieval_mode: RetrievalMode
    reranker_unavailable: bool = False


class LibraryQuerySession:
    """Search a loaded library without modifying its manifest."""

    def __init__(
        self,
        index: LibraryIndex,
        embedding_service: EmbeddingService,
        *,
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
            index,
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
    citation_numbers = citation_number_by_source(documents)
    output: list[str] = []
    for document in documents:
        preview = document.page_content[:240].replace("\n", " ")
        source_path = str(document.metadata.get("source_path", ""))
        title = document.metadata.get("title", "Unknown")
        citation_number = citation_numbers.get(source_path or str(title), 0)
        output.append(f"[{citation_number}] {title}{page_label(document.metadata)}\n  {preview}")
    return output


@dataclass(frozen=True)
class SourceMatch:
    """One library file that matched a query, with how strongly it matched."""

    rank: int
    title: str
    source_path: str
    passages: int
    best_page: int | None


def matching_sources(outcome: QueryOutcome) -> list[SourceMatch]:
    """Group retrieved chunks by file, keeping the order of their best passage."""
    order: list[str] = []
    titles: dict[str, str] = {}
    counts: Counter[str] = Counter()
    best_pages: dict[str, int | None] = {}
    for document in outcome.documents:
        metadata = document.metadata
        key = str(metadata.get("source_path", "")) or str(metadata.get("title", ""))
        if key not in titles:
            order.append(key)
            titles[key] = str(metadata.get("title", "Unknown"))
            best_pages[key] = page_number(metadata)
        counts[key] += 1
    return [
        SourceMatch(
            rank=rank,
            title=titles[key],
            source_path=key,
            passages=counts[key],
            best_page=best_pages[key],
        )
        for rank, key in enumerate(order, start=1)
    ]


def format_sources(matches: Sequence[SourceMatch]) -> list[str]:
    """Render one line per matching file for terminal output."""
    lines = []
    for match in matches:
        page = f", page {match.best_page}" if match.best_page is not None else ""
        plural = "" if match.passages == 1 else "s"
        lines.append(f"[{match.rank}] {match.title}{page}  ({match.passages} passage{plural})")
    return lines


def sources_json(question: str, outcome: QueryOutcome) -> dict[str, object]:
    """Return the matching files for a query as JSON."""
    warnings = (
        ["Reranker unavailable; used hybrid retrieval."] if outcome.reranker_unavailable else []
    )
    return {
        "query": question,
        "retrieval_mode": outcome.retrieval_mode,
        "warnings": warnings,
        "files": [
            {
                "rank": match.rank,
                "title": match.title,
                "path": match.source_path,
                "page": match.best_page,
                "passages": match.passages,
            }
            for match in matching_sources(outcome)
        ],
    }


def evidence_json(question: str, outcome: QueryOutcome) -> dict[str, object]:
    citation_numbers = citation_number_by_source(outcome.documents)
    results = []
    for rank, document in enumerate(outcome.documents, start=1):
        metadata = document.metadata
        source_path = str(metadata.get("source_path", ""))
        title = str(metadata.get("title", "Unknown"))
        results.append(
            {
                "rank": rank,
                "citation": citation_numbers.get(source_path or title, 0),
                "title": title,
                "path": source_path,
                "page": page_number(metadata),
                "chunk_id": metadata.get("chunk_id"),
                "text": document.page_content,
            }
        )
    warnings = (
        ["Reranker unavailable; used hybrid retrieval."] if outcome.reranker_unavailable else []
    )
    return {
        "query": question,
        "retrieval_mode": outcome.retrieval_mode,
        "warnings": warnings,
        "results": results,
    }
