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


WEAK_MATCH_SCORE = -5.0
HISTORY_TURNS = 3
RETRIEVAL_HISTORY_TURNS = 1


@dataclass(frozen=True)
class QueryOutcome:
    """Documents returned from a query and the retrieval mode actually used."""

    documents: list[Document]
    retrieval_mode: RetrievalMode
    reranker_unavailable: bool = False
    best_score: float | None = None

    @property
    def weak_match(self) -> bool:
        """Whether the best passage scored too low to be worth trusting."""
        return self.best_score is not None and self.best_score < WEAK_MATCH_SCORE


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

    def query(self, question: str, history: Sequence[tuple[str, str]] = ()) -> QueryOutcome:
        """Retrieve evidence for a question, using earlier turns to resolve what it refers to."""
        retrieval_text = retrieval_query(question, history)
        query_vector = self._embedding_service.embed_query(retrieval_text)
        try:
            documents, best_score = self._retriever.search_with_best_score(
                retrieval_text,
                query_vector,
                mode=self._retrieval_mode,
                k=self._top_k,
                candidate_k=self._candidate_k,
            )
        except RerankerUnavailableError:
            if self._retrieval_mode != "hybrid-rerank":
                raise
            documents = self._retriever.search(
                retrieval_text,
                query_vector,
                mode="hybrid",
                k=self._top_k,
                candidate_k=self._candidate_k,
            )
            return QueryOutcome(documents, retrieval_mode="hybrid", reranker_unavailable=True)
        return QueryOutcome(documents, retrieval_mode=self._retrieval_mode, best_score=best_score)


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
    return {
        "query": question,
        "retrieval_mode": outcome.retrieval_mode,
        "warnings": query_warnings(outcome),
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


def retrieval_query(question: str, history: Sequence[tuple[str, str]] = ()) -> str:
    """Prefix a follow-up with recent questions so pronouns still retrieve the right passages."""
    if not history:
        return question
    earlier = [past_question for past_question, _ in history[-RETRIEVAL_HISTORY_TURNS:]]
    return " ".join([*earlier, question])


def query_warnings(outcome: QueryOutcome) -> list[str]:
    """Return the notices a caller should repeat to the user."""
    warnings = []
    if outcome.reranker_unavailable:
        warnings.append("Reranker unavailable; used hybrid retrieval.")
    if outcome.weak_match:
        warnings.append("No strongly relevant passage was found; the library may not cover this.")
    return warnings


def format_context(chunk_id: str, documents: Sequence[Document]) -> list[str]:
    """Render a chunk and its neighbours, marking the one that was asked for."""
    lines = []
    for document in documents:
        metadata = document.metadata
        marker = ">" if str(metadata.get("chunk_id", "")) == chunk_id else " "
        title = str(metadata.get("title", "Unknown"))
        lines.append(f"{marker} {title}{page_label(metadata)}\n  {document.page_content}")
    return lines


def context_json(chunk_id: str, documents: Sequence[Document]) -> dict[str, object]:
    """Return a chunk and its neighbours as JSON."""
    return {
        "chunk_id": chunk_id,
        "passages": [
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "title": str(document.metadata.get("title", "Unknown")),
                "path": str(document.metadata.get("source_path", "")),
                "page": page_number(document.metadata),
                "matched": str(document.metadata.get("chunk_id", "")) == chunk_id,
                "text": document.page_content,
            }
            for document in documents
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
    return {
        "query": question,
        "retrieval_mode": outcome.retrieval_mode,
        "warnings": query_warnings(outcome),
        "results": results,
    }
