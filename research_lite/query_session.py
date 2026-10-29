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
from research_lite.chunk_kind import BODY
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


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """Documents returned from a query and the retrieval mode actually used."""

    documents: list[Document]
    retrieval_mode: RetrievalMode
    reranker_unavailable: bool = False
    best_score: float | None = None
    scores: tuple[float, ...] = ()

    @property
    def weak_match(self) -> bool:
        """Whether the best passage scored too low to be worth trusting."""
        return self.best_score is not None and self.best_score < WEAK_MATCH_SCORE

    def matching_sources(self, *, prose_only: bool = False) -> list[SourceMatch]:
        """Group retrieved chunks by file, keeping the order of their best passage.

        ``prose_only`` drops a document's back matter before grouping: a submission checklist or a
        reference list that happens to match does not make its paper a document about the subject.
        """
        order: list[str] = []
        titles: dict[str, str] = {}
        counts: Counter[str] = Counter()
        best_pages: dict[str, int | None] = {}
        best_scores: dict[str, float] = {}
        scores = self.scores or (None,) * len(self.documents)
        for document, score in zip(self.documents, scores, strict=False):
            metadata = document.metadata
            if prose_only and metadata.get("kind", BODY) != BODY:
                continue
            key = str(metadata.get("source_path", "")) or str(metadata.get("title", ""))
            if key not in titles:
                order.append(key)
                titles[key] = str(metadata.get("title", "Unknown"))
                best_pages[key] = page_number(metadata)
            counts[key] += 1
            if score is not None and (key not in best_scores or score > best_scores[key]):
                best_scores[key] = score
        return [
            SourceMatch(
                rank=rank,
                title=titles[key],
                source_path=key,
                passages=counts[key],
                best_page=best_pages[key],
                score=best_scores.get(key),
            )
            for rank, key in enumerate(order, start=1)
        ]

    def warnings(self) -> list[str]:
        """Return the notices a caller should repeat to the user."""
        warnings = []
        if self.reranker_unavailable:
            warnings.append("Reranker unavailable; used hybrid retrieval.")
        if self.weak_match:
            warnings.append(
                "No strongly relevant passage was found; the library may not cover this."
            )
        return warnings

    def as_evidence_lines(self) -> list[str]:
        """Human-readable evidence lines for terminal output."""
        numbers = citation_number_by_source(self.documents)
        lines: list[str] = []
        for document in self.documents:
            preview = document.page_content[:240].replace("\n", " ")
            source_path = str(document.metadata.get("source_path", ""))
            title = document.metadata.get("title", "Unknown")
            number = numbers.get(source_path or str(title), 0)
            lines.append(f"[{number}] {title}{page_label(document.metadata)}\n  {preview}")
        return lines

    def as_evidence_json(self, question: str) -> dict[str, object]:
        """The retrieved passages with their text, for a caller that parses."""
        numbers = citation_number_by_source(self.documents)
        results = []
        for rank, document in enumerate(self.documents, start=1):
            metadata = document.metadata
            source_path = str(metadata.get("source_path", ""))
            title = str(metadata.get("title", "Unknown"))
            results.append(
                {
                    "rank": rank,
                    "citation": numbers.get(source_path or title, 0),
                    "title": title,
                    "path": source_path,
                    "page": page_number(metadata),
                    "chunk_id": metadata.get("chunk_id"),
                    "text": document.page_content,
                }
            )
        return {
            "query": question,
            "retrieval_mode": self.retrieval_mode,
            "warnings": self.warnings(),
            "results": results,
        }

    def as_sources_json(self, question: str) -> dict[str, object]:
        """The matching files without their text, for listing rather than reading."""
        return {
            "query": question,
            "retrieval_mode": self.retrieval_mode,
            "warnings": self.warnings(),
            "files": [
                {
                    "rank": match.rank,
                    "title": match.title,
                    "path": match.source_path,
                    "page": match.best_page,
                    "passages": match.passages,
                    "score": match.score,
                    "match": match_strength(match.score),
                }
                for match in self.matching_sources()
            ],
        }


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

    def query(self, question: str, *, top_k: int | None = None) -> QueryOutcome:
        """Retrieve evidence for a question."""
        query_vector = self._embedding_service.embed_query(question)
        k = self._top_k if top_k is None else max(1, top_k)
        try:
            documents, best_score, scores = self._retriever.search_with_best_score(
                question,
                query_vector,
                mode=self._retrieval_mode,
                k=k,
                candidate_k=self._candidate_k,
            )
        except RerankerUnavailableError:
            if self._retrieval_mode != "hybrid-rerank":
                raise
            documents = self._retriever.search(
                question,
                query_vector,
                mode="hybrid",
                k=k,
                candidate_k=self._candidate_k,
            )
            return QueryOutcome(documents, retrieval_mode="hybrid", reranker_unavailable=True)
        return QueryOutcome(
            documents,
            retrieval_mode=self._retrieval_mode,
            best_score=best_score,
            scores=tuple(scores),
        )


@dataclass(frozen=True, slots=True)
class SourceMatch:
    """One library file that matched a query, with how strongly it matched."""

    rank: int
    title: str
    source_path: str
    passages: int
    best_page: int | None
    score: float | None = None


def match_strength(score: float | None) -> str:
    """Name a relevance score, using the bands measured on the arXiv corpus."""
    if score is None:
        return ""
    if score >= 2.0:
        return "strong"
    if score >= -2.0:
        return "moderate"
    if score >= WEAK_MATCH_SCORE:
        return "weak"
    return "very weak"


def format_sources(matches: Sequence[SourceMatch]) -> list[str]:
    """Render one line per matching file for terminal output."""
    lines = []
    for match in matches:
        page = f", page {match.best_page}" if match.best_page is not None else ""
        plural = "" if match.passages == 1 else "s"
        lines.append(f"[{match.rank}] {match.title}{page}  ({match.passages} passage{plural})")
    return lines


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
