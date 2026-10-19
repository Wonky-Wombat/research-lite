#
# mcp_tools.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-18.
#

"""The retrieval tools an MCP host can call."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from research_lite import sync_job
from research_lite.defaults import DEFAULT_EMBEDDING_MODEL
from research_lite.library_stats import inspect_library
from research_lite.manifest import IngestionManifest, config_fingerprint, resolve_library_root
from research_lite.preprocessing import SplitConfig
from research_lite.query_session import LibraryQuerySession, context_json
from research_lite.retrieval import RetrievalMode

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from research_lite.embedding import EmbeddingService
    from research_lite.retrieval import CrossEncoderReranker
    from research_lite.vectorstore import LibraryIndex

Reply = tuple[str, Any]

MODES = ("dense", "hybrid", "hybrid-rerank")
TERMS = __import__("re").compile(r"[a-z0-9][a-z0-9-]*")


def _summarise(payload: Mapping[str, Any]) -> str:
    """The text block beside the structured result, kept short on purpose."""
    return json.dumps(payload, ensure_ascii=False, indent=1)


class Tools:
    """Holds the open library so each call does not reload the index or the models."""

    def __init__(self, library_root: Path) -> None:
        self.root = resolve_library_root(library_root)
        self._index: LibraryIndex | None = None
        self._parts: tuple[EmbeddingService, CrossEncoderReranker] | None = None
        self._sessions: dict[str, LibraryQuerySession] = {}

    def _library(self) -> LibraryIndex:
        """The index and the two models, loaded once and shared by every mode."""
        if self._parts is None:
            from research_lite.embedding.embedding_builder import (
                build_default_embedding_service,
            )
            from research_lite.retrieval import CrossEncoderReranker
            from research_lite.vectorstore import LibraryIndex

            self._index = LibraryIndex.load(self.root)
            self._parts = (build_default_embedding_service(), CrossEncoderReranker())
        assert self._index is not None
        return self._index

    def _session(self, mode: RetrievalMode) -> LibraryQuerySession:
        """One session per mode, so two calls in flight cannot see each other's setting."""
        if mode not in self._sessions:
            index = self._library()
            assert self._parts is not None
            service, reranker = self._parts
            self._sessions[mode] = LibraryQuerySession(
                index,
                service,
                retrieval_mode=mode,
                top_k=4,
                candidate_k=20,
                rrf_constant=60,
                reranker=reranker if mode == "hybrid-rerank" else None,
            )
        return self._sessions[mode]

    def search(self, arguments: Mapping[str, Any]) -> Reply:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("search needs a query.")
        asked = str(arguments.get("mode", "hybrid-rerank"))
        if asked not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}.")
        mode: RetrievalMode = asked  # type: ignore[assignment]
        top_k = int(arguments.get("top_k", 4))
        outcome = self._session(mode).query(query, top_k=max(1, min(top_k, 12)))
        payload = outcome.as_evidence_json(query)
        return _summarise(payload), payload

    def overview(self, arguments: Mapping[str, Any]) -> Reply:
        words = str(arguments.get("terms", "") or "")
        terms = TERMS.findall(words.lower())
        payload = inspect_library(self.root, self._library(), terms=terms).overview(
            terms, titles=bool(arguments.get("titles"))
        )
        return _summarise(payload), payload

    def read_context(self, arguments: Mapping[str, Any]) -> Reply:
        chunk_id = str(arguments.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("read_context needs a chunk_id from a search result.")
        radius = max(0, min(int(arguments.get("radius", 1)), 3))
        manifest = IngestionManifest.open(self.root)
        try:
            documents = manifest.chunk_neighbourhood(chunk_id, radius=radius)
        finally:
            manifest.close()
        if not documents:
            raise ValueError(f"No indexed passage with chunk_id '{chunk_id}'.")
        payload = context_json(chunk_id, documents)
        return _summarise(payload), payload

    def status(self, _arguments: Mapping[str, Any]) -> Reply:
        stats = inspect_library(self.root, self._library())
        job = sync_job.read(self.root)
        payload: dict[str, Any] = {
            "library": str(self.root),
            "documents": stats.indexed_sources,
            "chunks": stats.chunks,
            "failed": stats.failed_sources,
            "last_sync": None if job is None else {"phase": job.phase, "job_id": job.job_id},
        }
        return _summarise(payload), payload

    def sync_start(self, _arguments: Mapping[str, Any]) -> Reply:
        from research_lite.embedding.embedding_builder import build_default_embedding_service
        from research_lite.library_refresh import refresh_library

        running = sync_job.read(self.root)
        if running is not None and running.running:
            payload = {"job_id": running.job_id, "phase": running.phase, "started": False}
            return "A refresh is already running.", payload

        split = SplitConfig()
        fingerprint = config_fingerprint(model_name=DEFAULT_EMBEDDING_MODEL, split_config=split)
        state = sync_job.start(
            self.root,
            lambda on_progress: refresh_library(
                self.root,
                current_config_fingerprint=fingerprint,
                load_embedding_service=build_default_embedding_service,
                split_config=split,
                on_progress=on_progress,
            ),
        )
        payload = {"job_id": state.job_id, "phase": state.phase, "started": True}
        return "Indexing started. Poll sync_status with this job_id.", payload

    def sync_status(self, arguments: Mapping[str, Any]) -> Reply:
        job_id = arguments.get("job_id")
        state = sync_job.read(self.root, str(job_id) if job_id else None)
        if state is None:
            return "This library has never been synced.", {"phase": "none"}
        payload = {
            "job_id": state.job_id,
            "phase": state.phase,
            "file": state.file,
            "index": state.index,
            "total": state.total,
            "done_bytes": state.done_bytes,
            "total_bytes": state.total_bytes,
            "indexed": state.indexed,
            "failed": state.failed,
            "added_chunks": state.added_chunks,
            "error": state.error,
        }
        return _summarise(payload), payload


def definitions() -> list[dict[str, Any]]:
    """The tool list, in a fixed order so a client can cache it."""
    return [
        {
            "name": "search",
            "title": "Search the library",
            "description": (
                "Find the passages in the user's own documents that bear on a question. "
                "Returns a few short passages with their chunk_id, title and page, not "
                "whole files. Use this before answering anything about what the documents "
                "say."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "mode": {
                        "type": "string",
                        "enum": list(MODES),
                        "description": (
                            "hybrid-rerank is the most accurate and the default; dense is "
                            "about a hundred times faster and a few points worse."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Passages to return, 1 to 12. Four by default.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "overview",
            "title": "Describe the whole library",
            "description": (
                "Answer a question about the collection itself rather than about what the "
                "documents say: how many there are, their titles, and how many contain "
                "every one of the given words. Counts are exact. Also states which fields "
                "the library does not record at all, such as authors and dates."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "string",
                        "description": (
                            "Subject words, such as 'deep learning'. Returns how many "
                            "documents hold all of them and which ones they are. Leave "
                            "out to describe the library as a whole."
                        ),
                    },
                    "titles": {
                        "type": "boolean",
                        "description": (
                            "List every title in the library. Costs roughly 20 tokens per "
                            "document, so ask only when the question is what the library "
                            "holds in full. Ignored when terms are given."
                        ),
                    },
                },
            },
        },
        {
            "name": "read_context",
            "title": "Widen one passage",
            "description": (
                "Return a passage from a search result together with its neighbours, for "
                "when a passage is cut off mid-argument. Prefer this over asking for more "
                "search results: it costs a few hundred tokens rather than a whole file."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id from a search result.",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Neighbouring passages on each side, 0 to 3.",
                    },
                },
                "required": ["chunk_id"],
            },
        },
        {
            "name": "status",
            "title": "Library status",
            "description": "How many documents and passages the library holds right now.",
            "inputSchema": {"type": "object", "additionalProperties": False},
        },
        {
            "name": "sync_start",
            "title": "Index new or changed files",
            "description": (
                "Pick up files added or changed since the last index. Returns at once with "
                "a job_id; indexing a hundred documents takes minutes. Poll sync_status."
            ),
            "inputSchema": {"type": "object", "additionalProperties": False},
        },
        {
            "name": "sync_status",
            "title": "Indexing progress",
            "description": "How far a sync started by sync_start has got, or how it ended.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "From sync_start. Omit for the most recent sync.",
                    }
                },
            },
        },
    ]


def build(library_root: Path) -> dict[str, Any]:
    """The definitions and handlers the protocol layer dispatches to."""
    tools = Tools(library_root)
    return {
        "definitions": definitions(),
        "handlers": {
            "search": tools.search,
            "overview": tools.overview,
            "read_context": tools.read_context,
            "status": tools.status,
            "sync_start": tools.sync_start,
            "sync_status": tools.sync_status,
        },
    }


__all__ = ["Tools", "build", "definitions"]
