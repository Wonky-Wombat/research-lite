#
# __init__.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from research_lite import Document
from research_lite.manifest import IngestionManifest


class LibraryIndex:
    """Search a library's chunks by vector similarity and FTS5 keywords."""

    def __init__(
        self, library_root: Path, ids: list[int], documents: list[Document], vectors: np.ndarray
    ) -> None:
        if not len(ids) == len(documents) == len(vectors):
            msg = f"Got {len(vectors)} vectors and {len(ids)} ids for {len(documents)} chunks."
            raise ValueError(msg)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(vectors) else 1.0
        self._library_root = library_root
        self._documents = documents
        self._documents_by_id = dict(zip(ids, documents, strict=True))
        self._vectors = vectors / np.maximum(norms, 1e-12)

    @classmethod
    def load(cls, library_root: Path) -> LibraryIndex:
        manifest = IngestionManifest.open(library_root)
        try:
            ids, documents, vectors = manifest.load_chunks()
        finally:
            manifest.close()
        return cls(library_root, ids, documents, vectors)

    def documents(self) -> list[Document]:
        return list(self._documents)

    def similarity_search(self, query_vector: Sequence[float], *, k: int = 4) -> list[Document]:
        if not self._documents or not query_vector or k < 1:
            return []
        scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [self._documents[i] for i in top[np.argsort(-scores[top], kind="stable")]]

    def keyword_search(self, query: str, *, k: int = 4) -> list[Document]:
        manifest = IngestionManifest.open(self._library_root)
        try:
            ids = manifest.keyword_search(query, k=k)
        finally:
            manifest.close()
        return [self._documents_by_id[i] for i in ids if i in self._documents_by_id]


__all__ = ["LibraryIndex"]
