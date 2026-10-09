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
from langchain_core.documents import Document

from research_lite.manifest import IngestionManifest


class VectorIndex:
    """Search stored chunk vectors in memory by cosine similarity."""

    def __init__(self, documents: list[Document], vectors: np.ndarray) -> None:
        if len(documents) != len(vectors):
            msg = f"Got {len(vectors)} vectors for {len(documents)} chunks."
            raise ValueError(msg)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(vectors) else 1.0
        self._documents = documents
        self._vectors = vectors / np.maximum(norms, 1e-12)

    @classmethod
    def load(cls, library_root: Path) -> VectorIndex:
        manifest = IngestionManifest.open(library_root)
        try:
            documents, vectors = manifest.load_chunks()
        finally:
            manifest.close()
        return cls(documents, vectors)

    def documents(self) -> list[Document]:
        return list(self._documents)

    def similarity_search(self, query_vector: Sequence[float], *, k: int = 4) -> list[Document]:
        if not self._documents or not query_vector or k < 1:
            return []
        scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [self._documents[i] for i in top[np.argsort(-scores[top], kind="stable")]]


__all__ = ["VectorIndex"]
