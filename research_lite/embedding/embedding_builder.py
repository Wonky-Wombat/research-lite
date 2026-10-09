#
# embedding_builder.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from research_lite.defaults import DEFAULT_EMBEDDING_MODEL
from research_lite.model_loading import model_cache_dir, silence_model_downloads, use_cuda

from . import EmbeddingConfig, EmbeddingService

DEFAULT_MODEL_NAME = DEFAULT_EMBEDDING_MODEL


class FastEmbedEmbeddings:
    def __init__(
        self, model_name: str, *, device: str = "cpu", local_files_only: bool = False
    ) -> None:
        from fastembed import TextEmbedding

        silence_model_downloads()
        self._model = TextEmbedding(
            model_name,
            cache_dir=model_cache_dir(),
            cuda=use_cuda(device),
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector: list[float] = next(iter(self._model.query_embed(text))).tolist()
        return vector


def build_default_embedding_service(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    local_files_only: bool = False,
) -> EmbeddingService:
    """Construct the project's default embedding service."""
    backend = FastEmbedEmbeddings(model_name, device=device, local_files_only=local_files_only)
    return EmbeddingService(backend, EmbeddingConfig(batch_size=batch_size))


__all__ = ["build_default_embedding_service", "DEFAULT_MODEL_NAME", "FastEmbedEmbeddings"]
