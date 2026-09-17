from __future__ import annotations

from . import EmbeddingConfig, EmbeddingService

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_default_embedding_service(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingService:
    """Construct the project's default embedding service."""
    from langchain_huggingface import HuggingFaceEmbeddings

    backend = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    return EmbeddingService(backend, EmbeddingConfig(batch_size=batch_size))


__all__ = ["build_default_embedding_service", "DEFAULT_MODEL_NAME"]
