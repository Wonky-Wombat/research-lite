from __future__ import annotations

import argparse
from collections.abc import Iterable

from knowledge_lite.embedding import EmbeddedDocument, EmbeddingService
from knowledge_lite.embedding.embedding_builder import (
    DEFAULT_MODEL_NAME,
    build_default_embedding_service,
)
from knowledge_lite.ingestion import load_documents
from knowledge_lite.preprocessing import SplitConfig, split_documents


def ingest_and_embed(
    path: str,
    *,
    extensions: Iterable[str] | None = None,
    split_config: SplitConfig | None = None,
    embedding_service: EmbeddingService | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[EmbeddedDocument]:
    """Run the load -> split -> embed pipeline for the provided path."""
    documents = load_documents(path, extensions=extensions)
    if not documents:
        return []

    chunks = split_documents(documents, config=split_config or SplitConfig())
    if not chunks:
        return []

    service = embedding_service or build_default_embedding_service(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    return service.embed_documents(chunks)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KnowledgeLite RAG pipeline locally.")
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument(
        "--ext",
        dest="extensions",
        action="append",
        help="File extensions to include (defaults to all supported).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    split_config = SplitConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedded = ingest_and_embed(
        args.path,
        extensions=args.extensions,
        split_config=split_config,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        f"Ingested path '{args.path}' with {len(embedded)} embedded chunks "
        f"using model '{args.model_name}'."
    )


if __name__ == "__main__":
    main()
