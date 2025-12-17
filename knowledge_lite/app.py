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
from knowledge_lite.vectorstore import FaissVectorStore


def ingest_and_embed(
    path: str,
    *,
    extensions: Iterable[str] | None = None,
    split_config: SplitConfig | None = None,
    embedding_service: EmbeddingService | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[list[EmbeddedDocument], FaissVectorStore | None, EmbeddingService | None]:
    """Run the load -> split -> embed pipeline for the provided path."""
    documents = load_documents(path, extensions=extensions)
    if not documents:
        return [], None, embedding_service

    chunks = split_documents(documents, config=split_config or SplitConfig())
    if not chunks:
        return [], None, embedding_service

    service = embedding_service or build_default_embedding_service(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    embedded = service.embed_documents(chunks)
    if not embedded:
        return [], None, service

    vector_store: FaissVectorStore | None = None
    try:
        vector_store = FaissVectorStore.from_documents(
            embedded,
            embedding_backend=service.backend,
        )
    except ImportError:
        pass
    return embedded, vector_store, service


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
    parser.add_argument(
        "--query",
        help="Optional text to embed and search against the built FAISS index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of search results to return when running a query.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    split_config = SplitConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedded, vector_store, service = ingest_and_embed(
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
    if vector_store is not None:
        print("Built FAISS vector store with the embedded chunks.")
    if args.query:
        if vector_store is None:
            print("Cannot run query because the FAISS index was not built (missing faiss?).")
        elif service is None:
            print("Cannot run query because no embedding service was available.")
        else:
            print(f"Running similarity search for query: {args.query!r}")
            query_vector = service.embed_query(args.query)
            results = vector_store.similarity_search(query_vector, k=max(1, args.top_k))
            if not results:
                print("No results returned from FAISS.")
            else:
                for idx, doc in enumerate(results, start=1):
                    preview = doc.page_content[:80].replace("\n", " ")
                    print(f"[{idx}] {preview!r} metadata={doc.metadata}")


if __name__ == "__main__":
    main()
