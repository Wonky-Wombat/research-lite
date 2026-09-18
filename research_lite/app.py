from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv

from research_lite.embedding import EmbeddedDocument, EmbeddingService
from research_lite.embedding.embedding_builder import (
    DEFAULT_MODEL_NAME,
    build_default_embedding_service,
)
from research_lite.generation import RAGGenerator
from research_lite.ingestion import IngestReport, load_documents
from research_lite.preprocessing import SplitConfig, split_documents
from research_lite.vectorstore import FaissVectorStore


def ingest_and_embed(
    path: str,
    *,
    extensions: Iterable[str] | None = None,
    split_config: SplitConfig | None = None,
    embedding_service: EmbeddingService | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
    ingest_report: IngestReport | None = None,
) -> tuple[list[EmbeddedDocument], FaissVectorStore | None, EmbeddingService | None]:
    """Run the load -> split -> embed pipeline for the provided path."""
    documents = load_documents(path, extensions=extensions, report=ingest_report)
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
    parser = argparse.ArgumentParser(description="Run the ResearchLite RAG pipeline locally.")
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
        "--save-index", help="Directory to save the built FAISS index (if available)."
    )
    parser.add_argument(
        "--load-index", help="Directory containing a previously saved FAISS index to load."
    )
    parser.add_argument("--index-name", default="index")
    parser.add_argument("--allow-dangerous-deserialization", action="store_true")
    parser.add_argument(
        "--query", help="Optional text to embed and search against the built FAISS index."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of search results to return when running a query.",
    )
    # LLM Generation arguments
    parser.add_argument(
        "--llm-model",
        default="gpt-5-mini",
        help="LLM model name for generation (e.g., gpt-3.5-turbo, gpt-4).",
    )
    parser.add_argument(
        "--llm-api-key",
        help="API Key for the LLM service. Can also be set via OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Base URL for the LLM service (useful for compatible APIs like DeepSeek/LocalAI).",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip LLM generation and only show retrieval results.",
    )
    return parser


def main() -> None:
    # Load environment variables from .env file if present
    load_dotenv()

    # Clean up API Key from env if present (remove whitespace/newlines)
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

    parser = _build_arg_parser()
    args = parser.parse_args()

    split_config = SplitConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    embedded: list[EmbeddedDocument] = []
    vector_store: FaissVectorStore | None = None
    service: EmbeddingService | None = None

    if args.load_index:
        service = build_default_embedding_service(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        try:
            vector_store = FaissVectorStore.load(
                Path(args.load_index),
                embedding_backend=service.backend,
                index_name=args.index_name,
                allow_dangerous_deserialization=args.allow_dangerous_deserialization,
            )
            print(f"Loaded FAISS index '{args.index_name}' from '{args.load_index}'.")
        except ImportError as exc:
            raise RuntimeError("FAISS is required to load a index.") from exc
    else:
        ingest_report = IngestReport()
        embedded, vector_store, service = ingest_and_embed(
            args.path,
            extensions=args.extensions,
            split_config=split_config,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            ingest_report=ingest_report,
        )
        print(
            f"Ingested path '{args.path}' with {len(embedded)} embedded chunks "
            f"using model '{args.model_name}'."
        )
        print(
            "Ingest report: "
            f"discovered={ingest_report.discovered_files}, "
            f"processed={ingest_report.processed_files}, "
            f"failed={ingest_report.failed_files}."
        )
        for failure in ingest_report.failures:
            print(f"  Failed: {failure.source_path}: {failure.reason}")
        if vector_store is not None:
            print("Built FAISS vector store with the embedded chunks.")
        if args.save_index:
            if vector_store is None:
                print("Skipping FAISS index save because the dependency is unavailable.")
            else:
                output_dir = Path(args.save_index)
                vector_store.save(output_dir, index_name=args.index_name)
                print(f"Saved FAISS index '{args.index_name}' to '{output_dir}'.")

    if args.query:
        if vector_store is None:
            print("Cannot run query because the FAISS index was not built or loaded.")
        elif service is None:
            print("Cannot run query because no embedding service was available.")
        else:
            print(f"Running similarity search for query: {args.query!r}")
            query_vector = service.embed_query(args.query)
            results = vector_store.similarity_search(query_vector, k=max(1, args.top_k))
            if not results:
                print("No results returned from FAISS.")
            else:
                print(f"Found {len(results)} relevant chunks:")
                for idx, doc in enumerate(results, start=1):
                    preview = doc.page_content[:80].replace("\n", " ")
                    print(f"[{idx}] {preview!r} metadata={doc.metadata}")

                if not args.no_generation:
                    print("\nGenerating answer...")
                    try:
                        # Ensure we pass the cleaned key if it wasn't passed via args
                        final_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")

                        generator = RAGGenerator(
                            model_name=args.llm_model,
                            api_key=final_api_key,
                            base_url=args.llm_base_url,
                        )
                        answer = generator.generate_answer(args.query, results)
                        print(f"\nAnswer:\n{answer}")
                    except Exception as e:
                        print(f"Failed to generate answer: {e}")
                        print("Tip: Ensure you have set OPENAI_API_KEY or passed --llm-api-key.")


if __name__ == "__main__":
    main()
