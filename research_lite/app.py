#
# app.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

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
from research_lite.library_refresh import refresh_library
from research_lite.manifest import IngestionManifest, config_fingerprint, resolve_library_root
from research_lite.preprocessing import SplitConfig, split_documents
from research_lite.retrieval import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    HybridRetriever,
    RerankerUnavailableError,
    RetrievalMode,
)
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
    local_files_only: bool = False,
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
        local_files_only=local_files_only,
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
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require embedding and reranker models to be available in the local cache.",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    library_commands = parser.add_mutually_exclusive_group()
    library_commands.add_argument(
        "--init-library",
        action="store_true",
        help="Initialize a local .researchlite manifest without ingesting documents.",
    )
    library_commands.add_argument(
        "--refresh-library",
        action="store_true",
        help="Incrementally synchronize a library's .researchlite FAISS index and manifest.",
    )
    parser.add_argument(
        "--library-dir",
        help="Directory that owns .researchlite.",
    )
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
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "hybrid", "hybrid-rerank"),
        default="hybrid-rerank",
        help=(
            "Advanced override: FAISS only, FAISS+BM25 RRF fusion, or fusion followed "
            "by a cross-encoder (default: hybrid-rerank)."
        ),
    )
    parser.add_argument(
        "--retrieval-candidates",
        type=int,
        default=20,
        help="Candidates retained before returning or reranking the final top-k.",
    )
    parser.add_argument(
        "--rrf-constant",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant used in hybrid modes.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model loaded only for hybrid-rerank queries.",
    )
    parser.add_argument(
        "--reranker-device",
        default="cpu",
        help="Device used by the optional cross-encoder reranker.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help="Batch size used by the optional cross-encoder reranker.",
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
        help="Skip LLM generation and show scored retrieval evidence only.",
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

    if args.init_library:
        library_root = resolve_library_root(args.path, args.library_dir)
        manifest = IngestionManifest.initialize(
            library_root,
            current_config_fingerprint=config_fingerprint(
                model_name=args.model_name,
                split_config=split_config,
            ),
        )
        try:
            print(f"Initialized ResearchLite library manifest at '{manifest.state_dir}'.")
        finally:
            manifest.close()
        return

    if args.refresh_library:
        library_root = resolve_library_root(args.path, args.library_dir)
        refresh_service = build_default_embedding_service(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        result = refresh_library(
            library_root,
            current_config_fingerprint=config_fingerprint(
                model_name=args.model_name,
                split_config=split_config,
            ),
            embedding_service=refresh_service,
            split_config=split_config,
            extensions=args.extensions,
            index_name=args.index_name,
        )
        plan = result.plan
        print(
            "Refresh plan: "
            f"new={len(plan.new)}, changed={len(plan.changed)}, "
            f"unchanged={len(plan.unchanged)}, deleted={len(plan.deleted)}, "
            f"inspection_failed={len(plan.failed)}."
        )
        print(
            "Refresh result: "
            f"indexed={len(result.indexed_sources)}, failed={len(result.failed_sources)}, "
            f"deleted_chunks={result.deleted_chunks}, added_chunks={result.added_chunks}."
        )
        for refresh_failure in plan.failed:
            print(f"  Inspection failed: {refresh_failure.path}: {refresh_failure.reason}")
        for source_file in result.failed_sources:
            print(f"  Refresh failed: {source_file}")
        return

    embedded: list[EmbeddedDocument] = []
    vector_store: FaissVectorStore | None = None
    service: EmbeddingService | None = None

    if args.load_index:
        service = build_default_embedding_service(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
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
            local_files_only=args.local_files_only,
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
            retrieval_mode: RetrievalMode = args.retrieval_mode
            reranker = (
                CrossEncoderReranker(
                    args.reranker_model,
                    device=args.reranker_device,
                    batch_size=args.reranker_batch_size,
                    local_files_only=args.local_files_only,
                )
                if retrieval_mode == "hybrid-rerank"
                else None
            )
            if embedded:
                retrieval_documents = [item.document for item in embedded]
            else:
                retrieval_documents = vector_store.documents()
            retriever = HybridRetriever(
                vector_store,
                retrieval_documents,
                rrf_constant=args.rrf_constant,
                reranker=reranker,
            )
            print(f"Running {retrieval_mode} search for query: {args.query!r}")
            query_vector = service.embed_query(args.query)
            search_kwargs = {
                "k": max(1, args.top_k),
                "candidate_k": max(1, args.retrieval_candidates),
            }
            try:
                results = retriever.search(
                    args.query,
                    query_vector,
                    mode=retrieval_mode,
                    **search_kwargs,
                )
            except RerankerUnavailableError as exc:
                if retrieval_mode != "hybrid-rerank":
                    raise
                print(f"Reranker unavailable ({exc}); falling back to hybrid retrieval.")
                results = retriever.search(
                    args.query,
                    query_vector,
                    mode="hybrid",
                    **search_kwargs,
                )
            if not results:
                print("No results returned from retrieval.")
            else:
                print(f"Found {len(results)} relevant evidence chunks:")
                for idx, doc in enumerate(results, start=1):
                    preview = doc.page_content[:240].replace("\n", " ")
                    print(
                        f"[{idx}] title={doc.metadata.get('title')!r} "
                        f"source={doc.metadata.get('source_path')!r} "
                        f"page_or_unit={doc.metadata.get('source_unit')!r} "
                        f"chunk_id={doc.metadata.get('chunk_id')!r}\n"
                        f"  excerpt={preview!r}\n"
                        f"  metadata={doc.metadata}"
                    )

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
