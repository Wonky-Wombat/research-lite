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
from research_lite.generation.llm_service import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from research_lite.ingestion import IngestReport, load_documents
from research_lite.library_lock import LibraryRefreshLockedError
from research_lite.library_refresh import RefreshResult, refresh_library
from research_lite.library_stats import inspect_library
from research_lite.manifest import IngestionManifest, config_fingerprint, resolve_library_root
from research_lite.preprocessing import SplitConfig, split_documents
from research_lite.query_session import LibraryQuerySession, format_evidence
from research_lite.retrieval import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankerUnavailableError,
    RetrievalMode,
)
from research_lite.settings import (
    LibrarySettings,
    LLMSettings,
    load_library_settings,
    load_llm_settings,
    save_active_library,
    save_llm_settings,
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


def load_persisted_library(
    library_root: Path,
    *,
    current_config_fingerprint: str,
    embedding_service: EmbeddingService,
    index_name: str = "index",
) -> FaissVectorStore:
    """Load a library created by ``--refresh-library`` without re-embedding it."""
    manifest = IngestionManifest.open(library_root)
    try:
        recorded_config_fingerprint = manifest.state_value("config_fingerprint")
        state_dir = manifest.state_dir
    finally:
        manifest.close()

    if recorded_config_fingerprint != current_config_fingerprint:
        msg = (
            f"Library at '{library_root}' was indexed with different embedding or chunking "
            "settings. Re-run --refresh-library with the settings used to query it."
        )
        raise RuntimeError(msg)

    index_path = state_dir / f"{index_name}.faiss"
    if not index_path.is_file():
        msg = (
            f"No persisted FAISS index found at '{index_path}'. "
            "Run --refresh-library before querying this library."
        )
        raise FileNotFoundError(msg)

    # RefreshLibrary creates this state locally. FAISS persists its document store in a
    # pickle file, so loading the managed local library necessarily opts into it here.
    return FaissVectorStore.load(
        state_dir,
        embedding_backend=embedding_service.backend,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def _initialize_library_if_needed(library_root: Path, *, current_config_fingerprint: str) -> bool:
    """Create a manifest on first one-click launch and report whether it was new."""
    manifest_path = library_root / ".researchlite" / "manifest.sqlite"
    if manifest_path.is_file():
        return False
    manifest = IngestionManifest.initialize(
        library_root,
        current_config_fingerprint=current_config_fingerprint,
    )
    manifest.close()
    return True


def _print_refresh_result(result: RefreshResult) -> None:
    """Print a concise progress summary for an incremental refresh."""
    plan = result.plan
    print(
        "Library sync: "
        f"new={len(plan.new)}, changed={len(plan.changed)}, "
        f"unchanged={len(plan.unchanged)}, deleted={len(plan.deleted)}."
    )
    print(
        "Index update: "
        f"indexed={len(result.indexed_sources)}, "
        f"failed={len(result.failed_sources)}, "
        f"chunks_added={result.added_chunks}, "
        f"chunks_removed={result.deleted_chunks}."
    )
    for refresh_failure in plan.failed:
        print(f"  Inspection failed: {refresh_failure.path}: {refresh_failure.reason}")
    for source_file in result.failed_sources:
        print(f"  Refresh failed: {source_file}")


def _refresh_library(
    parser: argparse.ArgumentParser,
    *,
    library_root: Path,
    current_config_fingerprint: str,
    args: argparse.Namespace,
    split_config: SplitConfig,
) -> None:
    """Run the explicit write phase used by refresh and one-click startup."""
    refresh_service = build_default_embedding_service(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    try:
        result = refresh_library(
            library_root,
            current_config_fingerprint=current_config_fingerprint,
            embedding_service=refresh_service,
            split_config=split_config,
            extensions=args.extensions,
            index_name=args.index_name,
        )
    except LibraryRefreshLockedError as exc:
        parser.error(str(exc))
    _print_refresh_result(result)


def _is_one_click_launch(args: argparse.Namespace) -> bool:
    """Return whether the user requested the default sync-then-chat experience."""
    return not any(
        (
            args.init_library,
            args.refresh_library,
            args.load_index,
            args.save_index,
            args.query,
            args.chat,
        )
    )


def _require_active_library(parser: argparse.ArgumentParser, settings: LibrarySettings) -> Path:
    """Return the selected library or give a useful first-run command."""
    if settings.active_library is None:
        parser.error(
            "No default ResearchLite library is configured. "
            "Run `researchlite add <path>` to create one."
        )
    return settings.active_library


def _print_libraries(settings: LibrarySettings) -> None:
    """Show locally remembered libraries without loading models or indexes."""
    if not settings.libraries:
        print("No libraries configured. Run `researchlite add <path>` to create one.")
        return
    print("ResearchLite libraries:")
    for library in settings.libraries:
        marker = "*" if library == settings.active_library else " "
        print(f"{marker} {library}")


def _resolve_command(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[str, str | None]:
    """Interpret friendly library commands while keeping the old path shorthand."""
    command = args.command_or_path
    path = args.path
    if command is None:
        if path is not None:
            parser.error("A path must follow `add` or `use`.")
        return "open-default", None
    if command in {"add", "use"}:
        if path is None:
            parser.error(f"`researchlite {command}` requires a document directory.")
        return command, path
    if command in {"sync", "libraries"}:
        if path is not None:
            parser.error(f"`researchlite {command}` does not accept a path.")
        return command, None
    if path is not None:
        parser.error(f"Unrecognized command: {command!r}.")
    return "open-path", command


def _apply_llm_defaults(args: argparse.Namespace) -> LLMSettings | None:
    """Resolve CLI overrides against non-secret, local-first user settings."""
    settings = load_llm_settings()
    requested_provider = args.llm_provider
    args.llm_provider = requested_provider or settings.provider
    uses_saved_provider = requested_provider is None or requested_provider == settings.provider
    if args.llm_model is None:
        if uses_saved_provider:
            args.llm_model = settings.model
        elif args.llm_provider == "ollama":
            args.llm_model = DEFAULT_OLLAMA_MODEL
        else:
            args.llm_model = DEFAULT_OPENAI_MODEL
    if args.llm_base_url is None and uses_saved_provider:
        args.llm_base_url = settings.base_url
    if args.llm_provider != "ollama":
        return None
    return LLMSettings(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url or DEFAULT_OLLAMA_BASE_URL,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ResearchLite RAG pipeline locally.",
        epilog=(
            "Examples:\n"
            "  researchlite                    Open the default library.\n"
            "  researchlite add ~/papers       Create or sync a library, then chat.\n"
            "  researchlite use ~/papers       Switch to an existing library.\n"
            "  researchlite sync               Incrementally sync the default library.\n"
            "  researchlite libraries          List known libraries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command_or_path",
        nargs="?",
        metavar="command_or_path",
        help="Library command (add, use, sync, libraries) or a document path shorthand.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        metavar="path",
        help="Document path required by the add and use commands.",
    )
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
    query_mode = parser.add_mutually_exclusive_group()
    query_mode.add_argument(
        "--query",
        help="Search text against an existing .researchlite library.",
    )
    query_mode.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive terminal chat against an existing .researchlite library.",
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
        "--llm-provider",
        choices=("openai", "ollama"),
        help="LLM backend. Defaults to the saved local setting, or Ollama on first launch.",
    )
    parser.add_argument(
        "--llm-model",
        help=("LLM model name. Defaults to the saved local setting, or llama3.2 on first launch."),
    )
    parser.add_argument(
        "--llm-api-key",
        help="API key for an OpenAI-compatible service. Can also be set via OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--llm-base-url",
        help=("Base URL for the LLM service. Defaults to the saved local setting."),
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
    command, command_path = _resolve_command(parser, args)
    requested_one_click = _is_one_click_launch(args)
    try:
        library_settings = load_library_settings()
        effective_llm_settings = _apply_llm_defaults(args)
    except RuntimeError as exc:
        parser.error(str(exc))

    if command == "libraries":
        _print_libraries(library_settings)
        return

    remember_library = command in {"add", "use", "open-path"}
    if command == "open-default":
        args.path = str(_require_active_library(parser, library_settings))
        if requested_one_click:
            args.chat = True
    elif command == "add":
        if any(
            (
                args.init_library,
                args.refresh_library,
                args.load_index,
                args.save_index,
                args.query,
                args.chat,
            )
        ):
            parser.error("`researchlite add` cannot be combined with another library action.")
        args.path = command_path
    elif command == "use":
        if any(
            (
                args.init_library,
                args.refresh_library,
                args.load_index,
                args.save_index,
                args.query,
            )
        ):
            parser.error("`researchlite use` only opens an existing library for chat.")
        args.path = command_path
        args.chat = True
    elif command == "sync":
        if any(
            (
                args.init_library,
                args.refresh_library,
                args.load_index,
                args.save_index,
                args.query,
                args.chat,
            )
        ):
            parser.error("`researchlite sync` only synchronizes the default library.")
        args.path = str(_require_active_library(parser, library_settings))
        args.refresh_library = True
    else:
        args.path = command_path

    one_click_launch = command == "add" or (command == "open-path" and requested_one_click)

    split_config = SplitConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    current_config_fingerprint = config_fingerprint(
        model_name=args.model_name,
        split_config=split_config,
    )

    if args.init_library:
        library_root = resolve_library_root(args.path, args.library_dir)
        manifest = IngestionManifest.initialize(
            library_root,
            current_config_fingerprint=current_config_fingerprint,
        )
        try:
            print(f"Initialized ResearchLite library manifest at '{manifest.state_dir}'.")
        finally:
            manifest.close()
        return

    library_root = resolve_library_root(args.path, args.library_dir)

    if one_click_launch:
        print("ResearchLite · Local research assistant")
        if _initialize_library_if_needed(
            library_root,
            current_config_fingerprint=current_config_fingerprint,
        ):
            print(f"Created local library at '{library_root / '.researchlite'}'.")
        print("Syncing library...")
        _refresh_library(
            parser,
            library_root=library_root,
            current_config_fingerprint=current_config_fingerprint,
            args=args,
            split_config=split_config,
        )
        if effective_llm_settings is not None:
            save_llm_settings(effective_llm_settings)
        args.chat = True

    if args.chat and args.load_index:
        parser.error(
            "--chat uses a managed .researchlite library and cannot be combined with --load-index."
        )

    if args.refresh_library:
        if _initialize_library_if_needed(
            library_root,
            current_config_fingerprint=current_config_fingerprint,
        ):
            print(f"Created local library at '{library_root / '.researchlite'}'.")
        _refresh_library(
            parser,
            library_root=library_root,
            current_config_fingerprint=current_config_fingerprint,
            args=args,
            split_config=split_config,
        )
        return

    embedded: list[EmbeddedDocument] = []
    vector_store: FaissVectorStore | None = None
    service: EmbeddingService | None = None
    has_persisted_library = (library_root / ".researchlite" / "manifest.sqlite").is_file()

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
    elif args.query or args.chat:
        if not has_persisted_library:
            parser.error(
                f"No ResearchLite library found at '{library_root / '.researchlite'}'. "
                "Run --refresh-library before querying, or run `researchlite <path>` once "
                "to create and synchronize it."
            )
        service = build_default_embedding_service(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        try:
            vector_store = load_persisted_library(
                library_root,
                current_config_fingerprint=current_config_fingerprint,
                embedding_service=service,
                index_name=args.index_name,
            )
            print(f"Loaded ResearchLite library index '{args.index_name}' from '{library_root}'.")
        except (FileNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))
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

    if args.query or args.chat:
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
            if args.chat and reranker is not None:
                print("Preparing local retrieval model...")
                try:
                    reranker.warm_up()
                except RerankerUnavailableError:
                    print("Reranker unavailable; chat will use hybrid retrieval.")
                    retrieval_mode = "hybrid"
                    reranker = None
            retrieval_documents = (
                [item.document for item in embedded] if embedded else vector_store.documents()
            )
            session = LibraryQuerySession(
                vector_store,
                service,
                retrieval_documents=retrieval_documents,
                retrieval_mode=retrieval_mode,
                top_k=args.top_k,
                candidate_k=args.retrieval_candidates,
                rrf_constant=args.rrf_constant,
                reranker=reranker,
            )
            if remember_library and has_persisted_library and not args.load_index:
                save_active_library(library_root)
            if args.chat:
                generator: RAGGenerator | None = None
                if not args.no_generation:
                    try:
                        generator = RAGGenerator(
                            model_name=args.llm_model,
                            api_key=args.llm_api_key or os.environ.get("OPENAI_API_KEY"),
                            base_url=args.llm_base_url,
                            provider=args.llm_provider,
                        )
                    except Exception as exc:
                        parser.error(str(exc))
                from research_lite.repl import TerminalChat

                library_stats = (
                    inspect_library(library_root, vector_store)
                    if has_persisted_library and not args.load_index
                    else None
                )
                TerminalChat(
                    session,
                    generate_answer=generator.generate_answer if generator else None,
                    library_stats=library_stats,
                ).run()
                return

            print(f"Running {retrieval_mode} search for query: {args.query!r}")
            outcome = session.query(args.query)
            results = outcome.documents
            if outcome.reranker_unavailable:
                print("Reranker unavailable; falling back to hybrid retrieval.")
            if not results:
                print("No results returned from retrieval.")
            else:
                print(f"Found {len(results)} relevant evidence chunks:")
                for evidence in format_evidence(results):
                    print(evidence)

                if not args.no_generation:
                    print("\nGenerating answer...")
                    try:
                        # Ensure we pass the cleaned key if it wasn't passed via args
                        final_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")

                        generator = RAGGenerator(
                            model_name=args.llm_model,
                            api_key=final_api_key,
                            base_url=args.llm_base_url,
                            provider=args.llm_provider,
                        )
                        answer = generator.generate_answer(args.query, results)
                        print(f"\nAnswer:\n{answer}")
                    except Exception as e:
                        print(f"Failed to generate answer: {e}")
                        if args.llm_provider == "ollama":
                            print(
                                "Tip: Start Ollama with `ollama serve` and pull the selected model."
                            )
                        else:
                            print(
                                "Tip: Ensure you have set OPENAI_API_KEY or passed --llm-api-key."
                            )


if __name__ == "__main__":
    main()
