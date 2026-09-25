#
# library_refresh.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-25.
#

"""Apply safe, incremental updates to a ResearchLite document library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from research_lite.embedding import EmbeddedDocument, EmbeddingService
from research_lite.ingestion import SUPPORTED_EXTENSIONS, IngestReport, load_documents
from research_lite.manifest import IngestionManifest
from research_lite.preprocessing import SplitConfig, split_documents
from research_lite.refresh_plan import RefreshPlan, plan_library_refresh
from research_lite.utils.loader_utils import build_source_metadata
from research_lite.vectorstore import FaissVectorStore

DEFAULT_INDEX_NAME = "index"


@dataclass
class RefreshResult:
    """The outcome of applying a library refresh plan."""

    plan: RefreshPlan
    indexed_sources: list[Path] = field(default_factory=list)
    failed_sources: list[Path] = field(default_factory=list)
    deleted_chunks: int = 0
    added_chunks: int = 0


def refresh_library(
    library_root: Path,
    *,
    current_config_fingerprint: str,
    embedding_service: EmbeddingService,
    split_config: SplitConfig,
    extensions: list[str] | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> RefreshResult:
    """Synchronize a library's persisted FAISS index and source manifest.

    Changed chunks remain available if loading or embedding their replacement fails.
    The local index is trusted because it is loaded only from the library's own
    ``.researchlite`` directory.
    """
    root = library_root.expanduser().resolve()
    normalized_extensions = extensions or list(SUPPORTED_EXTENSIONS)
    manifest = IngestionManifest.open(root)
    try:
        plan = plan_library_refresh(
            root,
            manifest,
            current_config_fingerprint=current_config_fingerprint,
            extensions=normalized_extensions,
        )
        result = RefreshResult(plan=plan)
        pending_sources = [*plan.new, *plan.changed]
        if not pending_sources and not plan.deleted:
            return result

        existing_records = manifest.list_sources()
        index_path = manifest.state_dir / f"{index_name}.faiss"
        vector_store: FaissVectorStore | None = None
        if any(record.status == "indexed" for record in existing_records):
            if not index_path.is_file():
                msg = f"Missing FAISS index at {index_path}; cannot safely refresh this library."
                raise FileNotFoundError(msg)
            vector_store = FaissVectorStore.load(
                manifest.state_dir,
                embedding_backend=embedding_service.backend,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

        replacements: list[EmbeddedDocument] = []
        successful_changed: list[Path] = []
        successful_sources: list[tuple[Path, str]] = []
        for source_file in pending_sources:
            report = IngestReport()
            try:
                documents = load_documents(
                    str(source_file), extensions=normalized_extensions, report=report
                )
                if report.failures:
                    raise RuntimeError(report.failures[0].reason)
                chunks = split_documents(documents, config=split_config)
                replacements.extend(embedding_service.embed_documents(chunks))
                source_id = build_source_metadata(source_file)["source_id"]
                successful_sources.append((source_file, source_id))
                if source_file in plan.changed:
                    successful_changed.append(source_file)
            except Exception:
                result.failed_sources.append(source_file)

        removal_paths = [*plan.deleted, *successful_changed]
        if vector_store is not None and removal_paths:
            result.deleted_chunks = vector_store.delete_by_source_paths(removal_paths)
        if replacements:
            if vector_store is None:
                vector_store = FaissVectorStore.from_documents(
                    replacements, embedding_backend=embedding_service.backend
                )
                result.added_chunks = len(replacements)
            else:
                vector_store.add(replacements)
                result.added_chunks = len(replacements)
        if vector_store is not None and (removal_paths or replacements):
            vector_store.save(manifest.state_dir, index_name=index_name)

        for source_file in plan.deleted:
            manifest.remove_source(source_file)
        for source_file, source_id in successful_sources:
            manifest.upsert_source(
                path=source_file,
                source_id=source_id,
                config_fingerprint=current_config_fingerprint,
            )
            result.indexed_sources.append(source_file)
        for source_file in result.failed_sources:
            try:
                source_id = build_source_metadata(source_file)["source_id"]
            except OSError:
                continue
            manifest.upsert_source(
                path=source_file,
                source_id=source_id,
                config_fingerprint=current_config_fingerprint,
                status="failed",
                error_message="Could not load, split, or embed this source during refresh.",
            )

        if plan.configuration_changed and not result.failed_sources and not plan.failed:
            manifest.set_state_value("config_fingerprint", current_config_fingerprint)
        return result
    finally:
        manifest.close()


__all__ = ["DEFAULT_INDEX_NAME", "RefreshResult", "refresh_library"]
