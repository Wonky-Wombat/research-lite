#
# library_refresh.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-25.
#

"""Apply safe, incremental updates to a ResearchLite document library."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from research_lite.embedding import EmbeddedDocument, EmbeddingService
from research_lite.ingestion import (
    SUPPORTED_EXTENSIONS,
    IngestReport,
    load_documents,
    read_title,
)
from research_lite.library_backup import RefreshBackup
from research_lite.library_lock import acquire_refresh_lock
from research_lite.manifest import IngestionManifest
from research_lite.preprocessing import SplitConfig, split_documents
from research_lite.refresh_plan import RefreshPlan, plan_library_refresh
from research_lite.utils.loader_utils import build_source_metadata

TITLE_RULES = "1"


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
    load_embedding_service: Callable[[], EmbeddingService],
    split_config: SplitConfig,
    extensions: list[str] | None = None,
) -> RefreshResult:
    """Synchronize a library while holding an exclusive refresh lock."""
    root = library_root.expanduser().resolve()
    state_dir = root / ".researchlite"
    normalized_extensions = extensions or list(SUPPORTED_EXTENSIONS)
    preflight_manifest = IngestionManifest.open(root)
    preflight_manifest.close()
    with acquire_refresh_lock(state_dir):
        RefreshBackup.recover_interrupted_refresh(state_dir)
        manifest = IngestionManifest.open(root)
        try:
            plan = plan_library_refresh(
                root,
                manifest,
                current_config_fingerprint=current_config_fingerprint,
                extensions=normalized_extensions,
            )
        finally:
            manifest.close()
        result = RefreshResult(plan=plan)
        if plan.new or plan.changed or plan.deleted:
            backup = RefreshBackup.create(state_dir)
            try:
                result = _apply_refresh_plan(
                    root,
                    plan,
                    current_config_fingerprint=current_config_fingerprint,
                    embedding_service=load_embedding_service(),
                    split_config=split_config,
                    extensions=normalized_extensions,
                )
            except BaseException:
                backup.restore()
                raise
            backup.complete()
        _refresh_titles(root, plan.unchanged)
        return result


def _refresh_titles(root: Path, sources: list[Path]) -> None:
    """Re-read titles of already indexed sources once after the title rules change."""
    manifest = IngestionManifest.open(root)
    try:
        if manifest.state_value("title_rules") == TITLE_RULES:
            return
        for source in sources:
            try:
                manifest.set_chunk_title(source, read_title(source))
            except Exception:
                continue
        manifest.set_state_value("title_rules", TITLE_RULES)
    finally:
        manifest.close()


def _apply_refresh_plan(
    root: Path,
    plan: RefreshPlan,
    *,
    current_config_fingerprint: str,
    embedding_service: EmbeddingService,
    split_config: SplitConfig,
    extensions: list[str],
) -> RefreshResult:
    """Synchronize a library's stored chunks and source manifest.

    Changed chunks remain available if loading or embedding their replacement fails.
    """
    manifest = IngestionManifest.open(root)
    try:
        result = RefreshResult(plan=plan)
        pending_sources = [*plan.new, *plan.changed]
        rebuild = plan.configuration_changed
        replacements: list[EmbeddedDocument] = []
        successful_changed: list[Path] = []
        successful_sources: list[tuple[Path, str]] = []
        failed_source_messages: list[tuple[Path, str]] = []
        for source_file in pending_sources:
            report = IngestReport()
            try:
                documents = load_documents(str(source_file), extensions=extensions, report=report)
                if report.failures:
                    raise RuntimeError(report.failures[0].reason)
                chunks = split_documents(documents, config=split_config)
                replacements.extend(embedding_service.embed_documents(chunks))
                source_id = build_source_metadata(source_file)["source_id"]
                successful_sources.append((source_file, source_id))
                if source_file in plan.changed:
                    successful_changed.append(source_file)
            except Exception as exc:
                result.failed_sources.append(source_file)
                failed_source_messages.append((source_file, str(exc)))

        if rebuild:
            manifest.clear_chunks()
        else:
            result.deleted_chunks = manifest.delete_chunks([*plan.deleted, *successful_changed])
        manifest.add_chunks(replacements)
        result.added_chunks = len(replacements)

        for source_file in plan.deleted:
            manifest.remove_source(source_file)
        for source_file, source_id in successful_sources:
            manifest.record_indexed_source(
                path=source_file,
                source_id=source_id,
                config_fingerprint=current_config_fingerprint,
            )
            result.indexed_sources.append(source_file)
        for source_file, error_message in failed_source_messages:
            try:
                source_id = str(build_source_metadata(source_file)["source_id"])
            except OSError:
                source_id = None
            manifest.record_refresh_failure(
                path=source_file,
                source_id=source_id,
                error_message=error_message or "Could not load, split, or embed this source.",
                keep_indexed_version=not rebuild,
            )

        if rebuild:
            for failure in plan.failed:
                manifest.record_refresh_failure(
                    path=failure.path,
                    source_id=None,
                    error_message=failure.reason,
                    keep_indexed_version=False,
                )
            manifest.set_state_value("config_fingerprint", current_config_fingerprint)
        return result
    finally:
        manifest.close()


__all__ = ["RefreshResult", "refresh_library"]
