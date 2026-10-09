#
# library_stats.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Read-only summary data for a persisted ResearchLite library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_lite.manifest import IngestionManifest
from research_lite.vectorstore import LibraryIndex


@dataclass(frozen=True)
class LibraryStats:
    """Counts that describe the current state of a local document library."""

    indexed_sources: int
    failed_sources: int
    chunks: int

    def display(self) -> str:
        """Return a short status line suitable for terminal output."""
        failure_suffix = f" · {self.failed_sources} failed" if self.failed_sources else ""
        return f"Library: {self.indexed_sources} papers · {self.chunks} chunks{failure_suffix}."


def inspect_library(library_root: Path, index: LibraryIndex) -> LibraryStats:
    """Read manifest counts and the loaded index's chunk count without writing state."""
    manifest = IngestionManifest.open(library_root)
    try:
        records = manifest.list_sources()
    finally:
        manifest.close()
    return LibraryStats(
        indexed_sources=sum(record.has_indexed_content for record in records),
        failed_sources=sum(record.last_refresh_error is not None for record in records),
        chunks=len(index.documents()),
    )
