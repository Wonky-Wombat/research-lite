#
# library_stats.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Read-only summary data for a persisted ResearchLite library."""

from __future__ import annotations

from collections.abc import Sequence
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
    titles: tuple[str, ...] = ()
    matching: tuple[str, ...] | None = None

    @property
    def containing(self) -> int | None:
        return None if self.matching is None else len(self.matching)

    def overview(self, terms: Sequence[str] = (), *, titles: bool = False) -> dict[str, object]:
        """The whole-library view, as JSON-ready data rather than prose.

        The title list is the bulk of this payload — 1,929 of 2,013 tokens for a hundred
        documents — so it is sent only when it answers the question. Given terms, the
        titles that match are what "which papers are about X" needs; without them, naming
        every document in the library is a cost with no question behind it.
        """
        view: dict[str, object] = {
            "documents": self.indexed_sources,
            "chunks": self.chunks,
            "failed": self.failed_sources,
            "terms": list(terms),
            "documents_containing_all_terms": self.containing,
            "recorded": ["title", "text", "page"],
            "not_recorded": ["publication date", "author", "venue"],
        }
        if self.matching is not None:
            view["matching_titles"] = list(self.matching)
        elif titles:
            view["titles"] = list(self.titles)
        else:
            view["titles_available"] = len(self.titles)
        return view

    def display(self) -> str:
        """Return a short status line suitable for terminal output."""
        failure_suffix = f" · {self.failed_sources} failed" if self.failed_sources else ""
        return f"Library: {self.indexed_sources} papers · {self.chunks} chunks{failure_suffix}."


def inspect_library(
    library_root: Path, index: LibraryIndex, *, terms: Sequence[str] = ()
) -> LibraryStats:
    """Read manifest counts and the loaded index's chunk count without writing state."""
    manifest = IngestionManifest.open(library_root)
    try:
        records = manifest.list_sources()
        matched = manifest.sources_containing(terms) if terms else None
    finally:
        manifest.close()
    documents = index.documents()
    titles: dict[str, str] = {}
    for document in documents:
        source_path = str(document.metadata.get("source_path", ""))
        if source_path and source_path not in titles:
            titles[source_path] = str(document.metadata.get("title", "")) or Path(source_path).name
    return LibraryStats(
        indexed_sources=sum(record.has_indexed_content for record in records),
        failed_sources=sum(record.last_refresh_error is not None for record in records),
        chunks=len(documents),
        titles=tuple(sorted(titles.values(), key=str.casefold)),
        matching=(
            None
            if matched is None
            else tuple(
                sorted((titles.get(path, Path(path).name) for path in matched), key=str.casefold)
            )
        ),
    )
