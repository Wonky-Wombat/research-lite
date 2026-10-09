#
# citations.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Stable, user-facing citations for retrieved document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from research_lite import Document


@dataclass(frozen=True)
class CitationSource:
    """One source paper and the number used to cite it in a response."""

    number: int
    title: str
    source_path: str

    @property
    def filename(self) -> str:
        """Return a concise, portable source label."""
        return Path(self.source_path).name if self.source_path else "Unknown"


def citation_sources(documents: Sequence[Document]) -> list[CitationSource]:
    """Return source papers in their first-retrieved order, deduplicated by path."""
    sources: list[CitationSource] = []
    seen_keys: set[str] = set()
    for document in documents:
        metadata = document.metadata
        source_path = str(metadata.get("source_path", ""))
        title = str(metadata.get("title", "Unknown"))
        key = source_path or title
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sources.append(
            CitationSource(number=len(sources) + 1, title=title, source_path=source_path)
        )
    return sources


def citation_number_by_source(documents: Sequence[Document]) -> dict[str, int]:
    """Return the stable citation number for each retrieved document source."""
    return {
        source.source_path or source.title: source.number for source in citation_sources(documents)
    }


def page_number(metadata: Mapping[str, object]) -> int | None:
    source_unit = str(metadata.get("source_unit", ""))
    page = source_unit.removeprefix("page-")
    if not source_unit.startswith("page-") or not page.isdigit():
        return None
    return int(page) + 1


def page_label(metadata: Mapping[str, object]) -> str:
    page = page_number(metadata)
    return "" if page is None else f", page {page}"
