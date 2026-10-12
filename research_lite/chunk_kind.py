#
# chunk_kind.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-10.
#

"""Tell a document's own prose apart from its reference list and submission forms."""

from __future__ import annotations

import re
from collections.abc import Sequence

BODY = "body"
REFERENCES = "references"
BOILERPLATE = "boilerplate"

_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]+|[IVX]+\.?[ \t]+)?"
    r"(?:references?|bibliography|works cited|literature cited)[ \t]*$",
    re.I | re.M,
)
_NUMBERED = re.compile(r"^\s*\[(\d{1,3})\]\s+\S", re.M)
_ENTRY = re.compile(
    r"(?:,|\.)\s*(?:19|20)\d{2}[a-z]?[.,]"
    r"|\((?:19|20)\d{2}[a-z]?\)[.,]?"
    r"|\s(?:19|20)\d{2},\s"
    r"|^\s*\[\d{1,3}\]\s",
    re.M,
)
_VENUE = re.compile(
    r"arXiv|In Proceedings|Advances in Neural|Conference on|Journal of|Proc\.|"
    r"Phys\.\s?Rev|ApJ|MNRAS|Nature|Science\b|doi|pp\.\s*\d|vol\.\s*\d|"
    r"Econometric|Review of|Transactions on",
    re.I,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_SENTENCE_END = re.compile(r"[.!?]\s")
_BOILERPLATE = re.compile(
    r"The answer NA means|Answer:\s*\[(?:Yes|No|NA)\]|Justification:|"
    r"(?:NeurIPS|ICLR|ICML)\s+(?:Paper\s+)?Checklist|Guidelines:\s*[•\-]",
    re.I,
)
_CITATION_QUESTION = re.compile(
    r"\b(?:cite[sd]?|citing|citations?|references?|referenced|bibliograph\w*)\b", re.I
)

_MIN_ENTRIES = 3
_MIN_VENUES = 2
_MIN_NUMBERED = 5
_MIN_YEARS_PER_1K = 4.0
_MIN_CAPITALIZED = 0.35
_MIN_SENTENCE_ENDS_PER_1K = 14.0


def reference_start_unit(pages: Sequence[str]) -> int | None:
    """Return the index of the unit where a source's reference list begins."""
    for index, page in enumerate(pages):
        if _HEADING.search(page):
            return index
    for index, page in enumerate(pages):
        if _opens_numbered_bibliography(page):
            return index
    return None


def classify_chunk(text: str, *, after_references: bool) -> str:
    """Classify one chunk, given whether it sits inside its source's reference list."""
    if len(_BOILERPLATE.findall(text)) >= 2:
        return BOILERPLATE
    if after_references and _reads_like_citations(text):
        return REFERENCES
    return BODY


def asks_for_citations(query: str) -> bool:
    """Whether a query is itself about citations, so reference lists are the answer."""
    return bool(_CITATION_QUESTION.search(query))


def _opens_numbered_bibliography(page: str) -> bool:
    numbers = [int(match.group(1)) for match in _NUMBERED.finditer(page)]
    if len(numbers) < _MIN_NUMBERED:
        return False
    rising = sum(later > earlier for earlier, later in zip(numbers, numbers[1:], strict=False))
    return numbers[0] <= 2 and rising >= len(numbers) - 2


def _reads_like_citations(text: str) -> bool:
    if len(_ENTRY.findall(text)) >= _MIN_ENTRIES and len(_VENUE.findall(text)) >= _MIN_VENUES:
        return True
    words = text.split()
    if not words:
        return False
    per_1k = 1000 / len(text)
    return (
        len(_YEAR.findall(text)) * per_1k >= _MIN_YEARS_PER_1K
        and sum(word[:1].isupper() for word in words) / len(words) >= _MIN_CAPITALIZED
        and len(_SENTENCE_END.findall(text)) * per_1k >= _MIN_SENTENCE_ENDS_PER_1K
    )


__all__ = [
    "BODY",
    "BOILERPLATE",
    "REFERENCES",
    "asks_for_citations",
    "classify_chunk",
    "reference_start_unit",
]
