#
# running_text.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-13.
#

"""Remove the running headers and page numbers that repeat on a document's pages."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

RUNNING_TEXT_RULES = "3"

MIN_REPEATS = 3
MIN_SHARE = 0.5
MIN_HEADER_LENGTH = 6
MAX_HEADER_LENGTH = 200
MAX_HEADER_VARIANTS = 3

_DECORATION = " \t-–—.,:;()[]|*#"
_PAGE_NUMBER = re.compile(r"^\d{1,4}(?:\s*/\s*\d{1,4})?$")


def running_headers(pages: Sequence[str]) -> list[str]:
    """Return the lines that open a document's pages, when a few of them cover most pages."""
    if len(pages) < MIN_REPEATS:
        return []
    openers = Counter(
        line
        for line in (_first_line(page) for page in pages)
        if MIN_HEADER_LENGTH <= len(line) <= MAX_HEADER_LENGTH
    )
    repeated = [
        (line, repeats)
        for line, repeats in openers.most_common(MAX_HEADER_VARIANTS)
        if repeats >= MIN_REPEATS
    ]
    covered = sum(repeats for _, repeats in repeated)
    if not repeated or covered < MIN_SHARE * len(pages):
        return []
    return [line for line, _ in repeated]


def has_page_number_footers(pages: Sequence[str]) -> bool:
    """Whether most of a document's pages end in a bare page number."""
    if len(pages) < MIN_REPEATS:
        return False
    numbered = sum(_is_page_number(_last_line(page)) for page in pages)
    return _repeats_enough(numbered, len(pages))


def strip_running_text(pages: Sequence[str]) -> list[str]:
    """Drop each page's repeated opening line and trailing page number."""
    headers = frozenset(running_headers(pages))
    footers = has_page_number_footers(pages)
    if not headers and not footers:
        return list(pages)
    return [_strip_page(page, headers=headers, footer=footers) for page in pages]


def _is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line.strip(_DECORATION)))


def _repeats_enough(repeats: int, pages: int) -> bool:
    return repeats >= MIN_REPEATS and repeats >= MIN_SHARE * pages


def _collapse(line: str) -> str:
    return " ".join(line.split())


def _first_line(page: str) -> str:
    for line in page.split("\n"):
        collapsed = _collapse(line)
        if collapsed:
            return collapsed
    return ""


def _last_line(page: str) -> str:
    for line in reversed(page.split("\n")):
        collapsed = _collapse(line)
        if collapsed:
            return collapsed
    return ""


def _strip_page(page: str, *, headers: frozenset[str], footer: bool) -> str:
    lines = page.split("\n")
    if headers:
        lines = _without(lines, lambda line: line in headers)
    if footer:
        lines = _without(lines[::-1], _is_page_number)[::-1]
    return "\n".join(lines).strip()


def _without(lines: list[str], matches: Callable[[str], bool]) -> list[str]:
    for index, line in enumerate(lines):
        collapsed = _collapse(line)
        if not collapsed:
            continue
        return lines[index + 1 :] if matches(collapsed) else lines
    return lines


__all__ = [
    "MAX_HEADER_VARIANTS",
    "MIN_REPEATS",
    "MIN_SHARE",
    "RUNNING_TEXT_RULES",
    "has_page_number_footers",
    "running_headers",
    "strip_running_text",
]
