#
# inline_citations.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-14.
#

"""Remove the citation markers a document's own text carries, so they cannot be mistaken for
the library's own documents."""

from __future__ import annotations

import re

_YEAR = r"(?:1[89]|20)\d{2}[a-z]?(?:\s*,\s*[a-z]\b)*"
_SURNAME = r"(?:[A-Z][\w'’-]+|[A-Z]{2,})"
_ET_AL = r"(?:et|ET)\s+(?:al|AL)\.?"
_JOIN = r"(?:&|and|AND)"

_NAMES = (
    rf"{_SURNAME}"
    rf"(?:\s*,\s*{_SURNAME}){{0,3}}"
    rf"(?:\s*,?\s*{_JOIN}\s*{_SURNAME})?"
)
_ITEM = rf"{_NAMES}(?:\s+{_ET_AL})?(?:'s)?[\s,]*{_YEAR}"
_LEAD = r"(?:(?:e\.g\.|i\.e\.|cf\.|see(?:\s+also)?|also|in)[,:]?\s*)?"

_PARENTHETICAL = re.compile(
    rf"\(\s*{_LEAD}{_ITEM}(?:\s*[;,]\s*{_LEAD}(?:{_ITEM}|{_YEAR}))*\s*\)"
)
_NARRATIVE = re.compile(
    rf"\b(?:{_SURNAME})"
    rf"(?:\s*(?:,\s*)?(?:{_JOIN})\s*(?:{_SURNAME}))?"
    rf"\s+{_ET_AL}\s*\(\s*{_YEAR}(?:\s*[;,]\s*{_YEAR})*\s*\)"
)
_BRACKET = r"\[\s*\d{1,3}(?:\s*[,;–—-]\s*\d{1,3})*\s*\]"
_NUMERIC = re.compile(rf"{_BRACKET}(?:\s*[,;]?\s*{_BRACKET})*")

_SPACE_BEFORE_MARK = re.compile(r"\s+([,.;:!?)\]])")
_EMPTY_GROUP = re.compile(r"\(\s*[,;]?\s*\)|\[\s*[,;]?\s*\]")
_DOUBLED_MARK = re.compile(r"([,;:])\s*(?=[,.;:])")


def strip_inline_citations(text: str) -> str:
    """Drop author-year and bracketed-number citations, leaving the surrounding prose intact."""
    without = _NUMERIC.sub("", _PARENTHETICAL.sub("", _NARRATIVE.sub("", text)))
    return _tidy(without)


def _tidy(text: str) -> str:
    """Close the gaps a removal leaves behind, without touching line structure."""
    text = _EMPTY_GROUP.sub("", text)
    text = _DOUBLED_MARK.sub("", text)
    text = _SPACE_BEFORE_MARK.sub(r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text)


__all__ = ["strip_inline_citations"]
