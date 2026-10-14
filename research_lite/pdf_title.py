#
# pdf_title.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-14.
#

"""Work out a PDF's real title from its own metadata or from how its first page is set."""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pypdfium2.raw as pdfium_raw

from research_lite.utils.loader_utils import usable_title

if TYPE_CHECKING:
    import pypdfium2 as pdfium

MIN_LENGTH = 12
MAX_LENGTH = 300
UPPER_PAGE_SHARE = 0.45
BAND_TOLERANCE = 0.8
TOP_MARGIN = 0.2
BOTTOM_MARGIN = 0.1
MIN_BAND_LENGTH = 4
MIN_PAGE_OVERLAP = 0.6

_WORD = re.compile(r"[a-z0-9]+")
_LATEX = re.compile(r"\\[a-zA-Z]+|[${}\\]")
_STAMP = re.compile(
    r"arxiv:\s*\d{4}\.\d{4,5}|biorxiv|medrxiv|doi:|preprint|licen[sc]e|copyright"
    r"|all rights reserved|submitted to|accepted (?:at|for)|published (?:as|in)"
    r"|under review|www\.|https?://",
    re.I,
)
_SECTION = re.compile(r"^\s*(?:abstract|introduction|keywords|summary)\b", re.I)
_AUTHORS = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][a-z]+(?:,|\s+and\b|\s*$)")
_INLINE_NUMBER = re.compile(r"(?<=[A-Za-z])\s+\d{1,3}\s+(?=[A-Za-z])")
_LEADING_NUMBER = re.compile(r"^\d{1,3}\s+(?=[A-Za-z])")


@dataclass(frozen=True)
class _Run:
    text: str
    size: float
    left: float
    top: float
    bottom: float


def normalise(text: str) -> str:
    """Reduce a title to comparable content words, ignoring LaTeX markup and punctuation."""
    return " ".join(_WORD.findall(_LATEX.sub(" ", text).lower()))


def looks_like_title(text: str) -> bool:
    """Whether a candidate reads like a paper title rather than a banner or an author line."""
    return bool(
        text
        and usable_title(text, "")
        and MIN_LENGTH <= len(text) <= MAX_LENGTH
        and not _STAMP.search(text)
        and not _SECTION.match(text)
        and not _AUTHORS.match(text)
        and sum(character.isalpha() for character in text) / len(text) > 0.5
    )


def page_title(page: pdfium.PdfPage, declared: object, page_text: str, fallback: str) -> str:
    """Return the page's title, preferring metadata the page itself confirms."""
    confirmed = _confirmed_metadata(declared, page_text)
    if confirmed:
        return confirmed
    return title_from_layout(page) or fallback


def _confirmed_metadata(declared: object, page_text: str) -> str:
    """Accept a declared title when the first page largely repeats it.

    A real title is set on the page it belongs to, so most of its words appear there. A file
    name, an arXiv identifier or a journal template name shares almost nothing with the page.
    """
    text = _clean(str(declared or ""))
    if not looks_like_title(text):
        return ""
    wanted = set(normalise(text).split())
    if not wanted:
        return ""
    present = wanted & set(normalise(page_text).split())
    return text if len(present) / len(wanted) >= MIN_PAGE_OVERLAP else ""


def title_from_layout(page: pdfium.PdfPage) -> str:
    """Read the largest-font opening block of the page, as the page itself lays it out."""
    runs = _text_runs(page)
    if not runs:
        return ""
    highest = max(run.top for run in runs)
    lowest = min(run.top for run in runs)
    cut = lowest + (highest - lowest) * UPPER_PAGE_SHARE
    pool = [run for run in runs if run.top >= cut and not _STAMP.search(run.text)] or runs

    bands: list[list[float]] = []
    for run in sorted(pool, key=lambda item: (-item.top, item.left)):
        if bands and abs(run.top - bands[-1][2]) <= max(3.0, bands[-1][1] * BAND_TOLERANCE):
            band = bands[-1]
            band[1] = max(band[1], run.size)
            band[2] = min(band[2], run.top)
            band[3] = min(band[3], run.bottom)
            band[4] += len(run.text)
            continue
        bands.append([run.top, run.size, run.top, run.bottom, float(len(run.text))])

    readable = [band for band in bands if band[4] >= MIN_BAND_LENGTH]
    if not readable:
        return ""
    biggest = max(band[1] for band in readable)
    chosen: list[list[float]] = []
    for band in readable:
        if band[1] >= biggest - 0.3:
            chosen.append(band)
        elif chosen:
            break
    if not chosen:
        return ""

    top = max(band[0] for band in chosen) + biggest * TOP_MARGIN
    bottom = min(band[3] for band in chosen) - biggest * BOTTOM_MARGIN
    try:
        inside = page.get_textpage().get_text_bounded(
            left=0, bottom=bottom, right=page.get_width(), top=top
        )
    except Exception:
        return ""
    candidate = _clean(inside)
    return candidate if looks_like_title(candidate) else ""


def _clean(text: str) -> str:
    """Collapse whitespace and drop the line numbers some preprint servers print inline."""
    collapsed = " ".join(text.split())
    collapsed = _LEADING_NUMBER.sub("", collapsed)
    collapsed = _INLINE_NUMBER.sub(" ", collapsed)
    return collapsed[:MAX_LENGTH].strip()


def _text_runs(page: pdfium.PdfPage) -> list[_Run]:
    """Collect the page's upright text runs with their font size and bounding box."""
    try:
        textpage = page.get_textpage()
    except Exception:
        return []
    runs: list[_Run] = []
    try:
        objects = list(page.get_objects())
    except Exception:
        return []
    for obj in objects:
        if obj.type != pdfium_raw.FPDF_PAGEOBJ_TEXT:
            continue
        try:
            matrix = obj.get_matrix()
            if abs(matrix.b) > 0.01 or abs(matrix.c) > 0.01:
                continue
            text = " ".join(_object_text(obj, textpage).split())
            if not text:
                continue
            left, bottom, _, top = obj.get_bounds()
            runs.append(
                _Run(
                    text=text,
                    size=round(obj.get_font_size(), 1),
                    left=left,
                    top=top,
                    bottom=bottom,
                )
            )
        except Exception:
            continue
    return runs


def _object_text(obj: object, textpage: object) -> str:
    handle, text_handle = obj.raw, textpage.raw  # type: ignore[attr-defined]
    length = pdfium_raw.FPDFTextObj_GetText(handle, text_handle, None, 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_string_buffer(length * 2)
    pdfium_raw.FPDFTextObj_GetText(
        handle, text_handle, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)), length
    )
    return buffer.raw.decode("utf-16-le", errors="replace").rstrip("\x00")


__all__ = ["looks_like_title", "normalise", "page_title", "title_from_layout"]
