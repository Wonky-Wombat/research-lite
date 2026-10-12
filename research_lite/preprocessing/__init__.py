#
# __init__.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from research_lite import Document
from research_lite.chunk_kind import classify_chunk, reference_start_unit

from ..utils.loader_utils import _sha1

ZERO_WIDTH = ["\ufeff", "\u200b", "\u200c", "\u200d"]


def _normalize_text(text: str) -> str:
    """Apply lightweight normalization to reduce noise before splitting."""
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    for zw in ZERO_WIDTH:
        text = text.replace(zw, "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _default_separators_by_ext() -> dict[str, tuple[str, ...]]:
    markdown_separators = ("\n## ", "\n### ", "<br/>", "<br>", "\n\n", "\n", " ")
    html_separators = ("</p>", "</div>", "<br/>", "<br>", "\n", " ")
    docx_separators = ("\n\n", "\n", " ")

    return {
        "md": markdown_separators,
        "markdown": markdown_separators,
        "html": html_separators,
        "htm": html_separators,
        "docx": docx_separators,
        "doc": docx_separators,
    }


@dataclass
class SplitConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    separators: tuple[str, ...] = ("\n\n", "\n", " ")
    separators_by_ext: dict[str, tuple[str, ...]] = field(
        default_factory=_default_separators_by_ext
    )
    keep_separator: bool = False


def _select_separators(ext: str, config: SplitConfig) -> tuple[str, ...]:
    normalized = ext.lower()
    return config.separators_by_ext.get(normalized, config.separators)


def _split_text(text: str, separators: tuple[str, ...], size: int, overlap: int) -> list[str]:
    separator = separators[-1]
    remaining: tuple[str, ...] = ()
    for index, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator, remaining = candidate, separators[index + 1 :]
            break
    pieces = [piece for piece in (text.split(separator) if separator else list(text)) if piece]
    chunks: list[str] = []
    pending: list[str] = []
    for piece in pieces:
        if len(piece) < size:
            pending.append(piece)
            continue
        chunks.extend(_merge_pieces(pending, separator, size, overlap))
        pending = []
        chunks.extend(_split_text(piece, remaining, size, overlap) if remaining else [piece])
    chunks.extend(_merge_pieces(pending, separator, size, overlap))
    return chunks


def _merge_pieces(pieces: list[str], separator: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    total = 0
    for piece in pieces:
        if window and total + len(piece) + len(separator) > size:
            chunks.append(separator.join(window).strip())
            while total > overlap or (total > 0 and total + len(piece) + len(separator) > size):
                total -= len(window.pop(0)) + (len(separator) if window else 0)
        total += len(piece) + (len(separator) if window else 0)
        window.append(piece)
    if window:
        chunks.append(separator.join(window).strip())
    return [chunk for chunk in chunks if chunk]


def _reference_starts(documents: list[Document]) -> dict[str, int]:
    """Find where each source's reference list starts, keeping unit order per source."""
    pages_by_source: dict[str, list[tuple[int, str]]] = {}
    for document in documents:
        source = str(document.metadata.get("source_path", ""))
        unit = _unit_index(document)
        if unit is not None:
            pages_by_source.setdefault(source, []).append((unit, document.page_content))
    starts: dict[str, int] = {}
    for source, pages in pages_by_source.items():
        ordered = [text for _, text in sorted(pages)]
        start = reference_start_unit(ordered)
        if start is not None:
            starts[source] = sorted(unit for unit, _ in pages)[start]
    return starts


def _unit_index(document: Document) -> int | None:
    unit = str(document.metadata.get("source_unit", ""))
    suffix = unit.rpartition("-")[2]
    return int(suffix) if suffix.isdigit() else None


def _is_after_references(document: Document, starts: dict[str, int]) -> bool:
    start = starts.get(str(document.metadata.get("source_path", "")))
    unit = _unit_index(document)
    return start is not None and unit is not None and unit >= start


def split_documents(
    documents: Iterable[Document], config: SplitConfig | None = None
) -> list[Document]:
    """Split documents into smaller chunks suitable for embedding."""
    cfg = config or SplitConfig()
    if cfg.keep_separator:
        raise ValueError("keep_separator=True is not supported.")
    ordered = list(documents)
    reference_starts = _reference_starts(ordered)
    chunked_docs: list[Document] = []
    for document in ordered:
        ext = str(document.metadata.get("ext", "") or "").lower()
        separators = _select_separators(ext, cfg)
        cleaned_content = _normalize_text(document.page_content)
        splits = [
            Document(page_content=text, metadata=dict(document.metadata))
            for text in _split_text(cleaned_content, separators, cfg.chunk_size, cfg.chunk_overlap)
        ]
        if not splits:
            continue
        total_chunks = len(splits)
        source_id = document.metadata.get("source_id")
        source_unit = document.metadata.get("source_unit", "unit-0")
        parent_id = document.metadata.get("doc_id") or _sha1(cleaned_content.strip())
        for index, chunk in enumerate(splits):
            chunk_id = (
                f"{source_id}:{source_unit}:{index}"
                if source_id is not None
                else f"{parent_id}:{index}"
            )
            metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_index": index,
                "num_chunks": total_chunks,
                "chunk_id": chunk_id,
                "kind": classify_chunk(
                    chunk.page_content,
                    after_references=_is_after_references(document, reference_starts),
                ),
            }
            chunk.metadata = metadata
            chunked_docs.append(chunk)
    return chunked_docs


__all__ = ["SplitConfig", "split_documents"]
