from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.loader_utils import _sha1

ZERO_WIDTH = ["\ufeff", "\u200b", "\u200c", "\u200d"]


def _normalize_text(text: str) -> str:
    """Apply lightweight normalization to reduce noise before splitting."""
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


def split_documents(
    documents: Iterable[Document], config: SplitConfig | None = None
) -> list[Document]:
    """Split documents into smaller chunks suitable for embedding."""
    cfg = config or SplitConfig()
    chunked_docs: list[Document] = []
    for document in documents:
        ext = str(document.metadata.get("ext", "") or "").lower()
        separators = _select_separators(ext, cfg)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=list(separators),
            keep_separator=cfg.keep_separator,
        )
        cleaned_content = _normalize_text(document.page_content)
        base_doc = Document(page_content=cleaned_content, metadata=dict(document.metadata))

        splits = splitter.split_documents([base_doc])
        if not splits:
            continue
        total_chunks = len(splits)
        parent_id = document.metadata.get("doc_id") or _sha1(cleaned_content.strip())
        for index, chunk in enumerate(splits):
            metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_index": index,
                "num_chunks": total_chunks,
                "chunk_id": f"{parent_id}:{index}",
            }
            chunk.metadata = metadata
            chunked_docs.append(chunk)
    return chunked_docs


__all__ = ["SplitConfig", "split_documents"]
