from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.documents import Document

from .csv_loader import load_csv
from .excel_loader import load_excel
from .html_loader import load_html
from .json_loader import load_json
from .markdown_loader import load_markdown
from .pdf_loader import load_pdf
from .text_loader import load_text
from .word_loader import load_word

_LOADER_BY_EXT: dict[str, Callable[[str], list[Document]]] = {
    "txt": load_text,
    "md": load_markdown,
    "pdf": load_pdf,
    "html": load_html,
    "htm": load_html,
    "csv": load_csv,
    "xlsx": load_excel,
    "xls": load_excel,
    "docx": load_word,
    "json": load_json,
}


def load_documents(path: str, extensions: Iterable[str] | None = None) -> list[Document]:
    """Load documents from path using the specified extension."""
    normalized_exts = (
        [ext.lower() for ext in extensions] if extensions is not None else list(_LOADER_BY_EXT)
    )
    if not normalized_exts:
        raise ValueError("Argument 'extensions' cannot be empty.")

    invalid_exts = [ext for ext in normalized_exts if ext not in _LOADER_BY_EXT]
    if invalid_exts:
        supported = ", ".join(sorted(_LOADER_BY_EXT))
        raise ValueError(f"unsupported extensions: {invalid_exts}; supported: {supported}")

    base_path = Path(path)
    if not base_path.exists():
        return []

    resolved_path = str(base_path)
    documents: list[Document] = []
    for ext in normalized_exts:
        loader = _LOADER_BY_EXT[ext]
        documents.extend(loader(resolved_path))

    return documents
