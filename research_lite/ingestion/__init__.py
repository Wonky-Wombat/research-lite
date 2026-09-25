#
# __init__.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from ..utils.loader_utils import discover_files
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

SUPPORTED_EXTENSIONS = tuple(_LOADER_BY_EXT)


@dataclass(frozen=True)
class IngestFailure:
    """A source that could not be loaded without stopping the full ingest."""

    source_path: str
    reason: str


@dataclass
class IngestReport:
    """Summary of a document-loading operation."""

    discovered_files: int = 0
    processed_files: int = 0
    loaded_documents: int = 0
    failures: list[IngestFailure] = field(default_factory=list)

    @property
    def failed_files(self) -> int:
        return len(self.failures)


def _normalize_extensions(extensions: Iterable[str] | None) -> list[str]:
    normalized_exts = (
        [ext.lower().lstrip(".") for ext in extensions]
        if extensions is not None
        else list(SUPPORTED_EXTENSIONS)
    )
    if not normalized_exts:
        raise ValueError("Argument 'extensions' cannot be empty.")

    invalid_exts = [ext for ext in normalized_exts if ext not in _LOADER_BY_EXT]
    if invalid_exts:
        supported = ", ".join(sorted(_LOADER_BY_EXT))
        raise ValueError(f"unsupported extensions: {invalid_exts}; supported: {supported}")

    return normalized_exts


def load_documents_with_report(
    path: str, extensions: Iterable[str] | None = None
) -> tuple[list[Document], IngestReport]:
    """Load supported files independently and return documents with an ingest report."""
    normalized_exts = _normalize_extensions(extensions)

    base_path = Path(path)
    if not base_path.exists():
        return [], IngestReport()

    documents: list[Document] = []
    report = IngestReport()
    source_files = discover_files(str(base_path), normalized_exts)
    report.discovered_files = len(source_files)

    for source_file in source_files:
        ext = source_file.suffix.lower().lstrip(".")
        loader = _LOADER_BY_EXT[ext]
        try:
            loaded_documents = loader(str(source_file))
        except Exception as exc:
            report.failures.append(IngestFailure(str(source_file), str(exc)))
            continue
        documents.extend(loaded_documents)
        report.processed_files += 1
        report.loaded_documents += len(loaded_documents)

    return documents, report


def load_documents(
    path: str, extensions: Iterable[str] | None = None, *, report: IngestReport | None = None
) -> list[Document]:
    """Load supported documents while preserving the original list-only API."""
    documents, generated_report = load_documents_with_report(path, extensions)
    if report is not None:
        report.discovered_files = generated_report.discovered_files
        report.processed_files = generated_report.processed_files
        report.loaded_documents = generated_report.loaded_documents
        report.failures = generated_report.failures
    return documents


__all__ = [
    "IngestFailure",
    "IngestReport",
    "SUPPORTED_EXTENSIONS",
    "load_documents",
    "load_documents_with_report",
]
