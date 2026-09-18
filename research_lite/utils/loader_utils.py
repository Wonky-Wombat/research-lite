import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def discover_files(path: str, extensions: Iterable[str]) -> list[Path]:
    """Discover supported files once, in a stable order, without cache directories."""
    base_path = Path(path)
    normalized_exts = {ext.lower().lstrip(".") for ext in extensions}

    if not base_path.exists():
        return []

    if base_path.is_file():
        suffix = base_path.suffix.lower().lstrip(".")
        return [base_path] if suffix in normalized_exts and not base_path.is_symlink() else []

    discovered: list[Path] = []
    for directory, subdirectories, filenames in os.walk(base_path, followlinks=False):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith("."))
        current_dir = Path(directory)
        for filename in sorted(filenames):
            file_path = current_dir / filename
            suffix = file_path.suffix.lower().lstrip(".")
            if suffix in normalized_exts and not file_path.is_symlink():
                discovered.append(file_path)

    return sorted(discovered, key=lambda item: item.relative_to(base_path).as_posix())


def iter_files(path: str, extensions: Iterable[str]) -> list[Path]:
    """Return matching files for the provided path and extensions."""
    return discover_files(path, extensions)


def build_source_metadata(source_file: Path) -> dict[str, Any]:
    """Calculate file-level metadata once for all documents parsed from a source."""
    resolved_file = source_file.resolve()
    source_stat = resolved_file.stat()
    return {
        "source_path": str(resolved_file),
        "title": source_file.stem,
        "ext": source_file.suffix.lstrip(".").lower(),
        "mtime": source_stat.st_mtime,
    }


def populate_document_metadata(document: Document, source_metadata: dict[str, Any]) -> Document:
    """Populate standard metadata fields for a document."""
    document.metadata.update(
        {
            "doc_id": _sha1(document.page_content.strip()),
            **source_metadata,
        }
    )
    return document


def load_text_by_ext(path: str, ext: str) -> list[Document]:
    text_files = iter_files(path=path, extensions=[ext])
    documents: list[Document] = []
    for text_file in text_files:
        source_metadata = build_source_metadata(text_file)
        loader = TextLoader(str(text_file), encoding="utf-8")
        for document in loader.load():
            populate_document_metadata(document, source_metadata)
            documents.append(document)

    return documents
