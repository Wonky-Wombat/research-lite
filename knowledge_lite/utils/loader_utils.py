import hashlib
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def iter_files(path: str, extensions: Iterable[str]) -> list[Path]:
    """Return matching files for the provided path and extensions."""
    base_path = Path(path)
    normalized_exts = {ext.lower().lstrip(".") for ext in extensions}

    if not base_path.exists():
        return []

    if base_path.is_file():
        suffix = base_path.suffix.lower().lstrip(".")
        return [base_path] if suffix in normalized_exts else []

    return [
        file_path
        for file_path in base_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower().lstrip(".") in normalized_exts
    ]


def populate_document_metadata(document: Document, source_file: Path) -> Document:
    """Populate standard metadata fields for a document."""
    document.metadata.update(
        {
            "source_path": str(source_file.resolve()),
            "title": source_file.stem,
            "ext": source_file.suffix.lstrip(".").lower(),
            "mtime": source_file.stat().st_mtime,
            "doc_id": _sha1(document.page_content.strip()),
        }
    )
    return document


def load_text_by_ext(path: str, ext: str) -> list[Document]:
    text_files = iter_files(path=path, extensions=[ext])
    documents: list[Document] = []
    for text_file in text_files:
        loader = TextLoader(str(text_file), encoding="utf-8")
        for document in loader.load():
            populate_document_metadata(document, text_file)
            documents.append(document)

    return documents
