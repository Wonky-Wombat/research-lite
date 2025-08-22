import hashlib
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def load_text_by_ext(path: str, ext: str) -> List[Document]:
    base_path: Path = Path(path)

    if base_path.is_file() and base_path.suffix.lower() != f".{ext}":
        return []

    text_files: List[Path] = (
        list(base_path.rglob(f"*.{ext}")) if base_path.is_dir() else [base_path]
    )

    documents: List[Document] = []
    for text_file in text_files:
        loader = TextLoader(str(text_file), encoding="utf-8")
        for document in loader.load():
            document.metadata.update({
                "source_path": str(text_file.resolve()),
                "title": text_file.stem,
                "ext": text_file.suffix.lstrip(".").lower(),
                "mtime": text_file.stat().st_mtime,
                "doc_id": _sha1(document.page_content.strip()),
            })
            documents.append(document)

    return documents
