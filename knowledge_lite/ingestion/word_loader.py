from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from ..utils.loader_utils import _sha1


def load_word(path: str) -> list[Document]:
    """Load Word (docx) documents from a file or directory."""
    base_path = Path(path)
    if base_path.is_file():
        word_files = [base_path] if base_path.suffix.lower() == ".docx" else []
    elif base_path.is_dir():
        word_files = [
            word_file
            for word_file in base_path.rglob("*")
            if word_file.is_file() and word_file.suffix.lower() == ".docx"
        ]
    else:
        return []

    documents: list[Document] = []
    for word_file in word_files:
        loader = Docx2txtLoader(str(word_file))
        for document in loader.load():
            document.metadata.update(
                {
                    "source_path": str(word_file.resolve()),
                    "title": word_file.stem,
                    "ext": word_file.suffix.lstrip(".").lower(),
                    "mtime": word_file.stat().st_mtime,
                    "doc_id": _sha1(document.page_content.strip()),
                }
            )
            documents.append(document)

    return documents
