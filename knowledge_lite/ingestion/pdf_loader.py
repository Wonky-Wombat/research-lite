from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..utils.loader_utils import _sha1


def load_pdf(path: str) -> list[Document]:
    """Load PDF documents from a single file or recursively from a directory."""
    base_path = Path(path)
    if base_path.is_file():
        pdf_files = [base_path] if base_path.suffix.lower() == ".pdf" else []
    elif base_path.is_dir():
        pdf_files = [
            pdf_file
            for pdf_file in base_path.rglob("*")
            if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf"
        ]
    else:
        return []

    documents: list[Document] = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        for document in loader.load():
            document.metadata.update(
                {
                    "source_path": str(pdf_file.resolve()),
                    "title": pdf_file.stem,
                    "ext": pdf_file.suffix.lstrip(".").lower(),
                    "mtime": pdf_file.stat().st_mtime,
                    "doc_id": _sha1(document.page_content.strip()),
                }
            )
            documents.append(document)

    return documents
