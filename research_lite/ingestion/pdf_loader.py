from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_pdf(path: str) -> list[Document]:
    """Load PDF documents from a single file or recursively from a directory."""
    pdf_files = iter_files(path, extensions=["pdf"])

    documents: list[Document] = []
    for pdf_file in pdf_files:
        source_metadata = build_source_metadata(pdf_file)
        loader = PyPDFLoader(str(pdf_file))
        for source_unit_index, document in enumerate(loader.load()):
            documents.append(
                populate_document_metadata(
                    document, source_metadata, source_unit_index=source_unit_index
                )
            )

    return documents
