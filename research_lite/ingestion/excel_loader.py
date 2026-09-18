from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_excel(path: str) -> list[Document]:
    """Load Excel documents from a file or directory."""
    excel_files = iter_files(path, extensions=["xlsx", "xls"])

    documents: list[Document] = []
    for excel_file in excel_files:
        source_metadata = build_source_metadata(excel_file)
        loader = UnstructuredExcelLoader(str(excel_file), mode="elements")
        for document in loader.load():
            documents.append(populate_document_metadata(document, source_metadata))

    return documents
