from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_html(path: str) -> list[Document]:
    """Load HTML documents from a file or directory."""
    html_files = iter_files(path, extensions=["html", "htm"])

    documents: list[Document] = []
    for html_file in html_files:
        source_metadata = build_source_metadata(html_file)
        loader = UnstructuredHTMLLoader(str(html_file), mode="elements")
        for source_unit_index, document in enumerate(loader.load()):
            documents.append(
                populate_document_metadata(
                    document, source_metadata, source_unit_index=source_unit_index
                )
            )

    return documents
