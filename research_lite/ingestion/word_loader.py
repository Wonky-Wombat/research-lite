from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_word(path: str) -> list[Document]:
    """Load Word (docx) documents from a file or directory."""
    word_files = iter_files(path, extensions=["docx"])

    documents: list[Document] = []
    for word_file in word_files:
        source_metadata = build_source_metadata(word_file)
        loader = Docx2txtLoader(str(word_file))
        for source_unit_index, document in enumerate(loader.load()):
            documents.append(
                populate_document_metadata(
                    document, source_metadata, source_unit_index=source_unit_index
                )
            )

    return documents
