from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from ..utils.loader_utils import iter_files, populate_document_metadata


def load_word(path: str) -> list[Document]:
    """Load Word (docx) documents from a file or directory."""
    word_files = iter_files(path, extensions=["docx"])

    documents: list[Document] = []
    for word_file in word_files:
        loader = Docx2txtLoader(str(word_file))
        for document in loader.load():
            documents.append(populate_document_metadata(document, word_file))

    return documents
