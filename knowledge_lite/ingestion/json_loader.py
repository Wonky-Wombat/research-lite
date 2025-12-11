from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from ..utils.loader_utils import iter_files, populate_document_metadata


def load_json(path: str) -> list[Document]:
    """Load JSON documents from a file or directory."""
    json_files = iter_files(path, extensions=["json"])

    documents: list[Document] = []
    for json_file in json_files:
        loader = JSONLoader(file_path=str(json_file), jq_schema=".", text_content=True)
        for document in loader.load():
            documents.append(populate_document_metadata(document, json_file))

    return documents
