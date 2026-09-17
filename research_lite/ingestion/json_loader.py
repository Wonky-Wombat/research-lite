import json

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from ..utils.loader_utils import iter_files, populate_document_metadata


def load_json(path: str) -> list[Document]:
    """Load JSON documents from a file or directory."""
    json_files = iter_files(path, extensions=["json"])

    documents: list[Document] = []
    for json_file in json_files:
        loader = JSONLoader(file_path=str(json_file), jq_schema=".", text_content=False)
        for document in loader.load():
            content = document.page_content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            normalized = Document(page_content=content, metadata=document.metadata)
            documents.append(populate_document_metadata(normalized, json_file))

    return documents
