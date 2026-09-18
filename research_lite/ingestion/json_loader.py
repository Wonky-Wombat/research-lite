import json

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_json(path: str) -> list[Document]:
    """Load JSON documents from a file or directory."""
    json_files = iter_files(path, extensions=["json"])

    documents: list[Document] = []
    for json_file in json_files:
        source_metadata = build_source_metadata(json_file)
        loader = JSONLoader(file_path=str(json_file), jq_schema=".", text_content=False)
        for source_unit_index, document in enumerate(loader.load()):
            content = document.page_content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            normalized = Document(page_content=content, metadata=document.metadata)
            documents.append(
                populate_document_metadata(
                    normalized, source_metadata, source_unit_index=source_unit_index
                )
            )

    return documents
