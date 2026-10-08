#
# json_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import json

from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_json(path: str) -> list[Document]:
    """Load JSON documents from a file or directory."""
    documents: list[Document] = []
    for json_file in iter_files(path, extensions=["json"]):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict | list):
            content = json.dumps(data, ensure_ascii=False) if data else ""
        else:
            content = "" if data is None else str(data)
        documents.append(
            populate_document_metadata(
                Document(page_content=content),
                build_source_metadata(json_file),
                source_unit_index=0,
            )
        )
    return documents
