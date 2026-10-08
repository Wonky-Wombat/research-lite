#
# csv_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import csv

from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_csv(path: str) -> list[Document]:
    """Load CSV documents from a file or directory, one document per row."""
    documents: list[Document] = []
    for csv_file in iter_files(path, extensions=["csv"]):
        source_metadata = build_source_metadata(csv_file)
        with csv_file.open(newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                content = "\n".join(
                    f"{_csv_text(key)}: {_csv_text(value)}" for key, value in row.items()
                )
                documents.append(
                    populate_document_metadata(
                        Document(page_content=content), source_metadata, source_unit_index=row_index
                    )
                )
    return documents


def _csv_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ",".join(item.strip() for item in value)
    return str(value)
