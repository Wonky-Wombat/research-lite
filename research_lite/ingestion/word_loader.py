#
# word_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import docx2txt

from research_lite import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_word(path: str) -> list[Document]:
    """Load Word (docx) documents from a file or directory."""
    documents: list[Document] = []
    for word_file in iter_files(path, extensions=["docx"]):
        document = Document(page_content=docx2txt.process(str(word_file)))
        documents.append(
            populate_document_metadata(
                document, build_source_metadata(word_file), source_unit_index=0
            )
        )
    return documents
