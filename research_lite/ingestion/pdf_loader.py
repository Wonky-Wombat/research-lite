#
# pdf_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from pypdf import PdfReader

from research_lite import Document

from ..utils.loader_utils import (
    build_source_metadata,
    iter_files,
    populate_document_metadata,
    usable_title,
)


def load_pdf(path: str) -> list[Document]:
    """Load PDF documents from a single file or recursively from a directory."""
    documents: list[Document] = []
    for pdf_file in iter_files(path, extensions=["pdf"]):
        reader = PdfReader(pdf_file)
        source_metadata = build_source_metadata(pdf_file)
        source_metadata["title"] = pdf_title(reader, pdf_file.stem)
        for page_number, page in enumerate(reader.pages):
            document = Document(
                page_content=page.extract_text(extraction_mode="plain").strip(),
                metadata={"page": page_number},
            )
            documents.append(
                populate_document_metadata(document, source_metadata, source_unit_index=page_number)
            )
    return documents


def pdf_title(reader: PdfReader, fallback: str) -> str:
    """Return the PDF's own title, or ``fallback`` when it has no usable title."""
    try:
        metadata = reader.metadata
        title = metadata.title if metadata is not None else None
    except Exception:
        title = None
    return usable_title(title, fallback)
