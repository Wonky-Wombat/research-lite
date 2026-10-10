#
# pdf_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from research_lite import Document

from ..utils.loader_utils import (
    build_source_metadata,
    iter_files,
    populate_document_metadata,
    usable_title,
)

PDF_READER = "pdfium"


def load_pdf(path: str) -> list[Document]:
    """Load PDF documents from a single file or recursively from a directory."""
    documents: list[Document] = []
    for pdf_file in iter_files(path, extensions=["pdf"]):
        pdf = pdfium.PdfDocument(pdf_file)
        try:
            source_metadata = build_source_metadata(pdf_file)
            source_metadata["title"] = _title(pdf, pdf_file.stem)
            for page_number in range(len(pdf)):
                document = Document(
                    page_content=_page_text(pdf, page_number), metadata={"page": page_number}
                )
                documents.append(
                    populate_document_metadata(
                        document, source_metadata, source_unit_index=page_number
                    )
                )
        finally:
            pdf.close()
    return documents


def pdf_title(path: Path) -> str:
    """Return the PDF's own title, or its file name when it has no usable title."""
    pdf = pdfium.PdfDocument(path)
    try:
        return _title(pdf, path.stem)
    finally:
        pdf.close()


def _title(pdf: pdfium.PdfDocument, fallback: str) -> str:
    try:
        title = pdf.get_metadata_dict().get("Title")
    except Exception:
        title = None
    return usable_title(title, fallback)


def _page_text(pdf: pdfium.PdfDocument, index: int) -> str:
    page = pdf[index]
    try:
        textpage = page.get_textpage()
        try:
            text: str = textpage.get_text_range()
        finally:
            textpage.close()
    finally:
        page.close()
    return text.replace("\ufffe", "").replace("\r\n", "\n").replace("\r", "\n").strip()
