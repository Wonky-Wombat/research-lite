#
# html_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import re

from research_lite import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def html_to_text(markup: str | bytes) -> str:
    """Convert HTML markup to Markdown text."""
    from bs4 import BeautifulSoup, CData, Declaration, ProcessingInstruction
    from markdownify import MarkdownConverter

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["head", "noscript", "template"]):
        tag.decompose()
    for node in soup.find_all(
        string=lambda text: isinstance(text, CData | Declaration | ProcessingInstruction)
    ):
        node.extract()
    converter = MarkdownConverter(heading_style="ATX", strip=["a", "img"])
    return re.sub(r"\n{3,}", "\n\n", converter.convert_soup(soup)).strip()


def load_html(path: str) -> list[Document]:
    """Load HTML documents from a file or directory as Markdown text."""
    documents: list[Document] = []
    for html_file in iter_files(path, extensions=["html", "htm"]):
        text = html_to_text(html_file.read_text(encoding="utf-8", errors="replace"))
        if text:
            documents.append(
                populate_document_metadata(
                    Document(page_content=text),
                    build_source_metadata(html_file),
                    source_unit_index=0,
                )
            )
    return documents
