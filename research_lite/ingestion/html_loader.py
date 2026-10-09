#
# html_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import re

from research_lite import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_html(path: str) -> list[Document]:
    """Load HTML documents from a file or directory as Markdown text."""
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter

    converter = MarkdownConverter(heading_style="ATX", strip=["a", "img"])
    documents: list[Document] = []
    for html_file in iter_files(path, extensions=["html", "htm"]):
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for tag in soup(["head", "noscript", "template"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", converter.convert_soup(soup)).strip()
        if text:
            documents.append(
                populate_document_metadata(
                    Document(page_content=text),
                    build_source_metadata(html_file),
                    source_unit_index=0,
                )
            )
    return documents
