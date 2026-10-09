#
# epub_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-09.
#

from __future__ import annotations

import posixpath
import zipfile
from urllib.parse import unquote
from xml.etree import ElementTree

from research_lite import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata
from .html_loader import html_to_text

_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_OPF = "{http://www.idpf.org/2007/opf}"
_HTML_TYPES = {"application/xhtml+xml", "text/html"}


def load_epub(path: str) -> list[Document]:
    """Load EPUB books in reading order, one document per spine item."""
    documents: list[Document] = []
    for epub_file in iter_files(path, extensions=["epub"]):
        source_metadata = build_source_metadata(epub_file)
        with zipfile.ZipFile(epub_file) as archive:
            for index, part in enumerate(_spine_parts(archive)):
                text = html_to_text(archive.read(part))
                if text:
                    documents.append(
                        populate_document_metadata(
                            Document(page_content=text), source_metadata, source_unit_index=index
                        )
                    )
    return documents


def _spine_parts(archive: zipfile.ZipFile) -> list[str]:
    container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
    rootfile = container.find(f".//{_CONTAINER}rootfile")
    if rootfile is None:
        raise ValueError("EPUB container lists no package document.")
    package_path = rootfile.attrib["full-path"]
    package = ElementTree.fromstring(archive.read(package_path))
    folder = posixpath.dirname(package_path)
    manifest = {
        item.attrib["id"]: item.attrib
        for item in package.iter(f"{_OPF}item")
        if "id" in item.attrib and "href" in item.attrib
    }
    names = set(archive.namelist())
    parts = []
    for itemref in package.iter(f"{_OPF}itemref"):
        item = manifest.get(itemref.attrib.get("idref", ""))
        if item is None or item.get("media-type") not in _HTML_TYPES:
            continue
        part = posixpath.normpath(posixpath.join(folder, unquote(item["href"])))
        if part in names:
            parts.append(part)
    return parts
