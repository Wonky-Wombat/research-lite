#
# epub_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-09.
#

from __future__ import annotations

import posixpath
import zipfile
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from research_lite import Document

from ..utils.loader_utils import (
    build_source_metadata,
    iter_files,
    populate_document_metadata,
    usable_title,
)
from .html_loader import html_to_text

_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_OPF = "{http://www.idpf.org/2007/opf}"
_DUBLIN_CORE = "{http://purl.org/dc/elements/1.1/}"
_HTML_TYPES = {"application/xhtml+xml", "text/html"}


def load_epub(path: str) -> list[Document]:
    """Load EPUB books in reading order, one document per spine item."""
    documents: list[Document] = []
    for epub_file in iter_files(path, extensions=["epub"]):
        source_metadata = build_source_metadata(epub_file)
        with zipfile.ZipFile(epub_file) as archive:
            package_path, package = _package(archive)
            source_metadata["title"] = _title(package, epub_file.stem)
            for index, part in enumerate(_spine_parts(archive, package_path, package)):
                text = html_to_text(archive.read(part))
                if text:
                    documents.append(
                        populate_document_metadata(
                            Document(page_content=text), source_metadata, source_unit_index=index
                        )
                    )
    return documents


def epub_title(path: Path) -> str:
    """Return the book title declared by an EPUB, or its file name."""
    with zipfile.ZipFile(path) as archive:
        return _title(_package(archive)[1], path.stem)


def _package(archive: zipfile.ZipFile) -> tuple[str, ElementTree.Element]:
    container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
    rootfile = container.find(f".//{_CONTAINER}rootfile")
    if rootfile is None:
        raise ValueError("EPUB container lists no package document.")
    package_path = rootfile.attrib["full-path"]
    return package_path, ElementTree.fromstring(archive.read(package_path))


def _title(package: ElementTree.Element, fallback: str) -> str:
    return usable_title(package.findtext(f".//{_DUBLIN_CORE}title"), fallback)


def _spine_parts(
    archive: zipfile.ZipFile, package_path: str, package: ElementTree.Element
) -> list[str]:
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
