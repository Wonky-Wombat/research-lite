#
# pptx_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-09.
#

from __future__ import annotations

import posixpath
import zipfile
from xml.etree import ElementTree

from research_lite import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata

_DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_PRESENTATION = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_RELATIONSHIPS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_RELATIONSHIP_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def load_pptx(path: str) -> list[Document]:
    """Load PowerPoint (pptx) slides and speaker notes, one document per slide."""
    documents: list[Document] = []
    for pptx_file in iter_files(path, extensions=["pptx"]):
        source_metadata = build_source_metadata(pptx_file)
        with zipfile.ZipFile(pptx_file) as archive:
            for index, slide in enumerate(_slide_parts(archive)):
                notes = _related_part(archive, slide, "/notesSlide")
                text = "\n\n".join(
                    part for part in (_text(archive, slide), _text(archive, notes)) if part
                )
                if text:
                    documents.append(
                        populate_document_metadata(
                            Document(page_content=text, metadata={"page": index}),
                            source_metadata,
                            source_unit_index=index,
                        )
                    )
    return documents


def _slide_parts(archive: zipfile.ZipFile) -> list[str]:
    targets = {
        rel_id: target for rel_id, _, target in _relationships(archive, "ppt/presentation.xml")
    }
    presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
    return [
        targets[slide_id.attrib[_RELATIONSHIP_ID]]
        for slide_id in presentation.iter(f"{_PRESENTATION}sldId")
        if slide_id.attrib.get(_RELATIONSHIP_ID) in targets
    ]


def _related_part(archive: zipfile.ZipFile, part: str, type_suffix: str) -> str | None:
    return next(
        (target for _, kind, target in _relationships(archive, part) if kind.endswith(type_suffix)),
        None,
    )


def _relationships(archive: zipfile.ZipFile, part: str) -> list[tuple[str, str, str]]:
    folder = posixpath.dirname(part)
    rels_path = posixpath.join(folder, "_rels", posixpath.basename(part) + ".rels")
    if rels_path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(rels_path))
    return [
        (
            rel.attrib.get("Id", ""),
            rel.attrib.get("Type", ""),
            posixpath.normpath(posixpath.join(folder, rel.attrib["Target"])),
        )
        for rel in root.iter(f"{_RELATIONSHIPS}Relationship")
        if rel.attrib.get("TargetMode") != "External"
    ]


def _text(archive: zipfile.ZipFile, part: str | None) -> str:
    if part is None or part not in archive.namelist():
        return ""
    root = ElementTree.fromstring(archive.read(part))
    paragraphs = (
        "".join(
            "\n" if child.tag == f"{_DRAWING}br" else child.findtext(f"{_DRAWING}t", "")
            for child in paragraph
            if child.tag in {f"{_DRAWING}r", f"{_DRAWING}br"}
        )
        for paragraph in root.iter(f"{_DRAWING}p")
    )
    return "\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
