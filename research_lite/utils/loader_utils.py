#
# loader_utils.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#


from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from research_lite import Document


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a content hash without loading a potentially large file all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


IGNORE_FILENAME = ".researchliteignore"


def load_ignore_rules(library_root: Path) -> list[tuple[str, bool, bool]]:
    """Read ``.researchliteignore`` as ordered (pattern, negated, directory only) rules."""
    path = library_root / IGNORE_FILENAME
    if not path.is_file():
        return []
    rules: list[tuple[str, bool, bool]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        negated = entry.startswith("!")
        entry = entry.removeprefix("!").strip()
        directory_only = entry.endswith("/")
        entry = entry.rstrip("/")
        if entry:
            rules.append((entry, negated, directory_only))
    return rules


def is_ignored(
    relative_path: str, rules: list[tuple[str, bool, bool]], *, is_directory: bool
) -> bool:
    """Apply ignore rules in order, letting the last matching rule decide."""
    ignored = False
    segments = relative_path.split("/")
    for pattern, negated, directory_only in rules:
        if directory_only and not is_directory:
            continue
        if "/" in pattern:
            matched = fnmatch(relative_path, pattern) or fnmatch(relative_path, f"*/{pattern}")
        else:
            matched = any(fnmatch(segment, pattern) for segment in segments)
        if matched:
            ignored = not negated
    return ignored


def discover_files(path: str, extensions: Iterable[str]) -> list[Path]:
    """Discover supported files once, in a stable order, without cache directories."""
    base_path = Path(path)
    normalized_exts = {ext.lower().lstrip(".") for ext in extensions}

    if not base_path.exists():
        return []

    if base_path.is_file():
        suffix = base_path.suffix.lower().lstrip(".")
        return [base_path] if suffix in normalized_exts and not base_path.is_symlink() else []

    rules = load_ignore_rules(base_path)
    discovered: list[Path] = []
    for directory, subdirectories, filenames in os.walk(base_path, followlinks=False):
        current_dir = Path(directory)
        prefix = current_dir.relative_to(base_path).as_posix()
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if not name.startswith(".")
            and not is_ignored(f"{prefix}/{name}".lstrip("/"), rules, is_directory=True)
        )
        for filename in sorted(filenames):
            file_path = current_dir / filename
            suffix = file_path.suffix.lower().lstrip(".")
            if suffix not in normalized_exts or file_path.is_symlink():
                continue
            if is_ignored(f"{prefix}/{filename}".lstrip("/"), rules, is_directory=False):
                continue
            discovered.append(file_path)

    return sorted(discovered, key=lambda item: item.relative_to(base_path).as_posix())


def iter_files(path: str, extensions: Iterable[str]) -> list[Path]:
    """Return matching files for the provided path and extensions."""
    return discover_files(path, extensions)


def build_source_metadata(source_file: Path) -> dict[str, Any]:
    """Calculate file-level metadata once for all documents parsed from a source."""
    resolved_file = source_file.resolve()
    source_stat = resolved_file.stat()
    return {
        "source_id": _sha256_file(resolved_file),
        "source_path": str(resolved_file),
        "title": source_file.stem,
        "ext": source_file.suffix.lstrip(".").lower(),
        "mtime": source_stat.st_mtime,
    }


_PLACEHOLDER_TITLES = {
    "document",
    "no title",
    "powerpoint presentation",
    "presentation",
    "slide 1",
    "title",
    "untitled",
}
_FILE_SUFFIXES = (".doc", ".docx", ".dvi", ".pdf", ".ppt", ".pptx", ".tex", ".txt")


def usable_title(title: object, fallback: str) -> str:
    """Return a cleaned metadata title, or ``fallback`` when it looks like a placeholder."""
    text = " ".join(str(title or "").split())
    lowered = text.lower()
    if (
        len(text) < 4
        or lowered in _PLACEHOLDER_TITLES
        or lowered.startswith("microsoft word - ")
        or lowered.endswith(_FILE_SUFFIXES)
    ):
        return fallback
    return text


def populate_document_metadata(
    document: Document, source_metadata: dict[str, Any], *, source_unit_index: int
) -> Document:
    """Populate standard metadata fields for a document."""
    page = document.metadata.get("page")
    source_unit = f"page-{page}" if page is not None else f"unit-{source_unit_index}"
    document.metadata.update(
        {
            "doc_id": _sha1(document.page_content.strip()),
            "source_unit": source_unit,
            **source_metadata,
        }
    )
    return document


def load_text_by_ext(path: str, ext: str) -> list[Document]:
    text_files = iter_files(path=path, extensions=[ext])
    documents: list[Document] = []
    for text_file in text_files:
        document = Document(page_content=text_file.read_text(encoding="utf-8"))
        documents.append(
            populate_document_metadata(
                document, build_source_metadata(text_file), source_unit_index=0
            )
        )

    return documents
