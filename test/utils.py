from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

DUMMY_CONTENT = "Hello KnowledgeLiteRAG"


class ConstantContentLoader:
    """Loader that returns a single Document with constant content."""

    def __init__(self, path: str | None = None, *args: Any, content: str, **kwargs: Any) -> None:
        raw_path = path or kwargs.get("file_path") or kwargs.get("path") or ""
        self.path = Path(raw_path)
        self.content = content

    def load(self) -> list[Document]:
        return [
            Document(
                page_content=self.content,
                metadata={"from_loader": True},
            )
        ]


def create_files_with_content(base_dir: Path, filenames: list[str], content: str) -> list[Path]:
    """Create files with the provided names and content, returning the file paths."""
    base_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []
    for filename in filenames:
        file_path = base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        created_files.append(file_path)
    return created_files


def make_dummy_loader(content: str) -> Callable[..., ConstantContentLoader]:
    """Return a dummy loader callable compatible with langchain loaders."""

    return partial(ConstantContentLoader, content=content)


def collect_docs_by_title(documents: list[Document]) -> dict[str, Document]:
    """Utility to index documents by their title metadata for easier assertions."""
    return {doc.metadata["title"]: doc for doc in documents}


def assert_metadata_matches_file(meta: Mapping[str, Any], source_file: Path) -> None:
    """Assert common metadata fields match the source file."""
    assert meta["title"] == source_file.stem
    assert Path(meta["source_path"]).resolve() == source_file.resolve()
    assert meta["ext"] == source_file.suffix.lstrip(".").lower()
    assert isinstance(meta["mtime"], (int | float))
    assert meta["source_id"] == hashlib.sha256(source_file.read_bytes()).hexdigest()
    assert meta["source_unit"] == "unit-0"
