import hashlib
from pathlib import Path
from typing import List, Dict

import pytest
from langchain.schema import Document
from knowledge_lite.ingestion.markdown_loader import load_markdown
from knowledge_lite.ingestion.text_loader import load_text
from typing import List, Callable

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

LOADERS: Dict[str, Callable[[str], List[Document]]] = {
    "md": load_markdown,
    "txt": load_text,
}

@pytest.mark.parametrize("ext", ["md", "txt"])
def test_load_single_md_file(tmp_path: Path, ext: str):
    file  = tmp_path / f"test.{ext}"
    content = "# Title\nHello World" if ext == "md" else "Hello World"
    file .write_text(content, encoding="utf-8")

    docs: List[Document] = LOADERS[ext](str(file))
    assert len(docs) == 1

    doc = docs[0]
    assert doc.page_content.strip() == content
    meta = doc.metadata
    assert meta["ext"] == ext
    assert meta["title"] == "test"
    assert Path(meta["source_path"]).resolve() == file.resolve()
    assert isinstance(meta["mtime"], (int, float))
    assert isinstance(meta["doc_id"], str) and len(meta["doc_id"]) == 40
    assert meta["doc_id"] == _sha1(content.strip())
