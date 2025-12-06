import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.documents import Document

import knowledge_lite.ingestion.pdf_loader as pdf_loader_module
from knowledge_lite.ingestion.markdown_loader import load_markdown
from knowledge_lite.ingestion.pdf_loader import load_pdf
from knowledge_lite.ingestion.text_loader import load_text


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


LOADERS: dict[str, Callable[[str], list[Document]]] = {
    "md": load_markdown,
    "txt": load_text,
}


@pytest.mark.parametrize("ext", ["md", "txt"])
def test_load_single_md_file(tmp_path: Path, ext: str) -> None:
    file = tmp_path / f"test.{ext}"
    content = "# Title\nHello World" if ext == "md" else "Hello World"
    file.write_text(content, encoding="utf-8")

    docs: list[Document] = LOADERS[ext](str(file))
    assert len(docs) == 1

    doc = docs[0]
    assert doc.page_content.strip() == content
    meta = doc.metadata
    assert meta["ext"] == ext
    assert meta["title"] == "test"
    assert Path(meta["source_path"]).resolve() == file.resolve()
    assert isinstance(meta["mtime"], int | float)
    assert isinstance(meta["doc_id"], str) and len(meta["doc_id"]) == 40
    assert meta["doc_id"] == _sha1(content.strip())


def test_load_pdf_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    nested = pdf_root / "nested"
    nested.mkdir()

    pdf_files = [pdf_root / "first.pdf", nested / "second.PDF"]
    for pdf in pdf_files:
        pdf.write_bytes(b"%PDF-1.0\n% dummy content\n")

    class DummyLoader:
        def __init__(self, path: str):
            self.path = Path(path)

        def load(self) -> list[Document]:
            return [
                Document(
                    page_content=f"content-{self.path.stem}",
                    metadata={"from_loader": True},
                )
            ]

    monkeypatch.setattr(pdf_loader_module, "PyPDFLoader", DummyLoader)

    docs = load_pdf(str(pdf_root))
    assert len(docs) == len(pdf_files)

    docs_by_title = {doc.metadata["title"]: doc for doc in docs}
    for pdf in pdf_files:
        doc = docs_by_title[pdf.stem]
        assert doc.page_content == f"content-{pdf.stem}"

        meta = doc.metadata
        assert meta["ext"] == "pdf"
        assert meta["title"] == pdf.stem
        assert Path(meta["source_path"]).resolve() == pdf.resolve()
        assert isinstance(meta["mtime"], float | int)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True
