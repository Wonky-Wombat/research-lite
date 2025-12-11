import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.documents import Document

import knowledge_lite.ingestion as ingestion_module
import knowledge_lite.ingestion.pdf_loader as pdf_loader_module
import knowledge_lite.ingestion.word_loader as word_loader_module
from knowledge_lite.ingestion import load_documents
from knowledge_lite.ingestion.markdown_loader import load_markdown
from knowledge_lite.ingestion.pdf_loader import load_pdf
from knowledge_lite.ingestion.text_loader import load_text
from knowledge_lite.ingestion.word_loader import load_word


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


LOADERS: dict[str, Callable[[str], list[Document]]] = {
    "md": load_markdown,
    "txt": load_text,
}


@pytest.mark.parametrize("ext", ["md", "txt"])
def test_load_single_md_file(tmp_path: Path, ext: str) -> None:
    file = tmp_path / f"test.{ext}"
    content = "Hello KnowledgeLiteRAG"
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
        pdf.write_bytes(b"%PDF-1.0\n% Hello KnowledgeLiteRAG\n")

    dummy_doc_content = "Hello KnowledgeLiteRAG"

    class DummyLoader:
        def __init__(self, path: str):
            self.path = Path(path)

        def load(self) -> list[Document]:
            return [
                Document(
                    page_content=dummy_doc_content,
                    metadata={"from_loader": True},
                )
            ]

    monkeypatch.setattr(pdf_loader_module, "PyPDFLoader", DummyLoader)

    docs = load_pdf(str(pdf_root))
    assert len(docs) == len(pdf_files)

    docs_by_title = {doc.metadata["title"]: doc for doc in docs}
    for pdf in pdf_files:
        doc = docs_by_title[pdf.stem]
        assert doc.page_content == dummy_doc_content

        meta = doc.metadata
        assert meta["ext"] == "pdf"
        assert meta["title"] == pdf.stem
        assert Path(meta["source_path"]).resolve() == pdf.resolve()
        assert isinstance(meta["mtime"], float | int)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_word_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    word_root = tmp_path / "docs"
    word_root.mkdir()
    nested = word_root / "nested"
    nested.mkdir()

    word_files = [word_root / "first.docx", nested / "second.DOCX"]
    dummy_file_content = "Hello KnowledgeLiteRAG"
    for word in word_files:
        word.write_text(dummy_file_content, encoding="utf-8")

    dummy_doc_content = "Hello KnowledgeLiteRAG"

    class DummyLoader:
        def __init__(self, path: str):
            self.path = Path(path)

        def load(self) -> list[Document]:
            return [
                Document(
                    page_content=dummy_doc_content,
                    metadata={"from_loader": True},
                )
            ]

    monkeypatch.setattr(word_loader_module, "Docx2txtLoader", DummyLoader)

    docs = load_word(str(word_root))
    assert len(docs) == len(word_files)

    docs_by_title = {doc.metadata["title"]: doc for doc in docs}
    for word in word_files:
        doc = docs_by_title[word.stem]
        assert doc.page_content == dummy_doc_content

        meta = doc.metadata
        assert meta["ext"] == "docx"
        assert meta["title"] == word.stem
        assert Path(meta["source_path"]).resolve() == word.resolve()
        assert isinstance(meta["mtime"], float | int)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_documents_dispatches_selected_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[tuple[str, str]] = []

    def make_loader(ext: str) -> Callable[[str], list[Document]]:
        def _loader(path: str) -> list[Document]:
            called.append((ext, path))
            return [Document(page_content=f"{ext}-doc", metadata={"ext": ext})]

        return _loader

    fake_loaders = {ext: make_loader(ext) for ext in ("txt", "md")}
    monkeypatch.setattr(ingestion_module, "_LOADER_BY_EXT", fake_loaders)

    docs = load_documents(str(tmp_path), extensions=["txt", "md"])
    assert called == [("txt", str(tmp_path)), ("md", str(tmp_path))]
    assert [doc.metadata["ext"] for doc in docs] == ["txt", "md"]


def test_load_documents_rejects_unknown_extensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported extensions"):
        load_documents(str(tmp_path), extensions=["csv"])


def test_load_documents_rejects_empty_extension_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        load_documents(str(tmp_path), extensions=[])


def test_load_documents_returns_empty_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    docs = load_documents(str(missing))
    assert docs == []
