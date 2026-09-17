import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.documents import Document

import research_lite.ingestion as ingestion_module
import research_lite.ingestion.csv_loader as csv_loader_module
import research_lite.ingestion.excel_loader as excel_loader_module
import research_lite.ingestion.html_loader as html_loader_module
import research_lite.ingestion.json_loader as json_loader_module
import research_lite.ingestion.pdf_loader as pdf_loader_module
import research_lite.ingestion.word_loader as word_loader_module
from research_lite.ingestion import load_documents
from research_lite.ingestion.csv_loader import load_csv
from research_lite.ingestion.excel_loader import load_excel
from research_lite.ingestion.html_loader import load_html
from research_lite.ingestion.json_loader import load_json
from research_lite.ingestion.markdown_loader import load_markdown
from research_lite.ingestion.pdf_loader import load_pdf
from research_lite.ingestion.text_loader import load_text
from research_lite.ingestion.word_loader import load_word
from test.utils import (
    DUMMY_CONTENT,
    assert_metadata_matches_file,
    collect_docs_by_title,
    create_files_with_content,
    make_dummy_loader,
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


LOADERS: dict[str, Callable[[str], list[Document]]] = {
    "md": load_markdown,
    "txt": load_text,
}


@pytest.mark.parametrize("ext", ["md", "txt"])
def test_load_single_md_file(tmp_path: Path, ext: str) -> None:
    file = tmp_path / f"test.{ext}"
    content = DUMMY_CONTENT
    file.write_text(content, encoding="utf-8")

    docs: list[Document] = LOADERS[ext](str(file))
    assert len(docs) == 1

    doc = docs[0]
    assert doc.page_content.strip() == content
    meta = doc.metadata
    assert_metadata_matches_file(meta, file)
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

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(pdf_loader_module, "PyPDFLoader", dummy_loader)

    docs = load_pdf(str(pdf_root))
    assert len(docs) == len(pdf_files)

    docs_by_title = collect_docs_by_title(docs)
    for pdf in pdf_files:
        doc = docs_by_title[pdf.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, pdf)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_html_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    html_root = tmp_path / "html"
    html_files = create_files_with_content(
        html_root,
        ["first.html", "nested/second.HTM"],
        "<html><body>Hello KnowledgeLiteRAG</body></html>",
    )

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(html_loader_module, "UnstructuredHTMLLoader", dummy_loader)

    docs = load_html(str(html_root))
    assert len(docs) == len(html_files)

    docs_by_title = collect_docs_by_title(docs)
    for html_file in html_files:
        doc = docs_by_title[html_file.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, html_file)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_csv_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_root = tmp_path / "csvs"
    csv_files = create_files_with_content(
        csv_root,
        ["first.csv", "nested/second.CSV"],
        "title,body\nhello,world",
    )

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(csv_loader_module, "CSVLoader", dummy_loader)

    docs = load_csv(str(csv_root))
    assert len(docs) == len(csv_files)

    docs_by_title = collect_docs_by_title(docs)
    for csv_file in csv_files:
        doc = docs_by_title[csv_file.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, csv_file)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_json_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_root = tmp_path / "json_data"
    json_files = create_files_with_content(
        json_root,
        ["first.json", "nested/second.JSON"],
        '[{"content": "Hello KnowledgeLiteRAG"}]',
    )

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(json_loader_module, "JSONLoader", dummy_loader)

    docs = load_json(str(json_root))
    assert len(docs) == len(json_files)

    docs_by_title = collect_docs_by_title(docs)
    for json_file in json_files:
        doc = docs_by_title[json_file.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, json_file)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_word_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    word_root = tmp_path / "docs"
    word_files = create_files_with_content(
        word_root,
        ["first.docx", "nested/second.DOCX"],
        DUMMY_CONTENT,
    )

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(word_loader_module, "Docx2txtLoader", dummy_loader)

    docs = load_word(str(word_root))
    assert len(docs) == len(word_files)

    docs_by_title = collect_docs_by_title(docs)
    for word in word_files:
        doc = docs_by_title[word.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, word)
        assert meta["doc_id"] == _sha1(doc.page_content.strip())
        assert meta["from_loader"] is True


def test_load_excel_enriches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    excel_root = tmp_path / "excel"
    excel_files = create_files_with_content(
        excel_root,
        ["first.xlsx", "nested/second.XLS"],
        "dummy",
    )

    dummy_loader = make_dummy_loader(DUMMY_CONTENT)
    monkeypatch.setattr(excel_loader_module, "UnstructuredExcelLoader", dummy_loader)

    docs = load_excel(str(excel_root))
    assert len(docs) == len(excel_files)

    docs_by_title = collect_docs_by_title(docs)
    for excel_file in excel_files:
        doc = docs_by_title[excel_file.stem]
        assert doc.page_content == DUMMY_CONTENT

        meta = doc.metadata
        assert_metadata_matches_file(meta, excel_file)
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
        load_documents(str(tmp_path), extensions=["foo"])


def test_load_documents_rejects_empty_extension_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        load_documents(str(tmp_path), extensions=[])


def test_load_documents_returns_empty_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    docs = load_documents(str(missing))
    assert docs == []
