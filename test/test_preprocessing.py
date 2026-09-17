from langchain_core.documents import Document

from research_lite.preprocessing import (
    SplitConfig,
    _select_separators,
    split_documents,
)


def test_split_documents_adds_chunk_metadata() -> None:
    doc = Document(
        page_content="Hello KnowledgeLiteRAG " * 40,
        metadata={"doc_id": "doc-1", "ext": "txt", "title": "Sample"},
    )
    config = SplitConfig(chunk_size=50, chunk_overlap=0, separators=(" ",))

    chunks = split_documents([doc], config=config)
    assert len(chunks) > 1

    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert meta["chunk_index"] == index
        assert meta["num_chunks"] == total_chunks
        assert meta["chunk_id"] == f"doc-1:{index}"
        assert meta["title"] == "Sample"


def test_split_documents_generates_chunk_id_when_missing_doc_id() -> None:
    doc = Document(
        page_content="Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
        metadata={"ext": "md"},
    )
    config = SplitConfig(chunk_size=15, chunk_overlap=0, separators=("\n",))

    chunks = split_documents([doc], config=config)
    assert len(chunks) >= 2

    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        assert ":" in chunk_id
        assert chunk.metadata["ext"] == "md"


def test_select_separators_prefers_ext_specific_config() -> None:
    config = SplitConfig(
        separators=("default",),
        separators_by_ext={"md": ("\n\n",)},
    )

    assert _select_separators("md", config) == ("\n\n",)
    assert _select_separators("txt", config) == ("default",)


def test_split_documents_normalizes_text_before_splitting() -> None:
    doc = Document(
        page_content="\ufeffHello\n\n\nKnowledgeLiteRAG\u200b",
        metadata={"ext": "txt"},
    )
    config = SplitConfig(chunk_size=100, chunk_overlap=0, separators=("\n\n",))

    chunks = split_documents([doc], config=config)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Hello\n\nKnowledgeLiteRAG"
