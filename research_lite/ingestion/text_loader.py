from langchain_core.documents import Document

from ..utils.loader_utils import load_text_by_ext


def load_text(path: str) -> list[Document]:
    documents = load_text_by_ext(path=path, ext="txt")
    return documents
