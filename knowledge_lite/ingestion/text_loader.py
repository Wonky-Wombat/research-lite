from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from ..utils.loader_utils import load_text_by_ext


def load_text(path: str) -> List[Document]:
    documents = load_text_by_ext(path=path, ext="txt")
    return documents
