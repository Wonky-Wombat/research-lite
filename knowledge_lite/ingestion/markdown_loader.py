from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader
import hashlib
from ..utils.loader_utils import load_text_by_ext


def load_markdown(path: str) -> List[Document]:
    documents = load_text_by_ext(path=path, ext="md")
    return documents
