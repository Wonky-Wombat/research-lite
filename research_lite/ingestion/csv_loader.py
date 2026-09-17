from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from ..utils.loader_utils import iter_files, populate_document_metadata


def load_csv(path: str) -> list[Document]:
    """Load CSV documents from a file or directory."""
    csv_files = iter_files(path, extensions=["csv"])

    documents: list[Document] = []
    for csv_file in csv_files:
        loader = CSVLoader(file_path=str(csv_file), csv_args={"delimiter": ",", "quotechar": '"'})
        for document in loader.load():
            documents.append(populate_document_metadata(document, csv_file))

    return documents
