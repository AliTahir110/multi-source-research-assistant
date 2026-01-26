import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdfs(pdf_paths: List[str]):
    """
    Load PDFs and split into chunks while preserving metadata (source + page).
    Returns a list of chunked Documents.
    """
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        loaded = loader.load()  # one Document per page

        # Add friendly source name for citations later
        for d in loaded:
            d.metadata["source"] = os.path.basename(path)

        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks
