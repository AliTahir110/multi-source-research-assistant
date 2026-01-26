import os

from app.ingestion.pdf_ingest import load_and_split_pdfs
from app.storage.vector_store import build_chroma_from_chunks

DATA_DIR = "data"


def main():
    pdf_paths = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        raise RuntimeError("No PDFs found in ./data. Put at least 1 PDF there.")

    chunks = load_and_split_pdfs(pdf_paths)
    _vectordb = build_chroma_from_chunks(chunks)

    print(f"✅ Indexed PDFs: {len(pdf_paths)}")
    print(f"✅ Total chunks: {len(chunks)}")
    print("✅ Saved Chroma to ./.chroma_store")


if __name__ == "__main__":
    main()
