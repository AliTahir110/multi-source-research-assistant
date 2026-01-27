from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# IMPORTANT: importing config loads .env and validates keys
from app.core.config import OPENAI_API_KEY  


def build_chroma_from_chunks(chunks, persist_dir: str = ".chroma_store"):
    """
    Creates a Chroma vector store from document chunks and persists it locally.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,   # explicit is better than implicit
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="docs",
    )
    return vectordb


def load_chroma(persist_dir: str = ".chroma_store"):
    """
    Loads an existing Chroma vector store from disk.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="docs",
    )
    return vectordb
