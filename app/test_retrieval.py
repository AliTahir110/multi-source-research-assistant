from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import OPENAI_API_KEY

PERSIST_DIR = ".chroma_store"


def main():
    # Same embedding model used during indexing
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # Load existing Chroma DB
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="docs",
    )

    # Example query
    query = "What was the objective of this project?"

    # Retrieve top 3 most relevant chunks
    docs = db.max_marginal_relevance_search(query, k=3, fetch_k=15)


    print("\n=== TOP MATCHES ===")
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source")
        page = d.metadata.get("page")

        print(f"\n[{i}] source={source}, page={page}")
        print(d.page_content[:500])  # show first 500 chars


if __name__ == "__main__":
    main()
