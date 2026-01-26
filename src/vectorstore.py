import chromadb
from pathlib import Path

from embeddings import embed_text

VECTORDB_PATH = Path(__file__).parent.parent / "data" / "vectordb"

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTORDB_PATH))

def get_or_create_collection(name: str = "documents") -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)

def search_vectorstore(query: str, k: int = 3) -> list[str]:
    collection = get_or_create_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return results["documents"][0]
