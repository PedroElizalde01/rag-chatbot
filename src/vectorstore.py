import chromadb
import logging
from pathlib import Path

from embeddings import embed_text

VECTORDB_PATH = Path(__file__).parent.parent / "data" / "vectordb"

logging.getLogger("chromadb").setLevel(logging.ERROR)

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTORDB_PATH))

def get_or_create_collection(name: str = "documents") -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)

def search_vectorstore(query: str, k: int = 3) -> list[dict]:
    collection = get_or_create_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    combined = []
    for rank, (doc_id, doc, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        combined.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": metadata,
                "distance": distance,
                "rank": rank,
            }
        )
    return combined
