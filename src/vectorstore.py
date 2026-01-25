import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

from loader import read_pdf, chunk_text
from embeddings import embed_text

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

VECTORDB_PATH = Path(__file__).parent.parent / "data" / "vectordb"


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTORDB_PATH))


def get_or_create_collection(name: str = "documents") -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)


def init_vector_store() -> chromadb.Collection:
    VECTORDB_PATH.mkdir(parents=True, exist_ok=True)
    collection = get_or_create_collection()
    print(f"Vector store initialized at: {VECTORDB_PATH}")
    return collection


def ingest_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    text = read_pdf(pdf_path)
    
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    collection = get_or_create_collection()
    
    pdf_name = Path(pdf_path).stem
    
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        
        collection.add(
            ids=[f"{pdf_name}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": pdf_path, "chunk_index": i}]
        )
        
    return len(chunks)


def verify_ingestion():
    collection = get_or_create_collection()
    count = collection.count()
    return count > 0


if __name__ == "__main__":
    ingest_pdf("data/documents/fastapi.pdf")
    verify_ingestion()
