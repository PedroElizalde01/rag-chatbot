from pathlib import Path

from embeddings import embed_text
from loader import read_pdf, chunk_text
from vectorstore import VECTORDB_PATH, get_or_create_collection


def init_vector_store():
    VECTORDB_PATH.mkdir(parents=True, exist_ok=True)
    collection = get_or_create_collection()
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
            metadatas=[{"source": pdf_path, "chunk_index": i}],
        )

    return len(chunks)

def verify_ingestion() -> bool:
    collection = get_or_create_collection()
    count = collection.count()
    return count > 0
