from pathlib import Path

from bm25 import rebuild_bm25_index, upsert_bm25_documents
from config import CHUNK_OVERLAP, CHUNK_SIZE
from embeddings import embed_text
from loader import read_pdf, chunk_text
from vectorstore import VECTORDB_PATH, get_or_create_collection

DOCUMENTS_DIR = Path(__file__).parent.parent / "data" / "documents"

def init_vector_store():
    VECTORDB_PATH.mkdir(parents=True, exist_ok=True)
    collection = get_or_create_collection()
    return collection


def ingest_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    text = read_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    collection = get_or_create_collection()
    pdf_name = Path(pdf_path).stem

    ids = []
    embeddings = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        ids.append(f"{pdf_name}_chunk_{i}")
        embeddings.append(embed_text(chunk))
        metadatas.append({"source": pdf_path, "chunk_index": i})

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        upsert_bm25_documents(ids, chunks, metadatas)

    return len(chunks)

def is_pdf_ingested(pdf_path: str) -> bool:
    collection = get_or_create_collection()
    pdf_name = Path(pdf_path).stem
    try:
        existing = collection.get(ids=[f"{pdf_name}_chunk_0"])
        return len(existing.get("ids", [])) > 0
    except Exception:
        return False

def ingest_documents_dir(documents_dir: Path = DOCUMENTS_DIR) -> int:
    if not documents_dir.exists():
        return 0

    total_chunks = 0
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        if is_pdf_ingested(str(pdf_path)):
            continue
        total_chunks += ingest_pdf(str(pdf_path))

    return total_chunks


def refresh_bm25_from_vectorstore() -> int:
    collection = get_or_create_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    if not ids:
        return 0
    return rebuild_bm25_index(ids, documents, metadatas)
