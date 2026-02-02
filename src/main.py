import sys

from bm25 import BM25_INDEX_PATH
from ingestion import ingest_documents_dir, init_vector_store, refresh_bm25_from_vectorstore
from memory import Memory
from rag import answer_question

GREEN = "\033[92m"
RESET = "\033[0m"

def ensure_documents_loaded() -> None:
    init_vector_store()
    ingested = ingest_documents_dir()
    if ingested > 0:
        print(f"Loaded {ingested} chunks from documents.")
    else:
        print("No new documents to load.")
    if not BM25_INDEX_PATH.exists():
        rebuilt = refresh_bm25_from_vectorstore()
        if rebuilt > 0:
            print(f"Built BM25 index for {rebuilt} chunks.")

def main():
    ensure_documents_loaded()
    memory = Memory()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"{GREEN}{answer_question(query, memory)}{RESET}")
        return

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        print(f"{GREEN}{answer_question(query, memory)}{RESET}")

if __name__ == "__main__":
    main()
