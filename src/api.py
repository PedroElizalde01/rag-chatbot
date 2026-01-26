import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion import ingest_pdf
from memory import Memory
from rag import answer_question
from vectorstore import get_or_create_collection

app = FastAPI(title="RAG Chatbot API", version="1.0")
memories_by_session: dict[str, Memory] = {}

@app.get("/")
def root() -> dict[str, str]:
    return {"mensaje": "Hello! The API is working"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0"}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.post("/query")
def query(request: QuestionRequest) -> dict[str, str]:
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        memory = memories_by_session.setdefault(request.session_id, Memory())
        answer = answer_question(request.question, memory)
        return {"answer": answer, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {exc}"
        ) from exc


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    suffix = Path(file.filename).suffix
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)

        chunk_count = ingest_pdf(str(temp_path))
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {exc}"
        ) from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()


@app.get("/metrics")
def metrics() -> dict[str, int]:
    try:
        collection = get_or_create_collection()
        total_chunks = collection.count()
        items = collection.get()
        unique_documents: set[str] = set()
        for metadata in items.get("metadatas", []) or []:
            source = metadata.get("source") if metadata else None
            if source:
                unique_documents.add(source)
        return {
            "total_documents": len(unique_documents),
            "total_chunks": total_chunks,
            "active_sessions": len(memories_by_session),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error getting metrics: {exc}"
        ) from exc

