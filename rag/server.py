import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.qdrant_rag import QdrantRag, RagSettings
from rag.review import review_doc


app = FastAPI(title="PDF RAG Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for dev
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = QdrantRag(RagSettings())


class AskRequest(BaseModel):
    doc_id: str = Field(..., description="Document id returned from /ingest")
    question: str
    fetch_k: int = 6
    final_k: int = 3


@app.get("/health")
def health() -> dict:
    return {"ok": True}


_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _check_extension(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    ext = os.path.splitext(filename.lower())[1]
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )
    return ext


@app.post("/ingest")
async def ingest_document(file: Annotated[UploadFile, File(...)]) -> dict:
    print(f"Received file upload: {file.filename} with content type {file.content_type}")
    _check_extension(file.filename)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    doc_id = rag.ingest_document(file_bytes=file_bytes, filename=file.filename)

    print(f"Ingested document with doc_id: {doc_id}")

    return {"doc_id": doc_id}


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    if not req.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return StreamingResponse(
        rag.ask_stream(question=req.question, doc_id=req.doc_id, fetch_k=req.fetch_k, final_k=req.final_k),
        media_type="text/event-stream",
    )


@app.post("/review")
async def review_document(file: Annotated[UploadFile, File(...)]) -> dict:
    _check_extension(file.filename)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    return review_doc(file_bytes=file_bytes, filename=file.filename)


if __name__ == "__main__":
    # Convenience for local runs outside Docker.
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))

