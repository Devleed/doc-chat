import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.qdrant_rag import QdrantRag, RagSettings


app = FastAPI(title="PDF RAG Server", version="0.1.0")

rag = QdrantRag(RagSettings())


class AskRequest(BaseModel):
    doc_id: str = Field(..., description="Document id returned from /ingest")
    question: str
    fetch_k: int = 6
    final_k: int = 3


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ingest")
async def ingest_pdf(file: Annotated[UploadFile, File(...)]) -> dict:
    print(f"Received file upload: {file.filename} with content type {file.content_type}")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    doc_id = rag.ingest_pdf(pdf_bytes=pdf_bytes, filename=file.filename)
    return {"doc_id": doc_id}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not req.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return rag.ask(question=req.question, doc_id=req.doc_id, fetch_k=req.fetch_k, final_k=req.final_k)


if __name__ == "__main__":
    # Convenience for local runs outside Docker.
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))

