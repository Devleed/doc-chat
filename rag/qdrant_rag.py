import os
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

import cohere
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi


def _clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class RagSettings:
    qdrant_url: str = _env("QDRANT_URL", default="http://qdrant:6333") or "http://qdrant:6333"
    collection_name: str = _env("QDRANT_COLLECTION", default="documents") or "documents"
    embedding_model: str = _env("EMBEDDING_MODEL", default="text-embedding-3-small") or "text-embedding-3-small"
    llm_model: str = _env("LLM_MODEL", default="gpt-4o") or "gpt-4o"
    openai_api_key: str | None = _env("OPENAI_API_KEY", "OPEN_AI_KEY")
    cohere_api_key: str | None = _env("COHERE_API_KEY")
    chunk_size: int = int(_env("CHUNK_SIZE", default="500") or "500")
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", default="50") or "50")


class QdrantRag:
    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or RagSettings()
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        self.cohere = cohere.Client(self.settings.cohere_api_key) if self.settings.cohere_api_key else None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.settings.collection_name in existing:
            return
        # text-embedding-3-small -> 1536 dims
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    def _extract_pdf_pages(self, pdf_bytes: bytes, source_name: str) -> list[Document]:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(Document(page_content=text, metadata={"page": i + 1, "source": source_name}))
        return pages

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        doc_id = str(uuid.uuid4())
        print(f"Ingesting {filename} as doc_id {doc_id} with size {len(pdf_bytes)} bytes")

        pages = self._extract_pdf_pages(pdf_bytes, source_name=filename)
        chunks = self.splitter.split_documents(pages)

        print(f"Extracted {len(pages)} pages and split into {len(chunks)} chunks")

        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        point_ids: list[str] = []

        print("Processing chunks and preparing for embedding...")
        for chunk_index, doc in enumerate(chunks):
            cleaned = _clean_text(doc.page_content)
            if not cleaned:
                continue
            texts.append(cleaned)
            payloads.append(
                {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "text": cleaned,
                }
            )
            point_ids.append(str(uuid.uuid4()))

        if not texts:
            return doc_id
        
        print(f"Embedding {len(texts)} chunks...")

        vectors = self.embeddings.embed_documents(texts)
        points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(point_ids, vectors, payloads, strict=False)
        ]

        print(f"Upserting {len(points)} points into Qdrant collection '{self.settings.collection_name}'...")

        self.client.upsert(collection_name=self.settings.collection_name, points=points)
        return doc_id

    def _scroll_doc_chunks(self, doc_id: str, limit: int = 10_000) -> list[dict[str, Any]]:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        offset = None
        out: list[dict[str, Any]] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                scroll_filter=flt,
                limit=min(limit, 1024),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                if p.payload:
                    out.append(dict(p.payload))
            if offset is None or len(out) >= limit:
                break
        return out

    def _semantic_search(self, query: str, doc_id: str, k: int) -> list[dict[str, Any]]:
        query_vec = self.embeddings.embed_query(query)
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        result = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=query_vec,
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
        out: list[dict[str, Any]] = []
        for h in result.points:
            payload = dict(h.payload or {})
            payload["score"] = float(getattr(h, "score", 0.0))
            out.append(payload)
        return out

    def _keyword_search(self, query: str, doc_chunks: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        texts = [(c.get("text") or "") for c in doc_chunks]
        tokenized_chunks = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query.lower().split())
        best_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [doc_chunks[i] for i in best_indices]

    def _rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not self.cohere:
            return candidates[:top_n]
        try:
            rerank = self.cohere.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[c.get("text", "") for c in candidates],
                top_n=min(top_n, len(candidates)),
            )
            return [candidates[r.index] for r in rerank.results]
        except Exception:
            return candidates[:top_n]

    def retrieve(self, query: str, doc_id: str, fetch_k: int = 6, final_k: int = 3) -> list[dict[str, Any]]:
        semantic = self._semantic_search(query=query, doc_id=doc_id, k=fetch_k)
        doc_chunks = self._scroll_doc_chunks(doc_id=doc_id)
        keyword = self._keyword_search(query=query, doc_chunks=doc_chunks, k=fetch_k) if doc_chunks else []

        seen: set[str] = set()
        combined: list[dict[str, Any]] = []
        for item in list(semantic) + list(keyword):
            text = (item.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            combined.append(item)
        return self._rerank(query=query, candidates=combined, top_n=final_k)

    def ask(self, question: str, doc_id: str, fetch_k: int = 6, final_k: int = 3) -> dict[str, Any]:
        llm = ChatOpenAI(model=self.settings.llm_model, temperature=0, openai_api_key=self.settings.openai_api_key)
        prompt = ChatPromptTemplate.from_template(
            """
You are a document assistant.
Answer the question using ONLY the document excerpts below.

For every claim you make, cite the chunk like this: [Chunk 1], [Chunk 2].
If the answer is truly not in the excerpts, say "This information is not in the document."

Document excerpts:
{context}

Question: {question}
""".strip()
        )

        results = self.retrieve(query=question, doc_id=doc_id, fetch_k=fetch_k, final_k=final_k)
        context = "\n\n".join([f"[Chunk {i+1}]: {r.get('text','')}" for i, r in enumerate(results)])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return {
            "answer": getattr(response, "content", str(response)),
            "sources": results,
        }

