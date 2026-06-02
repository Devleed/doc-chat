import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import cohere
import re

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)       # remove excessive newlines
    text = re.sub(r'  +', ' ', text)       # remove double spaces
    return text.strip()

def load_pdf():
    loader = PyPDFLoader("files/Fullstack_Resume_v4_Waleed.pdf")
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages

def split_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = max 500 characters
        chunk_overlap=50,     # chunks overlap by 50 chars so context isn't cut off
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_bm25(chunks):
    tokenized_chunks = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

def create_qdrant_collection():
    client = QdrantClient(host="localhost", port=6333)
    client.delete_collection("resume")
    existing = [c.name for c in client.get_collections().collections]
    if "resume" not in existing:
        client.create_collection(
            collection_name="resume",
            vectors_config=VectorParams(
                size=1536,        # OpenAI text-embedding-3-small outputs 1536 numbers per chunk
                distance=Distance.COSINE  # how similarity is measured
            )
        )
    return client

def embed_and_store(chunks, client):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",  openai_api_key=os.getenv("OPEN_AI_KEY"))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="resume",
        embedding=embeddings,
    )
    # Apply before embedding
    for doc in chunks:
        doc.page_content = clean_text(doc.page_content)
    vector_store.add_documents(chunks)
    print("\n✅ Resume embedded and stored in Qdrant")
    return vector_store

def hybrid_search(query: str, fetch_k: int = 6, final_k: int = 3):
    # Semantic results
    semantic_results = vector_store.similarity_search(query, k=fetch_k)
    
    # Keyword results
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)
    top_keyword_indices = sorted(
        range(len(keyword_scores)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )[:fetch_k]
    keyword_results = [chunks[i] for i in top_keyword_indices]

    # Merge
    seen = set()
    combined = []
    for doc in semantic_results + keyword_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    # Step 4: rerank — score each chunk against the query
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[doc.page_content for doc in combined],
        top_n=final_k  # return only best 3
    )

    # Step 5: pick top chunks by reranker score
    reranked = [combined[r.index] for r in rerank_response.results]

    print(f'Combined[0].page_content: {combined[0].page_content}')

    print("\n── Chunks sent to reranker ──")
    for i, doc in enumerate(combined):
        print(f"Chunk {i}: {doc.page_content[:80]}")

    print("\n── Reranker scores ──")
    for r in rerank_response.results:
        print(f"Chunk {r.index} score {r.relevance_score:.3f}: {combined[r.index].page_content[:80]}")


    return reranked

# ── Q&A function ──────────────────────────────────────────────────────────────
def ask(question: str):
    global bm25, chunks, vector_store

    pages = load_pdf()
    chunks = split_chunks(pages)
    bm25 = create_bm25(chunks)
    client = create_qdrant_collection()
    vector_store = embed_and_store(chunks, client)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPEN_AI_KEY"))

    # ── Prompt ────────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_template("""
You are a resume assistant helping evaluate a candidate.
Answer the question using ONLY the resume excerpts below.

Important: The resume describes what the candidate built and worked on.
If asked about a company or project name, describe what the candidate did there.
Do not look for definitions — look for work experience.

For every claim you make, cite the chunk like this: [Chunk 1], [Chunk 2].
If the answer is truly not in the excerpts, say "This information is not in the resume."

Resume excerpts:
{context}

Question: {question}
""")

    # Step 1: retrieve relevant chunks
    results = hybrid_search(question)

    # Step 2: format chunks with labels so LLM can cite them
    context = "\n\n".join([
        f"[Chunk {i+1}]: {doc.page_content}"
        for i, doc in enumerate(results)
    ])

    # Step 3: LLM answers using only those chunks
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\nQ: {question}")
    print(f"\nA: {response.content}")
    print("\n── Sources used ──")

# ── Test it ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ask("What is BullionFX?")
    # ask("Is the dev familiar in Rust?")
    # ask("Can the dev speak English? Does he have any certifications?")
