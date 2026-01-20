import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("knowb")

def _chunk(text: str, chunk_words=350):
    words = text.split()
    return [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

def ingest_pdf(path: str, source_name: str):
    reader = PdfReader(path)
    full = "\n".join([(p.extract_text() or "") for p in reader.pages])
    chunks = [c.strip() for c in _chunk(full) if c.strip()]
    embs = EMBED_MODEL.encode(chunks).tolist()

    ids = [f"{source_name}:{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embs)

def search(query: str, k=5):
    q_emb = EMBED_MODEL.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return list(zip(docs, metas))
