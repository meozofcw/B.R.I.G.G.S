import os, requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import rag
import memory

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

app = FastAPI(title="B.R.I.G.G.S")
memory.init_db()

class ChatIn(BaseModel):
    message: str

class RememberIn(BaseModel):
    kind: str
    content: str

def call_ollama(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/remember")
def remember(data: RememberIn):
    memory.add_memory(data.kind, data.content)
    return {"saved": True}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    os.makedirs("data/uploads", exist_ok=True)
    path = os.path.join("data/uploads", file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    rag.ingest_pdf(path, source_name=file.filename)
    return {"ingested": True, "source": file.filename}

@app.post("/chat")
def chat(data: ChatIn):
    msg = data.message

    # 1) Busca contexto na base (RAG)
    hits = rag.search(msg, k=5)
    context = "\n\n".join(
        [f"[Fonte: {m.get('source')} | Trecho {m.get('chunk')}]\n{d}" for d, m in hits]
    ) if hits else "Sem contexto recuperado."

    # 2) Puxa memórias recentes
    mems = memory.get_memories(limit=15)
    mem_txt = "\n".join([f"- ({k}) {c}" for k, c, _ in mems]) if mems else "Nenhuma memória salva."

    # 3) Prompt base simples (depois a gente refina)
    prompt = f"""
Você é uma IA domiciliar pessoal. Seja direta, prática e passo a passo.
Use o CONTEXTO e as MEMÓRIAS quando ajudarem. Se faltar dado, pergunte.

MEMÓRIAS:
{mem_txt}

CONTEXTO (RAG):
{context}

PERGUNTA:
{msg}

Responda em português. Termine com um "Próximo passo:".
"""

    reply = call_ollama(prompt)
    return {"reply": reply, "sources": [m for _, m in hits]}
