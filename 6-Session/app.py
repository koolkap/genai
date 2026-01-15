import faiss
import pickle
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

INDEX_PATH = "index.faiss"
META_PATH = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Load
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
documents = meta["docs"]

embedder = SentenceTransformer(MODEL_NAME)
app = FastAPI()

class Query(BaseModel):
    question: str

def query_ollama(prompt: str, model="llama3.2"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload)
    data = resp.json()
    return data.get("response", "")


@app.post("/ask")
async def ask(data: Query):
    q = data.question

    q_emb = embedder.encode([q])
    D, I = index.search(q_emb, k=5)
    
    context = "\n\n".join(documents[i] for i in I[0])

    prompt = f"""You are a helpful assistant answering questions about a pizza restaurant.

Context:
{context}

Question: {q}

Answer:"""

    answer = query_ollama(prompt)
    return {"answer": answer}
