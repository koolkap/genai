import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# Load FAISS + metadata
INDEX_PATH = "index.faiss"
META_PATH = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

documents = meta["docs"]
embedder = SentenceTransformer(MODEL_NAME)

# ---- STATE ----
class RAGState(dict):
    question: str
    query_embedding: list
    retrieved_docs: list
    answer: str

# ---- NODES ----
def embed_query(state: RAGState):
    q = state["question"]
    emb = embedder.encode([q])
    state["query_embedding"] = emb
    return state

def retrieve(state: RAGState):
    D, I = index.search(state["query_embedding"], k=5)
    docs = [documents[i] for i in I[0]]
    state["retrieved_docs"] = docs
    return state

def query_ollama(prompt, model="llama3.2"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload)
    return resp.json().get("response", "")

def generate(state: RAGState):
    ctx = "\n\n".join(state["retrieved_docs"])
    prompt = f"""
You are a restaurant assistant.

Context:
{ctx}

Question:
{state["question"]}

Answer:
"""
    answer = query_ollama(prompt)
    state["answer"] = answer
    return state

# ---- GRAPH ----
workflow = StateGraph(RAGState)

workflow.add_node("embed_query", embed_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("embed_query")
workflow.add_edge("embed_query", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()
