import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "index.faiss"))
META_PATH  = os.path.normpath(os.path.join(BASE_DIR, "..", "embeddings.pkl"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

docs = meta["docs"]

def retrieve_context(query: str, k: int = 5):
    q_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]
