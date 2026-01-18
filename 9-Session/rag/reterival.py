import faiss
import pickle
from sentence_transformers import SentenceTransformer

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..",  "index.faiss")
print( INDEX_PATH)
INDEX_PATH = os.path.normpath(INDEX_PATH)

META_PATH = os.path.join(BASE_DIR, "..", "embeddings.pkl")
META_PATH = os.path.normpath(META_PATH)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

docs = meta["docs"]

def retrieve_context(query, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]
