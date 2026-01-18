# ingest.py

import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "realistic_restaurant_reviews.csv")
CSV_PATH = os.path.normpath(CSV_PATH)

INDEX_PATH = "index.faiss"
META_PATH = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def ingest():
    print("[+] Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    docs = (df["Title"] + " " + df["Review"]).tolist()

    print("[+] Loading embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)

    print("[+] Encoding docs...")
    embeddings = embedder.encode(docs, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("[+] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("[+] Saving metadata...")
    with open(META_PATH, "wb") as f:
        pickle.dump({"docs": docs}, f)

    print("[✓] Ingestion completed!")

if __name__ == "__main__":
    ingest()
