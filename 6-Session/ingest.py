import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

CSV_PATH = "realistic_restaurant_reviews.csv"
INDEX_PATH = "index.faiss"
META_PATH = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

print("[+] Loading data...")
df = pd.read_csv(CSV_PATH)

documents = (df["Title"] + " " + df["Review"]).tolist()

print("[+] Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("[+] Encoding documents...")
embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump({"docs": documents}, f)

print("[✓] Index built and saved!")
