import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

CSV_PATH = "realistic_restaurant_reviews.csv"
INDEX_PATH = "index.faiss"
META_PATH = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

print("[+] Loading CSV...")
df = pd.read_csv(CSV_PATH)

documents = (df["Title"] + " " + df["Review"]).tolist()

print("[+] Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("[+] Encoding...")
embs = model.encode(documents, show_progress_bar=True)

dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump({"docs": documents}, f)

print("[✓] Done!")
