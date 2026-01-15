from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd
import os
import pickle

CSV_PATH = "realistic_restaurant_reviews.csv"
FAISS_PATH = "faiss_index"
META_PATH = "faiss_meta.pkl"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def load_or_build_vectorstore():
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            docs = pickle.load(f)
        return FAISS.load_local(FAISS_PATH, embeddings), docs
    
    df = pd.read_csv(CSV_PATH)
    docs = []

    for i, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
            )
        )

    store = FAISS.from_documents(docs, embeddings)
    store.save_local(FAISS_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    return store, docs


vectorstore, docs = load_or_build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
