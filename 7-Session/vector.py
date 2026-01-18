# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import os
# import pandas as pd

# df = pd.read_csv("realistic_restaurant_reviews.csv")
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chrome_langchain_db"
# add_documents = not os.path.exists(db_location)

# if add_documents:
#     documents = []
#     ids = []
    
#     for i, row in df.iterrows():
#         document = Document(
#             page_content=row["Title"] + " " + row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#             id=str(i)
#         )
#         ids.append(str(i))
#         documents.append(document)
        
# vector_store = Chroma(
#     collection_name="restaurant_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )

# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)
    
# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 5}
# )

# vectorstore, docs = None, None#

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pandas as pd
import os
import pickle

CSV_PATH = "realistic_restaurant_reviews.csv"
FAISS_PATH = "faiss_index"
META_PATH = "faiss_meta.pkl"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def load_or_build_vectorstore():
    # Load if exists
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            docs = pickle.load(f)

        store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # NEW
        )

        return store, docs
    
    # Build new
    df = pd.read_csv(CSV_PATH)
    docs = []

    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=f"{row['Title']} {row['Review']}",
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

