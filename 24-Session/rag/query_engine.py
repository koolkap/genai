from langchain_community.vectorstores import FAISS

def query_vector_db(db, query):
    docs = db.similarity_search(query, k=1)
    return docs[0].page_content
