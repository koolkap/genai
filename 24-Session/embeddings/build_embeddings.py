from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from loaders.text_loader import load_text
from config.models import EMBED_MODEL

def build_vector_db(text_file):
    docs = load_text(text_file)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

    return db
