from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from config.models import EMBED_MODEL


def build_vector_db(text_file_path):
    """
    Builds FAISS vector database from text file
    """

    # 1. Load text
    loader = TextLoader(text_file_path)
    documents = loader.load()

    # 2. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 4. Build FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 5. Save FAISS index
    vector_db.save_local("faiss_index")

    return vector_db
