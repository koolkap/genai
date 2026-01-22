"""
Embeddings + Retrieval using Ollama and a local vector store (FAISS)

This script demonstrates a full Retrieval-Augmented Generation (RAG) workflow:
1. Split documents into chunks
2. Convert chunks into embeddings using a local embedding model
3. Store embeddings in a FAISS vector database
4. Retrieve relevant chunks for a user query
5. Use retrieved context to answer the question with a local LLM
"""

# Import Ollama embedding wrapper to generate embeddings locally
from langchain_community.embeddings import OllamaEmbeddings

# Import FAISS vector store for similarity search
from langchain_community.vectorstores import FAISS

# Import text splitter for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import prompt template
from langchain_core.prompts import PromptTemplate

# Import Ollama LLM wrapper
from langchain_community.llms import Ollama


# -----------------------------
# 1. Initialize Models
# -----------------------------

# Local embedding model (must be pulled in Ollama)
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest"
)

# Local LLM for answering questions
llm = Ollama(model="llama3.2:latest")


# -----------------------------
# 2. Prepare Documents
# -----------------------------

# Sample document corpus (replace with real data)
text = """
LangChain helps developers build LLM applications.
It supports agents, tools, and memory.
Ollama runs open-source LLMs locally.
Vector databases enable semantic search.
""" * 10  # repeated to simulate a larger corpus


# -----------------------------
# 3. Chunk the Documents
# -----------------------------

# Split text into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

documents = splitter.split_text(text)


# -----------------------------
# 4. Create Vector Store
# -----------------------------

# Convert text chunks into embeddings and store in FAISS
vectorstore = FAISS.from_texts(
    documents,
    embedding=embeddings
)


# -----------------------------
# 5. Create Retriever
# -----------------------------

# Retrieve top-k most relevant chunks
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 6. User Query
# -----------------------------

query = "How does Ollama help with local LLMs?"


# -----------------------------
# 7. Retrieve Relevant Chunks
# -----------------------------

# Retriever returns a list of Document objects
retrieved_docs = retriever.invoke(query)

# Extract text from Document objects
context = "\n".join(doc.page_content for doc in retrieved_docs)


# -----------------------------
# 8. Prompt with Retrieved Context
# -----------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}
    """
)


# -----------------------------
# 9. Build LCEL Chain
# -----------------------------

# Prompt → LLM
chain = prompt | llm


# -----------------------------
# 10. Generate Answer
# -----------------------------

answer = chain.invoke({
    "context": context,
    "question": query
})


# -----------------------------
# 11. Output Result
# -----------------------------

print("\nANSWER:\n")
print(answer)
