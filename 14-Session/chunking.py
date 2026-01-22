"""
Chunking large text to mitigate token limits

This script demonstrates how to split large text into smaller chunks
so that each chunk fits within an LLM's context window.
Chunking is a foundational technique for summarization, RAG, and retrieval.
"""

# Import RecursiveCharacterTextSplitter from LangChain's text splitters
# This splitter intelligently breaks text while preserving sentence structure
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Define a sample input text
# In real applications, this could come from a file, database, or API
text = """
LangChain is a framework for building applications powered by large language models.
It supports chaining, tools, agents, and memory.
Ollama allows running models locally without cloud APIs.
""" * 20  # Repeat the text to simulate a large document


# Create a RecursiveCharacterTextSplitter instance
# chunk_size: maximum number of characters per chunk
# chunk_overlap: number of overlapping characters between chunks
# Overlap helps preserve context across chunk boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# Split the large text into smaller chunks
# The result is a list of text chunks, each within the specified size limit
chunks = splitter.split_text(text)


# Iterate over each chunk and print it
# enumerate() provides both the index and the chunk content
for i, chunk in enumerate(chunks):
    # Print chunk number (1-based index) and its content
    print(f"\n--- Chunk {i+1} ---\n{chunk}")
