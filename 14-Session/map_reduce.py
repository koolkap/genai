"""
Map-Reduce Summarization using local Ollama LLM

This script demonstrates how to summarize very large documents
by splitting them into chunks (Map step), summarizing each chunk,
and then combining those summaries into a final summary (Reduce step).
This avoids LLM token limit issues.
"""

# Import a text splitter that intelligently breaks long text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import PromptTemplate to define reusable prompts with variables
from langchain_core.prompts import PromptTemplate

# Import Ollama LLM wrapper to use a local language model
from langchain_community.llms import Ollama


# Initialize the local LLM
# "llama3.2:latest" must be available and running in Ollama
llm = Ollama(model="llama3.2:latest")


# Create a text splitter instance
# chunk_size: maximum number of characters per chunk
# chunk_overlap: overlapping characters to preserve context between chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# Define a long document to summarize
# In real applications, this could be a file, article, or report
document = """
LangChain enables developers to build LLM-powered applications.
It provides abstractions for prompts, chains, tools, and memory.
Ollama enables running open-source models locally.
""" * 30  # Repeated to simulate a large document


# Split the large document into smaller chunks
# Each chunk is small enough to fit within the LLM’s context window
chunks = splitter.split_text(document)


# Define the MAP prompt
# This prompt summarizes each individual chunk independently
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text concisely:\n{text}"
)


# Create the MAP chain
# Each chunk will be passed through this prompt and the LLM
map_chain = map_prompt | llm


# MAP step
# Iterate over all chunks and summarize each one
# The result is a list of short summaries (one per chunk)
chunk_summaries = [
    map_chain.invoke({"text": chunk})
    for chunk in chunks
]


# Define the REDUCE prompt
# This prompt combines all chunk-level summaries into a single summary
reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="Combine the following summaries into one coherent summary:\n{text}"
)


# Create the REDUCE chain
# This chain takes the c
