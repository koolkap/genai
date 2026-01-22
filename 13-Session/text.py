"""
Structured Data Extraction using LangChain + Ollama (Modern)

This script uses a local LLM (via Ollama) to extract structured
information from unstructured text and return it as JSON.
It uses LangChain Expression Language (LCEL), not deprecated APIs.
"""

# Import the Ollama LLM wrapper from langchain-community
# This allows LangChain to communicate with a locally running Ollama model
from langchain_community.llms import Ollama

# Import PromptTemplate from langchain-core
# PromptTemplate lets us define reusable prompts with placeholders
from langchain_core.prompts import PromptTemplate


# Initialize the local LLM
# "llama3.2:latest" refers to the model name available in Ollama
# Ollama must be running locally for this to work
llm = Ollama(model="llama3.2:latest")


# Define the prompt template
# This tells the LLM exactly what task to perform and what format to return
prompt = PromptTemplate(
    # Declares the variable names that will be filled at runtime
    input_variables=["text"],

    # The actual prompt text sent to the model
    template="""
    Extract structured data from the text below.
    Return ONLY valid JSON.

    Fields:
    - customer_name
    - product
    - quantity
    - order_date
    - total_cost
    - delivery_eta_days

    Text:
    {text}
    """
)


# Create an LCEL chain by piping the prompt into the LLM
# This replaces the old LLMChain class
# Data flows: PromptTemplate → Ollama LLM
chain = prompt | llm


# Define the unstructured input text
# This is the raw text from which we want to extract structured data
text = """
John Smith placed an order for 2 laptops on January 10, 2026.
The total cost was $2400 and delivery is expected in 5 days.
"""


# Invoke the chain with input values
# The dictionary key "text" matches the prompt's input_variables
# The LLM processes the prompt and generates a response
response = chain.invoke({"text": text})


# Print the LLM output
# Expected output: a JSON string with the extracted fields
print(response)
