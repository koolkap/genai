"""
Abstractive Summarization using LangChain + Ollama (Local LLM)

This script generates a concise summary by sending a prompt
to a locally running LLM via Ollama.
The summary is abstractive, meaning the model creates new
sentences instead of copying from the original text.
"""

# Import PromptTemplate from langchain-core
# PromptTemplate allows defining reusable prompts with placeholders
from langchain_core.prompts import PromptTemplate

# Import the Ollama LLM wrapper from langchain-community
# This enables LangChain to interact with a local Ollama model
from langchain_community.llms import Ollama


# Initialize the local LLM
# "llama3.2:latest" refers to the model served by Ollama
# Ollama must be running locally for this to work
llm = Ollama(model="llama3.2:latest")


# Define the prompt template for summarization
# input_variables specifies which values will be injected at runtime
prompt = PromptTemplate(
    input_variables=["text"],

    # The prompt instructs the model to produce a 3-sentence summary
    template="""
    Summarize the following text in 3 concise sentences:

    {text}
    """
)


# Create an LCEL chain (LangChain Expression Language)
# The output of the prompt is piped directly into the LLM
# This replaces the deprecated LLMChain class
chain = prompt | llm


# Define the input text to summarize
# This is the unstructured source document
text = """
President Donald Trump says the US is exploring a potential deal on Greenland after talks with Nato as he backed off threats to tariff European allies that had opposed his plans for America to acquire the island.

On social media, Trump offered few details about a discussion that both he and Nato described as "very productive".

After rattling the transatlantic alliance with weeks of rhetoric, the US president said the meeting had led to the "framework" of a potential agreement.

But there was no suggestion of a deal that might meet Trump's demand for "ownership" of Greenland, an ambition he restated at the World Economic Forum in Switzerland, while also ruling out military force.

On Truth Social on Wednesday, the US president said: "We have formed the framework of a future deal with respect to Greenland and, in fact, the entire Arctic Region.

"This solution, if consummated, will be a great one for the United States of America, and all Nato Nations."

Diplomatic sources told the BBC's US partner CBS that there was no agreement for American control or ownership of the autonomous Danish dependent territory.
"""


# Invoke the chain with the input text
# The key "text" must match input_variables in the prompt
# The LLM generates a paraphrased summary
summary = chain.invoke({"text": text})


# Print the generated abstractive summary
print(summary)
