from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# Initialize Ollama (ensure you have a model pulled, e.g. "llama3" or "mistral")
llm = OllamaLLM(model="llama3.2:3b")

# Chain-of-thought prompt example
template = """
You are an expert reasoning assistant.
Think step-by-step and explain your reasoning clearly before giving the final answer.

Question: {question}

First, explain your reasoning.
Then provide the final answer labeled as "Answer:".
"""

prompt = PromptTemplate.from_template(template)

question = "If I have 3 apples and buy 4 more, then give 2 away, how many do I have?"

formatted = prompt.format(question=question)

response = llm.invoke(formatted)

print(response)
