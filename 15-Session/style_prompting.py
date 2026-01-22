"""
Style & Tone Prompting (Safe Style Transfer)
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:latest")

prompt = PromptTemplate(
    input_variables=["topic", "style"],
    template="""
    Write a paragraph about the topic below.

    Topic:
    {topic}

    Style Guidelines:
    {style}

    Avoid referencing real authors or works.
    """
)

chain = prompt | llm

output = chain.invoke({
    "topic": "artificial intelligence in healthcare",
    "style": "clear, authoritative, slightly optimistic, non-technical"
})

print("\nSTYLED OUTPUT:\n")
print(output)
