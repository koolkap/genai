"""
Dialogue Generation with Character Personalities
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:latest")

prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    Generate a natural dialogue based on the situation below.

    Rules:
    - Distinct personalities
    - Natural back-and-forth
    - No narration, only dialogue

    Situation:
    {context}
    """
)

chain = prompt | llm

dialogue = chain.invoke({
    "context": "A senior engineer mentoring a junior developer who just broke production"
})

print("\nDIALOGUE:\n")
print(dialogue)
