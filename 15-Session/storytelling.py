"""
Creative Storytelling with Style & Genre Prompting
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Initialize local LLM
llm = Ollama(model="llama3.2:latest")

# Storytelling prompt with style control
prompt = PromptTemplate(
    input_variables=["genre", "style", "theme"],
    template="""
    Write a short story in the {genre} genre.
    Writing style: {style}.
    Theme: {theme}.

    Requirements:
    - Strong opening hook
    - Vivid imagery
    - Clear beginning, middle, and end
    """
)

chain = prompt | llm

story = chain.invoke({
    "genre": "science fiction",
    "style": "cinematic and emotional",
    "theme": "humanity's last message to Earth"
})

print("\nSTORY:\n")
print(story)
