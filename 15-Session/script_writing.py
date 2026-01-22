"""
Script Writing with Dialogue and Scene Directions
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:latest")

prompt = PromptTemplate(
    input_variables=["setting", "characters"],
    template="""
    Write a short movie scene.

    Setting:
    {setting}

    Characters:
    {characters}

    Format:
    - Scene heading
    - Action descriptions
    - Dialogue in screenplay format
    """
)

chain = prompt | llm

scene = chain.invoke({
    "setting": "A rain-soaked rooftop in a cyberpunk city",
    "characters": "A detective nearing retirement and a rogue AI hologram"
})

print("\nSCREENPLAY SCENE:\n")
print(scene)
