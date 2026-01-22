"""
Role-Play Simulation with Persistent Character Behavior
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:latest")

prompt = PromptTemplate(
    input_variables=["role", "scenario", "user_input"],
    template="""
    You are role-playing as: {role}.

    Scenario:
    {scenario}

    Rules:
    - Stay fully in character
    - Respond as the character would
    - Do not break role

    User says:
    {user_input}
    """
)

chain = prompt | llm

response = chain.invoke({
    "role": "a medieval fantasy innkeeper",
    "scenario": "A weary traveler arrives during a storm",
    "user_input": "I need a room and information."
})

print("\nROLE-PLAY RESPONSE:\n")
print(response)
