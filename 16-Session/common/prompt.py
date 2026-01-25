from langchain.prompts import PromptTemplate

def get_prompt():
    return PromptTemplate(
        input_variables=["history", "input"],
        template="""
You are a helpful assistant.

Conversation so far:
{history}

User: {input}
Assistant:
"""
    )
