from langchain_community.llms import Ollama
from config.models import LLM_MODEL

def generate_answer(context, question):
    llm = Ollama(model=LLM_MODEL)
    prompt = f"""
    Use the following context to answer:

    {context}

    Question: {question}
    """
    return llm.invoke(prompt)
