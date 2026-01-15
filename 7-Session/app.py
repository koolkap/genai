from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI()

model = OllamaLLM(model="llama3.2")

template = """
You are an expert answering questions about a pizza restaurant.

Relevant reviews:
{reviews}

User question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: QueryRequest):
    reviews = retriever.invoke(data.question)
    answer = chain.invoke({"reviews": reviews, "question": data.question})
    return {"answer": answer}
