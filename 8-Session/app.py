from fastapi import FastAPI
from pydantic import BaseModel
from graph import rag_app, RAGState

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    initial_state = RAGState(question=q.question)
    result = rag_app.invoke(initial_state)
    return {"answer": result["answer"]}
