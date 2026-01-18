from langgraph.graph import StateGraph, END
from .retrieval import retrieve_context
from .prompting import zero_shot_prompt, few_shot_prompt
from .llm import ollama_generate


class RAGState(dict):
    question: str
    mode: str            # "zero" or "few"
    context: str
    prompt: str
    answer: str

def node_retrieve(state: RAGState):
    state["context"] = "\n\n".join(retrieve_context(state["question"]))
    return state

def node_build_prompt(state: RAGState):
    q, ctx, mode = state["question"], state["context"], state["mode"]
    if mode == "zero":
        state["prompt"] = zero_shot_prompt(ctx, q)
    else:
        state["prompt"] = few_shot_prompt(ctx, q)
    return state

def node_generate(state: RAGState):
    state["answer"] = ollama_infer(state["prompt"])
    return state

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("prompt", node_build_prompt)
workflow.add_node("generate", node_generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "prompt")
workflow.add_edge("prompt", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()
