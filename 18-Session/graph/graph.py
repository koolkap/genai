from langgraph.graph import StateGraph, END
from graph.state import CodeState
from graph.nodes import (
    code_generation_node,
    execution_node,
    debugging_node,
    security_review_node
)


def build_graph():
    graph = StateGraph(CodeState)

    graph.add_node("generate", code_generation_node)
    graph.add_node("execute", execution_node)
    graph.add_node("debug", debugging_node)
    graph.add_node("security", security_review_node)

    graph.set_entry_point("generate")

    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "debug")
    graph.add_edge("debug", "security")
    graph.add_edge("security", END)

    return graph.compile()
