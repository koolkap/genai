from langgraph.graph import StateGraph, END
from graph.state import TableState
from graph.nodes import schema_node, descriptive_node, answer_node


def build_graph():
    graph = StateGraph(TableState)

    graph.add_node("schema", schema_node)
    graph.add_node("describe", descriptive_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("schema")

    graph.add_edge("schema", "describe")
    graph.add_edge("describe", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
