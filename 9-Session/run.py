from rag.graph import rag_app, RAGState


if __name__ == "__main__":
    question = "Do they serve gluten free pizza?"

    for mode in ["zero", "few"]:
        initial = RAGState(question=question, mode=mode)
        result = rag_app.invoke(initial)
        print(f"\n=== {mode.upper()} SHOT RESULT ===\n")
        print(result["answer"])
