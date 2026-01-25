from graph.graph import build_graph


def main():
    app = build_graph()

    result = app.invoke({
        "question": "What can you tell me about sales patterns?"
    })

    print("\n--- FINAL ANSWER ---\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
