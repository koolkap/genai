from graph.graph import build_graph


def main():
    app = build_graph()

    result = app.invoke({
        "request": "Write a Python function that divides two numbers."
    })

    print("\n--- GENERATED / FIXED CODE ---\n")
    print(result["code"])

    print("\n--- SECURITY REVIEW ---\n")
    print(result["security_review"])


if __name__ == "__main__":
    main()
