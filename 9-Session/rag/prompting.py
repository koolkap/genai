def zero_shot_prompt(context, question):
    return f"""
You are a helpful restaurant assistant.

Use the context to answer the question.

Context:
{context}

Question: {question}
Answer:
"""

def few_shot_prompt(context, question):
    examples = """
Q: Do they have vegan pizza?
A: Yes, one customer mentioned vegan cheese options.

Q: Do they have outdoor seating?
A: Yes, multiple reviews mentioned patio seating.
"""
    return f"""
You are a helpful restaurant assistant.
Follow the example answer style.

Examples:
{examples}

Context:
{context}

Q: {question}
A:
"""
