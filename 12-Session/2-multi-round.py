from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")


# -------------------------------------------
#  SECTION 2 — MULTI-ROUND ITERATION LOOP
# -------------------------------------------

def refine(draft, feedback):
    prompt = f"""
    You are an expert writing assistant. We will refine a draft iteratively.

    Draft:
    {draft}

    Feedback:
    {feedback}

    Task:
    Produce a revised version of the draft that incorporates the feedback while improving clarity, organization, and correctness. Return only the updated draft, no explanations.
    """
    return llm.invoke(prompt)

draft = "Explain quantum computing."
feedbacks = [
    "Make it simpler.",
    "Add an analogy.",
    "Shorten to under 120 words."
]

print("\n====== MULTI-ROUND REFINEMENT ======\n")
for fb in feedbacks:
    draft = refine(draft, fb)
    print(f"\n--- After Feedback: {fb} ---\n{draft}")
