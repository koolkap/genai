from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")


# -------------------------------------------
#  SECTION 1 — BASIC ITERATIVE REFINEMENT
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


# Example usage for iterative refinement
draft = "The product is good but the explanation is unclear."
feedback = "Make it more persuasive and add an example."

refined = refine(draft, feedback)
print("\n====== ITERATIVE REFINEMENT RESULT ======\n")
print(refined)

