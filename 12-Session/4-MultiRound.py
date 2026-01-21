from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")
# -------------------------------------------
#  SECTION 4 — MULTI-ROUND PROMPT DEBUGGING LOOP
# -------------------------------------------
def debug_prompt(prompt, feedback):
    debug_template = f"""
We are debugging a prompt.

Prompt to debug:
{prompt}

Observed issues:
{feedback}

Task:
Produce an improved prompt that fixes the issues and generates higher-quality results.

Return only the improved prompt.
"""
    return llm.invoke(debug_template)

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

prompt = "write a business plan"
feedbacks = [
    "too generic",
    "missing detailed sections",
    "needs numbers and realistic estimates",
    "needs structured formatting and clarity"
]

print("\n====== MULTI-ROUND PROMPT DEBUGGING ======\n")
for fb in feedbacks:
    prompt = debug_prompt(prompt, fb)
    print(f"\n--- After Feedback: {fb} ---\n{prompt}")
