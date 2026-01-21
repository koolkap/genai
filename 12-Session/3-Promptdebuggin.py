from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")

# -------------------------------------------
#  SECTION 3 — PROMPT DEBUGGING (IMPROVING PROMPTS)
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


initial_prompt = "write a business plan"
feedback = "too generic, lacks structured sections and depth"

improved_prompt = debug_prompt(initial_prompt, feedback)

print("\n====== PROMPT DEBUGGING RESULT ======\n")
print("Original Prompt:", initial_prompt)
print("\nImproved Prompt:", improved_prompt)

