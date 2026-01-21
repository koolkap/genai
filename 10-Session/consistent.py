from langchain_ollama import OllamaLLM
from collections import Counter

# Instantiate local ollama LLM
llm = OllamaLLM(model="llama3.2:3b")   # change to your model

def ask_reasoning(prompt, n=5):
    """Ask model for multiple reasoning paths"""
    responses = []
    reasoning_prompt = (
        "Solve this step by step carefully, showing your reasoning:\n\n"
        + prompt +
        "\n\nFinal answer should be clearly stated after reasoning."
    )
    
    for _ in range(n):
        out = llm.invoke(reasoning_prompt)  # invoke() instead of __call__()
        responses.append(out)
    return responses

def extract_answers(responses):
    """Extract final numeric or text answer (simple heuristic)"""
    answers = []
    for r in responses:
        lines = r.strip().split("\n")
        # last non-empty line assumed to contain final answer
        final = next((l for l in reversed(lines) if l.strip()), "")
        answers.append(final)
    return answers

def self_consistency(prompt, trials=5):
    reasoning_paths = ask_reasoning(prompt, trials)
    answers = extract_answers(reasoning_paths)

    # vote for most frequent
    counter = Counter(answers)
    best = counter.most_common(1)[0]

    return {
        "all_reasoning": reasoning_paths,
        "final_votes": counter,
        "final_answer": best[0],
        "vote_count": best[1],
    }

# Example usage
result = self_consistency(
    "If John has 3 apples and buys 4 more, how many apples does he have?",
    trials=7
)

print("=== Final Selected Answer ===")
print(result["final_answer"])
print("\n=== Vote Distribution ===")
print(result["final_votes"])
