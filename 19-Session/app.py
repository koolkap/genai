import requests
import json
import re
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"


# -------------------------
# Ollama Client
# -------------------------

def ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()


# -------------------------
# Academic Writing Tasks
# -------------------------

def summarize(text: str) -> str:
    prompt = f"""
You are an academic writing assistant.

Task: Summarize the following text clearly and concisely.
Rules:
- Preserve original meaning
- No new information
- No citations

Text:
{text}
"""
    return ollama_generate(prompt)


def paraphrase(text: str) -> str:
    prompt = f"""
You are an academic paraphrasing assistant.

Task: Paraphrase the following text.
Rules:
- Preserve meaning
- Change structure and wording substantially
- Do NOT quote the original
- Do NOT add citations or references

Text:
{text}
"""
    return ollama_generate(prompt)


def adapt_style(text: str, style: str = "formal academic") -> str:
    prompt = f"""
You are an expert academic editor.

Task: Rewrite the text in {style} style.
Rules:
- Maintain meaning
- Improve clarity and cohesion
- No citations unless explicitly provided

Text:
{text}
"""
    return ollama_generate(prompt)


# -------------------------
# Plagiarism Risk Estimation
# -------------------------

def plagiarism_similarity(original: str, generated: str) -> float:
    vectorizer = CountVectorizer(
        ngram_range=(3, 5),
        stop_words="english"
    )
    vectors = vectorizer.fit_transform([original, generated])
    similarity = cosine_similarity(vectors)[0, 1]
    return float(similarity)


# -------------------------
# Hallucinated Citation Detection
# -------------------------

def detect_hallucinated_citations(text: str) -> List[str]:
    """
    Flags suspicious citation-like patterns:
    - (Author, 2023)
    - [12]
    - DOI-like strings
    """
    patterns = [
        r"\([A-Z][a-z]+,\s?\d{4}\)",
        r"\[\d+\]",
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    ]

    findings = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        findings.extend(matches)

    return findings


# -------------------------
# End-to-End Analysis
# -------------------------

def analyze_academic_output(original: str, generated: str) -> dict:
    return {
        "plagiarism_similarity_score": plagiarism_similarity(original, generated),
        "possible_hallucinated_citations": detect_hallucinated_citations(generated)
    }


# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    original_text = """
Large language models have become increasingly influential in academic writing,
raising concerns about originality, plagiarism, and factual accuracy.
"""

    summary = summarize(original_text)
    paraphrased = paraphrase(original_text)
    styled = adapt_style(original_text, "high-impact journal academic")

    analysis = analyze_academic_output(original_text, paraphrased)

    print("SUMMARY:\n", summary)
    print("\nPARAPHRASED:\n", paraphrased)
    print("\nSTYLE-ADAPTED:\n", styled)
    print("\nRISK ANALYSIS:\n", json.dumps(analysis, indent=2))
