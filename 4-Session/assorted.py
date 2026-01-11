import os
import pymupdf
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ------------------------------------
# CONFIG
# ------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL = "deepseek-r1:1.5b"

PDF_PATH = "The Almanack of Naval Ravikant PDF.pdf"
TOPIC = "Money"

# Azure setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# ------------------------------------
# STAGE-1: Extract PDF
# ------------------------------------
def extract_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + " "
    doc.close()
    return text


# ------------------------------------
# STAGE-2: DeepSeek local processing
# ------------------------------------
def call_deepseek(prompt: str) -> str:
    payload = { "model": DEEPSEEK_MODEL, "prompt": prompt, "stream": False }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json().get("response", "")


# ------------------------------------
# STAGE-3: Azure OpenAI refinement
# ------------------------------------
def call_azure_gpt(content: str) -> str:
    completion = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional book summarization assistant. "
                    "Write clearly, concisely, and avoid chain-of-thought reasoning."
                )
            },
            {"role": "user", "content": content}
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()


# ------------------------------------
# MAIN PIPELINE
# ------------------------------------
def main():

    print("📘 Loading PDF...")
    book_text = extract_pdf(PDF_PATH)
    print(f"Extracted {len(book_text)} characters from PDF.")

    print("\n🧩 Stage-1 — DeepSeek topic extraction\n")
    ds_prompt = f"""
From the provided manuscript, extract passages strongly related to the topic: '{TOPIC}'.

Return them as a clean bullet list. No chain-of-thought. No reasoning traces.

Manuscript:
{book_text}
"""
    extracted = call_deepseek(ds_prompt)
    print(extracted[:1000])  # show preview

    print("\n🧩 Stage-2 — Azure GPT summarization\n")
    gpt_prompt = f"""
Using the extracted passages below, produce a structured explanation of the book's insights related to '{TOPIC}'.

Extracted Passages:
{extracted}
"""
    summary = call_azure_gpt(gpt_prompt)

    print("\n========== FINAL OUTPUT ==========\n")
    print(summary)
    print("\n==================================\n")


if __name__ == "__main__":
    main()
