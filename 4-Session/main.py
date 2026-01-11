import pymupdf
import requests
import json
from prompt import system_message, generate_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"

PDF_PATH = "The Almanack of Naval Ravikant PDF.pdf"
TOPIC = "Money"   # you can change this "Embrace Your Unique Skills"


def extract_text_from_pdf(path: str) -> str:
    """Extracts plain text from a PDF file using PyMuPDF."""
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def call_deepseek(prompt: str) -> str:
    """Sends prompt to DeepSeek via Ollama and returns final output."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False  # final output only, no chain-of-thought
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response_data = response.json()
    return response_data.get("response", "").strip()


def main():
    print("📘 Loading PDF...")
    book_text = extract_text_from_pdf(PDF_PATH)
    print(f"📄 Extracted {len(book_text)} characters from the PDF.")  #this you can use to calculate the token required for each computation
    print("🧱 Building prompt...")
    final_prompt = generate_prompt(book_text, TOPIC)

    # Combine system + task prompt
    full_prompt = system_message + "\n\n" + final_prompt

    print("🤖 Sending to DeepSeek model...")
    result = call_deepseek(full_prompt)

    print("\n================ RESULT ================\n")
    print(result)
    print("\n========================================\n")


if __name__ == "__main__":
    main()
