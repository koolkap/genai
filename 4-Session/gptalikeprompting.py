import pymupdf
import requests
import json
from prompt import system_message, generate_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"

PDF_PATH = "The Almanack of Naval Ravikant PDF.pdf"
TOPIC = "Money"

# ---------------------------
# Load PDF
# ---------------------------
def extract_pdf(path):
    doc = pymupdf.open(path)
    book = ""
    for page in doc:
        book += page.get_text() + " "
    doc.close()
    return book

book_text = extract_pdf(PDF_PATH)

# ---------------------------
# Build prompts
# ---------------------------
system = system_message
prompt = generate_prompt(book_text, TOPIC)

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
]

# ---------------------------
# Helper function (like screenshot)
# ---------------------------
def get_summary():
    payload = {
        "model": MODEL,
        "prompt": build_chat_payload(messages),
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("response", "").strip()

# ---------------------------
# Helper: convert messages[] to a chat-formatted string
# (DeepSeek via Ollama does not accept OpenAI-style chat natively)
# ---------------------------
def build_chat_payload(messages):
    text = ""
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        text += f"{role}: {content}\n"
    return text


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    summary = get_summary()
    print(summary)
