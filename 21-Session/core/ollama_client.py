import requests
from config.settings import OLLAMA_URL, MODEL_NAME


def llm(prompt: str, temperature: float) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["response"].strip()
