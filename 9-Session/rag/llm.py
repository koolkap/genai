import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_infer(prompt, model="llama3.2"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload)
    data = resp.json()
    return data.get("response", "")
