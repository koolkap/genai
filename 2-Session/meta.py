import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b-instruct-q4_0",
        "prompt": "Explain machine learning in one paragraph.",
        "stream": False  # Set to True for streaming responses (thinking mode)
    }
)

print(response.json()["response"])
