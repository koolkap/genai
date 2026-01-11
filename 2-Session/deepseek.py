import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:1.5b",
    "prompt": "Explain quantum computing in simple words.",
    "stream": False # Set to True for streaming responses (thinking mode)
}

####JSON STREAMING EXAMPLE####

# with requests.post(url, json=payload, stream=True) as r:
#     for line in r.iter_lines():
#         if line:
#             data = line.decode('utf-8')
#             print(data, end="")

###TEXT RESPONSE EXAMPLE###

response = requests.post(
    url=url,
    json=payload
)

print(response.json()["response"])


