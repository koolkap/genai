import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_URL")
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Explain solar energy in simple words."}
    ]
)

print(response.choices[0].message.content)
