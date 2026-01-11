print("###Sample Code to Learn OPENAI API with Python###")
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file
load_dotenv()

# Read key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ]
)

print(response.choices[0].message.content)
