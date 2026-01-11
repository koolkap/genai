from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Can you tell 2+2?"}],
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

print(response.choices[0].message.content)

