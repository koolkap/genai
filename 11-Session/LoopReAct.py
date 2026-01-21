from langchain.agents import initialize_agent, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import requests
import os

# --- CONFIG ---
OPENWEATHER_API_KEY = "959b6a836f775fe65ec9263c8f6bf404" ##os.environ.get("OPENWEATHER_API_KEY")  # or paste directly

# --- TOOL IMPLEMENTATION ---
def get_weather(city: str):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return f"Error: {r.text}"
    data = r.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"{city}: {temp}°C, {desc}"

tools = [
    Tool(
        name="WeatherTool",
        func=get_weather,
        description="Get current weather for a city. Input should be a city name."
    )
]

llm = Ollama(model="llama3.2:3b", temperature=0)

template = """
Use this format:

Thought:
Action:
Action Input:
Observation:
Final Answer:

Available tools:
{tools}

Question: {input}
"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

result = agent_executor.run("What's the weather in Seoul and should I carry an umbrella?")
print(result)
