import os
import requests
from langchain.agents import initialize_agent, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# =======================
# CONFIGURATION
# =======================

# Set your API Key (from environment variable or hardcoded for testing)
OPENWEATHER_API_KEY = "959b6a836f775fe65ec9263c8f6bf404"##os.getenv("OPENWEATHER_API_KEY", "REPLACE_ME")

# =======================
# TOOL IMPLEMENTATION
# =======================

def get_weather(city: str):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return f"Error fetching weather: {r.text}"

    data = r.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    main = data["weather"][0]["main"]
    raining = main.lower() in ["rain", "drizzle", "thunderstorm"]

    return f"{city}: {temp}°C, {desc}, raining={raining}"

tools = [
    Tool(
        name="WeatherTool",
        func=get_weather,
        description="Get current weather for a city. Input must be a city name."
    )
]

# =======================
# LLM CONFIG
# =======================

llm = Ollama(model="llama3.2:3b", temperature=0)

# =======================
# REACT PROMPT
# =======================

template = """
Answer the question using the ReAct format.

Format:
Thought: reasoning
Action: tool name (only if needed)
Action Input: what to send to the tool
Observation: tool result
... repeat Thought/Action/Observation as needed ...
Final Answer: final answer to the user (no more tool calls)

Rules:
- After getting weather, decide if user needs an umbrella.
- If raining=True → umbrella recommended.
- If raining=False → umbrella not required.
- Do NOT repeat the question.
- Do NOT loop forever.
- Always finish with Final Answer.

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
    verbose=True,
    max_iterations=3  # safety
)

# =======================
# RUN A QUERY
# =======================

result = agent_executor.run("What's the weather in Seoul and should I carry an umbrella?")
print("\n=== RESULT ===")
print(result)
