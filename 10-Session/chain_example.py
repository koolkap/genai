from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2:3b")

prompt = PromptTemplate.from_template("""
Solve step-by-step:

{question}
""")

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "What is 36 / 3?"}))
