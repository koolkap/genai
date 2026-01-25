from langchain.prompts import PromptTemplate

SCHEMA_PROMPT = PromptTemplate(
    input_variables=["schema"],
    template="""
You are a data analyst.

Explain the table schema below.
Identify column roles and relationships.
Do NOT perform calculations.

Schema:
{schema}
"""
)

DESCRIPTIVE_PROMPT = PromptTemplate(
    input_variables=["schema", "sample"],
    template="""
You are analyzing tabular data.

Rules:
- Do NOT calculate exact values
- Do NOT sum, average, or count
- Describe trends, patterns, and anomalies only
- Use qualitative language

Schema understanding:
{schema}

Sample rows:
{sample}

Provide descriptive insights.
"""
)

ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "analysis"],
    template="""
Based on the analysis below, answer the user's question.

Analysis:
{analysis}

Question:
{question}
"""
)
