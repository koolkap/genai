"""
Structured Data Extraction with Enforced Schema using Pydantic

This script extracts order details from text using a local Ollama LLM
and guarantees valid output using a Pydantic schema.
"""

# Import BaseModel for schema definition
from pydantic import BaseModel

# Import Pydantic output parser (modern location)
from langchain_core.output_parsers import PydanticOutputParser

# Import PromptTemplate from langchain-core
from langchain_core.prompts import PromptTemplate

# Import Ollama LLM wrapper
from langchain_community.llms import Ollama


# Define expected structured output
class Order(BaseModel):
    customer_name: str
    product: str
    quantity: int
    order_date: str
    total_cost: float
    delivery_eta_days: int


# Create output parser that enforces the schema
parser = PydanticOutputParser(pydantic_object=Order)


# Prompt template with parser instructions injected
prompt = PromptTemplate(
    template="""
    Extract the order details from the text below.
    {format_instructions}

    Text:
    {text}
    """,
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# Initialize local LLM
llm = Ollama(model="llama3.2:latest")


# Build LCEL pipeline:
# Prompt → LLM → Pydantic Parser
chain = prompt | llm | parser


# Run the pipeline
order = chain.invoke({
    "text": """
    John Smith placed an order for 2 laptops on January 10, 2026.
    The total cost was $2400 and delivery is expected in 5 days.
    """
})


# Output is a validated Python object
print(order)
