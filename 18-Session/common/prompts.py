from langchain_core.prompts import PromptTemplate


CODE_GEN_PROMPT = PromptTemplate(
    input_variables=["request"],
    template="""
You are a Python software engineer.

Generate Python 3.12 code.
Avoid dangerous functions.
Keep code minimal and readable.

Task:
{request}
"""
)

DEBUG_PROMPT = PromptTemplate(
    input_variables=["code", "error"],
    template="""
You are debugging Python code.

Given the code and the error traceback,
identify the bug and propose a corrected version.

Code:
{code}

Traceback:
{error}
"""
)

SECURITY_REVIEW_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
You are a security reviewer.

Analyze the code for:
- arbitrary code execution
- unsafe imports
- file system access
- shell injection
- insecure eval/exec usage

List risks and suggest safer alternatives.

Code:
{code}
"""
)
