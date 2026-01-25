from typing import TypedDict

class CodeState(TypedDict):
    request: str
    code: str
    execution_result: str
    error: str
    security_review: str
