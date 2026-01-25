from common.llm import get_llm
from common.prompts import (
    CODE_GEN_PROMPT,
    DEBUG_PROMPT,
    SECURITY_REVIEW_PROMPT
)
from common.sandbox import safe_execute
from graph.state import CodeState


def code_generation_node(state: CodeState):
    llm = get_llm()

    code = llm.invoke(
        CODE_GEN_PROMPT.format(
            request=state["request"]
        )
    )

    return {**state, "code": code}


def execution_node(state: CodeState):
    result = safe_execute(state["code"])

    if result["success"]:
        return {
            **state,
            "execution_result": "Execution succeeded",
            "error": ""
        }

    return {
        **state,
        "execution_result": "Execution failed",
        "error": result["error"]
    }


def debugging_node(state: CodeState):
    if not state["error"]:
        return state

    llm = get_llm()

    fixed_code = llm.invoke(
        DEBUG_PROMPT.format(
            code=state["code"],
            error=state["error"]
        )
    )

    return {**state, "code": fixed_code}


def security_review_node(state: CodeState):
    llm = get_llm()

    review = llm.invoke(
        SECURITY_REVIEW_PROMPT.format(
            code=state["code"]
        )
    )

    return {**state, "security_review": review}
