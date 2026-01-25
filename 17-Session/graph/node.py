from common.llm import get_llm
from common.data import load_dataframe
from common.prompts import (
    SCHEMA_PROMPT,
    DESCRIPTIVE_PROMPT,
    ANSWER_PROMPT,
)
from graph.state import TableState


df = load_dataframe()


def schema_node(state: TableState):
    llm = get_llm()

    schema_text = llm.invoke(
        SCHEMA_PROMPT.format(
            schema=str(df.dtypes)
        )
    )

    return {
        **state,
        "schema_description": schema_text,
    }


def descriptive_node(state: TableState):
    llm = get_llm()

    analysis = llm.invoke(
        DESCRIPTIVE_PROMPT.format(
            schema=state["schema_description"],
            sample=df.head(4).to_string(index=False)
        )
    )

    return {
        **state,
        "analysis": analysis,
    }


def answer_node(state: TableState):
    llm = get_llm()

    answer = llm.invoke(
        ANSWER_PROMPT.format(
            question=state["question"],
            analysis=state["analysis"]
        )
    )

    return {
        **state,
        "answer": answer,
    }
