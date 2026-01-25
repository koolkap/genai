from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory

from common.llm import get_llm
from common.prompt import get_prompt


def main():
    llm = get_llm()
    prompt = get_prompt()

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="history",
        return_messages=False
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    print(chain.invoke({"input": "My name is Alex and I love astronomy."})["text"])
    print(chain.invoke({"input": "I build homemade telescopes."})["text"])
    print(chain.invoke({"input": "What do you remember about me?"})["text"])


if __name__ == "__main__":
    main()
