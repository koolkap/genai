from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

from common.llm import get_llm
from common.prompt import get_prompt


def main():
    llm = get_llm()
    prompt = get_prompt()

    memory = ConversationBufferWindowMemory(
        k=2,  # only last 2 exchanges
        memory_key="history",
        return_messages=False
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    print(chain.invoke({"input": "My name is Alex."})["text"])
    print(chain.invoke({"input": "I love astronomy."})["text"])
    print(chain.invoke({"input": "What is my name?"})["text"])


if __name__ == "__main__":
    main()
