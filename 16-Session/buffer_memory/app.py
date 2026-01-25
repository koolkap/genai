from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from common.llm import get_llm
from common.prompt import get_prompt


def main():
    llm = get_llm()
    prompt = get_prompt()

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    print(chain.invoke({"input": "My name is Alex and I love astronomy."})["text"])
    print(chain.invoke({"input": "I also enjoy astrophotography."})["text"])
    print(chain.invoke({"input": "What hobbies did I mention?"})["text"])


if __name__ == "__main__":
    main()
