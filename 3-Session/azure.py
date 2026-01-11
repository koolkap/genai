import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

st.set_page_config(page_title="Azure GPT Chatbot", layout="wide")
st.title("🤖 Azure OpenAI Chatbot")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state["messages"].append(("user", prompt))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                *[
                    {"role": role, "content": content}
                    for role, content in st.session_state["messages"]
                ],
                {"role": "user", "content": prompt},
            ],
            stream=True
        )

        for chunk in response:
            if not chunk.choices:
                continue  # skip empty events

            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_response += delta.content
                placeholder.write(full_response)

    st.session_state["messages"].append(("assistant", full_response))


