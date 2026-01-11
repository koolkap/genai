import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="ChatGPT Chatbot", layout="wide")
st.title("🤖 ChatGPT - API Based Chatbot")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display old messages
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)

# Input box
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state["messages"].append(("user", prompt))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",      # or gpt-4.1 for stronger reasoning
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
            if chunk.choices and chunk.choices[0].delta.get("content"):
                token = chunk.choices[0].delta.content
                full_response += token
                placeholder.write(full_response)

    st.session_state["messages"].append(("assistant", full_response))
