import streamlit as st
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"

st.set_page_config(page_title="DeepSeek Chatbot", layout="wide")
st.title("🤖 DeepSeek Chatbot")

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)

# User input
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state["messages"].append(("user", prompt))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True
        }

        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line.decode("utf-8"))
                    token = obj.get("response", "")
                    full_response += token
                    placeholder.write(full_response)

        st.session_state["messages"].append(("assistant", full_response))
