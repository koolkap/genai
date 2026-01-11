"""
===========================================
DeepSeek Chatbot (Local via Ollama + Streamlit)
-------------------------------------------
This script creates a simple chatbot using:

✓ DeepSeek model running locally via Ollama
✓ Streamlit for web UI
✓ Streaming token output for ChatGPT-like typing
✓ Session state for multi-turn conversation

Requirements:
- Install Ollama: https://ollama.com/download
- Pull model:  ollama pull deepseek-r1:1.5b

Run with:
    streamlit run deepseek_chatbot.py
===========================================
"""

import streamlit as st
import requests
import json


# -------------------------------------------
# 1. Ollama API Config
# -------------------------------------------
# This tells our app where to send prompts and which model to use.
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"


# -------------------------------------------
# 2. Streamlit Page Setup
# -------------------------------------------
# Defines title, layout, and basic UI structure.
st.set_page_config(page_title="DeepSeek Chatbot", layout="wide")
st.title("🤖 DeepSeek Chatbot")


# -------------------------------------------
# 3. Initialize Chat Memory
# -------------------------------------------
# Streamlit reruns when inputs change, so we use session_state to keep chat history.
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# -------------------------------------------
# 4. Display Chat History from Memory
# -------------------------------------------
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)


# -------------------------------------------
# 5. Capture User Input (Chat Style)
# -------------------------------------------
prompt = st.chat_input("Ask something...")


# -------------------------------------------
# 6. If User Enters a Prompt, Process It
# -------------------------------------------
if prompt:

    # Save user's message to chat history
    st.session_state["messages"].append(("user", prompt))

    # Render assistant bubble for streaming output
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Payload for Ollama: we request streaming tokens
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True
        }

        # -------------------------------------------
        # 7. Send Request to Ollama (Streaming Mode)
        # -------------------------------------------
        # Ollama streams JSON lines like:
        #   {"response": "Hello "}
        #   {"response": "world!"}
        #
        # We accumulate them and update UI live.
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line.decode("utf-8"))
                    token = obj.get("response", "")
                    full_response += token
                    placeholder.write(full_response)

        # Save assistant's final output into memory
        st.session_state["messages"].append(("assistant", full_response))
