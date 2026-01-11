"""
==============================================
Azure GPT Chatbot Using Streamlit + Azure OpenAI
----------------------------------------------
This script creates a chatbot using:

✓ Azure OpenAI GPT deployment
✓ Streamlit for chat UI
✓ Streaming response (token-by-token)
✓ Multi-turn conversation using session_state

Requirements:
- pip install streamlit openai python-dotenv
- Azure OpenAI resource deployed with a model

Run with:
    streamlit run azure_chatbot.py
==============================================
"""

import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI


# ----------------------------------------------
# 1. Load environment variables from .env file
# ----------------------------------------------
load_dotenv()


# ----------------------------------------------
# 2. Setup Azure OpenAI client
# ----------------------------------------------
# These environment variables should be defined in your .env file:
#   AZURE_OPENAI_KEY
#   API_VERSION
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_DEPLOYMENT
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# ----------------------------------------------
# 3. Streamlit Page Configuration
# ----------------------------------------------
st.set_page_config(page_title="Azure GPT Chatbot", layout="wide")
st.title("🤖 Azure OpenAI Chatbot")


# ----------------------------------------------
# 4. Initialize Chat Memory
# ----------------------------------------------
# Streamlit reruns code every interaction, so session_state
# keeps the history of previous messages.
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ----------------------------------------------
# 5. Render Chat History on Screen
# ----------------------------------------------
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)


# ----------------------------------------------
# 6. Capture User Input from Chat Box
# ----------------------------------------------
prompt = st.chat_input("Ask something...")


# ----------------------------------------------
# 7. Process User Prompt and Stream Answer
# ----------------------------------------------
if prompt:

    # Save user's message into memory
    st.session_state["messages"].append(("user", prompt))

    # Create assistant message bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Send conversation context + new message to Azure
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
            stream=True  # enable token-by-token streaming
        )

        # ----------------------------------------------
        # 8. Consume Streaming Tokens from Azure
        # ----------------------------------------------
        # Azure streams chunk objects like:
        #   chunk.choices[0].delta.content
        # Some events contain no tokens, so we skip empty chunks.
        for chunk in response:
            if not chunk.choices:
                continue  # skip empty stream events

            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_response += delta.content
                placeholder.write(full_response)

    # Save assistant response to memory
    st.session_state["messages"].append(("assistant", full_response))
