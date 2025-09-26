import streamlit as st
from datetime import datetime
import requests
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Health Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Health Agent 🤖")
st.caption(f"📍 Operating for Warangal, Telangana | {datetime.now().strftime('%B %d, %Y')}")

# --- Session State Initialization ---
# This is to remember the chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your health today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask about health or describe your symptoms..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Call FastAPI Backend ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # The URL of your FastAPI backend
            api_url = "http://127.0.0.1:8000/chat"
            
            # The data to send to the API
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }

            # Make the API request
            response = requests.post(api_url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Get the agent's response from the JSON
            agent_response = response.json()["response"]
            message_placeholder.markdown(agent_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Could not connect to the AI backend. Please make sure the FastAPI server is running. Error: {e}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})