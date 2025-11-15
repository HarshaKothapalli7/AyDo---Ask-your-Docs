import streamlit as st
import uuid

def init_session_state():
    """
    Initialize all required Streamlit session state variables so that data
    persists across app reruns. Sets up chat history, a unique session ID,
    and default configuration values.
    """

    # Store chat messages across interactions
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a unique session ID for LangGraph checkpointing and conversation context
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        })

    # Track whether web search is enabled (default = True)
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True