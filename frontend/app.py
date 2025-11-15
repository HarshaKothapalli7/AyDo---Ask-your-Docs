import requests
import json
import streamlit as st
from config import FRONTEND_CONFIG
from session_manager import init_session_state
from ui_components import (
    display_header, 
    render_document_upload_section, 
    render_agent_settings_section, 
    display_chat_history, 
    display_trace_events
)
from backend_api import chat_with_backend_agent

def process_query(prompt, fastapi_base_url):
    """Process a query and return response with error handling."""
    try:
        # Call the backend API for chat
        agent_response, trace_events = chat_with_backend_agent(
            fastapi_base_url,
            st.session_state.session_id,
            prompt,
            st.session_state.web_search_enabled
        )
        return agent_response, trace_events, None
    except requests.exceptions.ConnectionError as e:
        error_msg = "❌ Could not connect to the FastAPI backend. Please ensure it's running on http://localhost:8000"
        return None, None, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Request failed: {str(e)}"
        return None, None, error_msg
    except json.JSONDecodeError:
        error_msg = "❌ Received an invalid response from the backend."
        return None, None, error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        return None, None, error_msg

def main():
    """Main function to run the Streamlit application."""

    # Initialize session state variables
    init_session_state()

    # Initialize error state if not exists
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

    # Get FastAPI base URL from config
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    # Render UI sections
    display_header()
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    st.header("💬 Chat with Your Documents")

    # Add clear chat button
    if st.button("🗑️ Clear Chat", help="Clear all messages"):
        st.session_state.messages = []
        st.session_state.last_error = None
        st.rerun()

    display_chat_history()

    # Show retry button if there was an error
    if st.session_state.last_error:
        error_query, error_msg = st.session_state.last_error
        if st.button("🔄 Retry Last Query", key="retry_error"):
            st.session_state.last_error = None
            # Retry the query
            with st.chat_message("assistant"):
                with st.spinner("🔄 Retrying..."):
                    agent_response, trace_events, error = process_query(error_query, fastapi_base_url)

                    if error:
                        st.error(error)
                        st.session_state.last_error = (error_query, error)
                        st.session_state.messages.append({"role": "assistant", "content": error})
                    else:
                        st.markdown(agent_response)
                        st.session_state.messages.append({"role": "assistant", "content": agent_response})
                        display_trace_events(trace_events)
                        st.session_state.last_error = None
            st.rerun()

    # User input field
    if prompt := st.chat_input("💭 Ask me anything..."):
        # Add user's message to chat history and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant's response and trace
        with st.chat_message("assistant"):
            # Custom animated spinner
            with st.spinner("🤔 Analyzing your question...\n🔍 Searching knowledge base...\n💡 Generating response..."):
                agent_response, trace_events, error = process_query(prompt, fastapi_base_url)

                if error:
                    st.error(error)
                    st.session_state.last_error = (prompt, error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
                else:
                    # Display the agent's final response
                    st.markdown(agent_response)
                    # Add the agent's response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})

                    # Display the workflow trace
                    display_trace_events(trace_events)
                    st.session_state.last_error = None

if __name__ == "__main__":
    main() 

