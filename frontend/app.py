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
    """Main entry point for running the Streamlit application."""

    # Initialize all required session-level variables
    init_session_state()

    # Initialize error state if not exists
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

    # Retrieve FastAPI backend URL from configuration
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    # Path to custom assistant avatar (favicon)
    # Using absolute path to ensure it works regardless of where the app is run from
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assistant_avatar = os.path.join(base_dir, "images", "favicon_io", "favicon-32x32.png")

    # Render the primary UI components
    display_header()
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    st.header("Shoot you Questions!")

    # Add clear chat button
    if st.button("🧹 Clear Chat", help="Clear all messages"):
        st.session_state.messages = []
        st.session_state.last_error = None
        st.rerun()

    display_chat_history(assistant_avatar=assistant_avatar)

    # Show retry button if there was an error
    if st.session_state.last_error:
        error_query, error_msg = st.session_state.last_error
        if st.button("🔄 Retry Last Query", key="retry_error"):
            st.session_state.last_error = None
            # Retry the query
            with st.chat_message("assistant", avatar=assistant_avatar):
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

    # Chat input field
    if prompt := st.chat_input("💭 Ask me anything..."):
        # Store and display the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process backend response inside an assistant message block
        with st.chat_message("assistant", avatar=assistant_avatar):
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

                    # Render workflow trace (router decisions, RAG steps, etc.)
                    display_trace_events(trace_events)
                    st.session_state.last_error = None

if __name__ == "__main__":
    main()
