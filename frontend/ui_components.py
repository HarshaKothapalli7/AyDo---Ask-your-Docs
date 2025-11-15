import streamlit as st
import streamlit.components.v1 as components
from backend_api import upload_document_to_backend, chat_with_backend_agent
from session_manager import init_session_state # Import to access session state

def display_header():
    """
    Render the page header, title, and introductory text for the AyDo application.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="AyDo - Ask Your Docs",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/aydo-rag',
            'Report a bug': 'https://github.com/yourusername/aydo-rag/issues',
            'About': '# AyDo - Ask Your Docs\nA powerful RAG system for document Q&A'
        }
    )

    # Load custom CSS
    try:
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Silently fail if CSS file doesn't exist

    # Application title
    st.title("🤖 AyDo - Ask Your Docs")

    # Introductory description
    st.markdown("Ask me anything! I can answer questions using my internal knowledge (RAG) or by searching the web.")

    # Horizontal separator
    st.markdown("---")


def render_document_upload_section(fastapi_base_url: str):
    """
    User interface for uploading PDF documents into the knowledge base.
    Includes file selection, upload action, and backend API integration.
    """

    # Section header for document upload
    st.header("Upload Document to Knowledge Base")

    # Collapsible section for uploading a new document
    with st.expander("Upload New Document (PDF Only)"):

        # File uploader widget (PDF restricted)
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            key="pdf_uploader"
        )

        # Button to trigger the upload request
        if st.button("Upload PDF", key="upload_pdf_button"):

            # Proceed only if a file is selected
            if uploaded_file is not None:
                # Display spinner while processing
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        # Send the file to the FastAPI backend for processing
                        upload_data = upload_document_to_backend(
                            fastapi_base_url,
                            uploaded_file
                        )

                        # Display success message with backend response details
                        st.success(
                            f"PDF '{upload_data.get('filename')}' uploaded successfully! "
                            f"Processed {upload_data.get('processed_chunks')} pages."
                        )

                    except Exception as e:
                        # Handle any errors from backend or network
                        st.error(f"An error occurred during upload: {e}")

            else:
                # Warn the user if they clicked upload with no file selected
                st.warning("Please upload a PDF file before clicking 'Upload PDF'.")

    # Divider between sections
    st.markdown("---")


def render_agent_settings_section():
    """
    Display configuration options for the agent, including the web search toggle.
    Updates the `web_search_enabled` flag in Streamlit session state.
    """

    st.header("Agent Settings")

    # Checkbox controlling the web search setting
    st.session_state.web_search_enabled = st.checkbox(
        "Enable Web Search (🌐)",
        value=st.session_state.web_search_enabled,
        help="If enabled, the agent can use web search when its knowledge base is insufficient. If disabled, it will only use uploaded documents."
    )

    st.markdown("---")


def display_chat_history():
    """
    Render the chat history stored in Streamlit session state.

    This function loops through all previously exchanged messages and displays
    each one using Streamlit's chat_message container, preserving role and formatting.
    """

    # Iterate through all stored messages (list of dicts with 'role' and 'content')
    for idx, message in enumerate(st.session_state.messages):
        # Create a chat bubble based on message role: "user" or "assistant"
        with st.chat_message(message["role"]):
            # Display the message content using markdown for formatting support
            st.markdown(message["content"], unsafe_allow_html=False)

def display_trace_events(trace_events: list):
    """
    Display the agent's internal reasoning trace in a structured, expandable panel.
    Uses icons, conditional styling, and better formatting for readability.
    Each trace event is rendered with descriptive labels and additional details when available.
    """

    if not trace_events:
        return

    # Expandable section to view agent workflow details with enhanced UI
    with st.expander("🔬 Agent Workflow Trace", expanded=False):
        # Add progress bar
        progress = st.progress(0)
        total_steps = len(trace_events)

        for idx, event in enumerate(trace_events):
            # Update progress
            progress.progress((idx + 1) / total_steps)

            # Mapping internal node names to emojis for better visualization
            icon_map = {
                'router': "🧭",
                'rag_lookup': "📚",
                'web_search': "🌐",
                'answer': "💡",
                '__end__': "✅"
            }
            icon = icon_map.get(event['node_name'], "⚙️")

            # Use containers for better organization
            with st.container():
                st.markdown(f"### {icon} Step {event['step']}: `{event['node_name']}`")
                st.markdown(f"**{event['description']}**")

                # RAG Lookup node - show verdict and sources
                if event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in event['details']:
                    verdict = event['details']['sufficiency_verdict']
                    source_docs = event['details'].get('source_documents', [])

                    if verdict == "Sufficient":
                        st.success(f"✓ **RAG Verdict:** {verdict} - Relevant info found in Knowledge Base.")
                        if source_docs:
                            st.info(f"📄 **Sources:** {', '.join(source_docs)}")
                    else:
                        st.warning(f"✗ **RAG Verdict:** {verdict} - Insufficient info in Knowledge Base.")

                    if 'retrieved_content_summary' in event['details']:
                        with st.expander("View Retrieved Content"):
                            st.code(event['details']['retrieved_content_summary'], language=None)

                # Web Search node
                elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
                    st.info("🌐 Web search completed successfully")
                    with st.expander("View Web Search Results"):
                        st.code(event['details']['retrieved_content_summary'], language=None)

                # Router node - show decision and any overrides
                elif event['node_name'] == 'router':
                    decision = event['details'].get('decision', 'Unknown')
                    st.info(f"🎯 **Decision:** Route to `{decision}`")

                    if 'router_override_reason' in event['details']:
                        st.warning(f"⚠️ **Override:** {event['details']['router_override_reason']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Initial Decision", event['details']['initial_decision'])
                        with col2:
                            st.metric("Final Decision", event['details']['final_decision'])

                # Answer node
                elif event['node_name'] == 'answer':
                    st.success("💡 Generating response based on gathered context...")

                # End node
                elif event['node_name'] == '__end__':
                    st.success("✅ Agent workflow completed successfully!")

                # Show additional details if available for other nodes
                if event['details'] and event['node_name'] not in ['rag_lookup', 'web_search', 'router']:
                    with st.expander("📊 Raw Event Data"):
                        st.json(event['details'])

                st.markdown("---")

        # Remove progress bar after completion
        progress.empty()
