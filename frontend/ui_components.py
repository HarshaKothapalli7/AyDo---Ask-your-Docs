import streamlit as st
# Import functions that send requests to the FastAPI backend
from backend_api import upload_document_to_backend, upload_documents_batch_to_backend

def display_header():
    """
    Render the page header, title, and introductory text for the AyDo application.
    """
    # Configure the Streamlit page
    st.set_page_config(page_title="AyDo", layout="wide")

    # Application title
    st.title("AyDo - Ask your Docs")

    # Introductory description
    st.markdown(
        "Interact with AyDo to ask questions about your documents. "
        "AyDo utilizes Retrieval-Augmented Generation (RAG) and optional web search "
        "to provide accurate, context-aware responses."
    )

    # Horizontal separator
    st.markdown("---")


def render_document_upload_section(fastapi_base_url: str):
    """
    User interface for uploading PDF documents into the knowledge base.
    Supports single and multiple file uploads with batch processing.
    """

    # Section header for document upload
    st.header("Upload Your Document (PDF only)")

    # Collapsible section for uploading documents
    with st.expander("Add a New Document"):

        # File uploader widget (PDF restricted, multiple files supported)
        uploaded_files = st.file_uploader(
            "Select PDF file(s) to add to the knowledge base - Multiple files supported",
            type="pdf",
            key="pdf_uploader",
            accept_multiple_files=True
        )

        # Button to trigger the upload request
        if st.button("Upload PDF", key="upload_pdf_button"):

            # Proceed only if files are selected
            if uploaded_files is not None and len(uploaded_files) > 0:
                # Validate file count
                max_files = 10
                if len(uploaded_files) > max_files:
                    st.error(f"Too many files selected. Maximum allowed: {max_files} files.")
                else:
                    total_files = len(uploaded_files)

                    # Track results
                    successful_uploads = []
                    failed_uploads = []

                    # Display spinner while processing
                    with st.spinner(f"Uploading and processing {total_files} file(s)..."):
                        try:
                            # Use batch upload for all files (even single file)
                            batch_response = upload_documents_batch_to_backend(fastapi_base_url, uploaded_files)

                            # Process results from batch response
                            for result in batch_response.get('results', []):
                                if result['status'] == 'success':
                                    successful_uploads.append({
                                        'filename': result['filename'],
                                        'chunks': result['processed_chunks']
                                    })
                                else:
                                    failed_uploads.append({
                                        'filename': result['filename'],
                                        'error': result['error_message']
                                    })
                        except Exception as e:
                            # If batch upload fails entirely, mark all as failed
                            for uploaded_file in uploaded_files:
                                failed_uploads.append({
                                    'filename': uploaded_file.name,
                                    'error': str(e)
                                })

                    # Display results summary
                    if successful_uploads:
                        st.success(f"Upload complete: {len(successful_uploads)}/{total_files} file(s) processed successfully.")
                        with st.expander("View upload details", expanded=False):
                            for upload in successful_uploads:
                                st.markdown(f"✓ **{upload['filename']}** - {upload['chunks']} chunks")

                    # Display failures
                    if failed_uploads:
                        st.error(f"Upload failed: {len(failed_uploads)}/{total_files} file(s) failed.")
                        with st.expander("View error details", expanded=True):
                            for upload in failed_uploads:
                                st.markdown(f"✗ **{upload['filename']}**")
                                st.code(f"Error: {upload['error']}", language=None)

            else:
                # Warn the user if they clicked upload with no file selected
                st.warning("Please select a PDF file before clicking 'Upload PDF'.")

    # Custom CSS for button styling (Lahari's design)
    st.markdown("""
    <style>
        .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        }
        .stButton>button:hover {
        background-color: #4338CA;
        }
        </style>""", unsafe_allow_html=True)

    # Divider between sections
    st.markdown("---")


def render_agent_settings_section():
    """
    Display configuration options for the agent, including the web search toggle.
    Updates the `web_search_enabled` flag in Streamlit session state.
    """

    st.header("Agent Settings")

    # Callback function executed when the checkbox state changes
    def on_toggle():
        # You can update logs, show notifications, or process changes here
        st.toast("Web search setting updated.")

    # Checkbox controlling the web search setting
    st.session_state.web_search_enabled = st.checkbox(
        label="Enable Web Search",
        key="web_search_toggle",
        help=(
        "When enabled, the agent can use external web search to supplement its knowledge.\n"
        "When disabled, the agent relies solely on the uploaded documents and the internal knowledge base."
        ),
        on_change=on_toggle,
        disabled=False,
        label_visibility="visible"
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
    with st.expander(" Agent Workflow Trace", expanded=False):
        # Add progress bar
        progress = st.progress(0)
        total_steps = len(trace_events)

        for idx, event in enumerate(trace_events):
            # Update progress
            progress.progress((idx + 1) / total_steps)

            # Mapping internal node names to emojis for better visualization
            # icon_map = {
            #     'router': "🧭",
            #     'rag_lookup': "📚",
            #     'web_search': "🌐",
            #     'answer': "💡",
            #     '__end__': "✅"
            # }
            # icon = icon_map.get(event['node_name'], "⚙️")

            # Use containers for better organization
            with st.container():
                st.markdown(f"###  Step {event['step']}: `{event['node_name']}`")
                st.markdown(f"**{event['description']}**")

                # RAG Lookup node - show verdict and sources
                if event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in event['details']:
                    verdict = event['details']['sufficiency_verdict']
                    source_docs = event['details'].get('source_documents', [])

                    if verdict == "Sufficient":
                        st.success(f"✓ **RAG Verdict:** {verdict} - Relevant info found in Knowledge Base.")
                        if source_docs:
                            st.info(f" **Sources:** {', '.join(source_docs)}")
                    else:
                        st.warning(f"✗ **RAG Verdict:** {verdict} - Insufficient info in Knowledge Base.")

                    if 'retrieved_content_summary' in event['details']:
                        with st.expander("View Retrieved Content"):
                            st.code(event['details']['retrieved_content_summary'], language=None)

                # Web Search node
                elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
                    st.info(" Web search completed successfully")
                    with st.expander("View Web Search Results"):
                        st.code(event['details']['retrieved_content_summary'], language=None)

                # Router node - show decision and any overrides
                elif event['node_name'] == 'router':
                    decision = event['details'].get('decision', 'Unknown')
                    st.info(f" **Decision:** Route to `{decision}`")

                    if 'router_override_reason' in event['details']:
                        st.warning(f" **Override:** {event['details']['router_override_reason']}")
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
