import streamlit as st
import streamlit.components.v1 as components
from backend_api import upload_document_to_backend, chat_with_backend_agent
from session_manager import init_session_state # Import to access session state

def display_header():
    """Renders the main title and introductory markdown."""
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

    st.title("🤖 AyDo - Ask Your Docs")
    st.markdown("Ask me anything! I can answer questions using my internal knowledge (RAG) or by searching the web.")
    st.markdown("---")

def render_document_upload_section(fastapi_base_url: str):
    """
    Renders the UI for uploading PDF documents to the knowledge base.
    Handles single and multiple file uploads with progress tracking.
    """
    st.header("Upload Document to Knowledge Base")
    with st.expander("Upload New Documents (PDF Only)"):
        uploaded_files = st.file_uploader(
            "Choose PDF file(s) - Multiple files supported",
            type="pdf",
            key="pdf_uploader",
            accept_multiple_files=True
        )

        if st.button("Upload PDF(s)", key="upload_pdf_button"):
            if uploaded_files is not None and len(uploaded_files) > 0:
                # Validate file count
                max_files = 10
                if len(uploaded_files) > max_files:
                    st.error(f"Too many files selected. Maximum allowed: {max_files} files.")
                    return

                # Display upload summary
                total_files = len(uploaded_files)
                st.info(f"📁 Uploading {total_files} file(s)...")

                # Progress tracking
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                # Track results
                successful_uploads = []
                failed_uploads = []

                # Upload each file
                for idx, uploaded_file in enumerate(uploaded_files):
                    current_progress = (idx) / total_files
                    progress_bar.progress(current_progress)
                    status_placeholder.text(f"Uploading {idx + 1}/{total_files}: {uploaded_file.name}...")

                    try:
                        upload_data = upload_document_to_backend(fastapi_base_url, uploaded_file)
                        successful_uploads.append({
                            'filename': upload_data.get('filename'),
                            'chunks': upload_data.get('processed_chunks', 0)
                        })
                    except Exception as e:
                        failed_uploads.append({
                            'filename': uploaded_file.name,
                            'error': str(e)
                        })

                # Complete progress
                progress_bar.progress(1.0)
                status_placeholder.empty()
                progress_bar.empty()

                # Display results summary
                st.markdown("---")
                st.subheader("📊 Upload Results")

                # Success summary
                if successful_uploads:
                    st.success(f"✅ Successfully uploaded: {len(successful_uploads)}/{total_files} file(s)")
                    with st.expander("View successful uploads", expanded=True):
                        for upload in successful_uploads:
                            st.markdown(f"✓ **{upload['filename']}** - {upload['chunks']} chunks processed")

                # Failure summary
                if failed_uploads:
                    st.error(f"❌ Failed: {len(failed_uploads)}/{total_files} file(s)")
                    with st.expander("View failed uploads", expanded=True):
                        for upload in failed_uploads:
                            st.markdown(f"✗ **{upload['filename']}**")
                            st.code(f"Error: {upload['error']}", language=None)

            else:
                st.warning("Please select at least one PDF file before clicking 'Upload PDF(s)'.")
    st.markdown("---")

def render_agent_settings_section():
    """
    Renders the section for agent settings, including the web search toggle.
    Updates the 'web_search_enabled' flag in session state.
    """
    st.header("Agent Settings")
    # Checkbox to enable/disable web search, linked to session state
    # The value is directly updated in st.session_state.web_search_enabled
    st.session_state.web_search_enabled = st.checkbox(
        "Enable Web Search (🌐)", 
        value=st.session_state.web_search_enabled,
        help="If enabled, the agent can use web search when its knowledge base is insufficient. If disabled, it will only use uploaded documents."
    )
    st.markdown("---")

def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display message content with markdown rendering
            st.markdown(message["content"], unsafe_allow_html=False)

def display_trace_events(trace_events: list):
    """
    Renders the detailed agent workflow trace in an expandable section.
    Uses icons, conditional styling, and better formatting for readability.
    """
    if trace_events:
        with st.expander("🔬 Agent Workflow Trace", expanded=False):
            # Add progress bar
            progress = st.progress(0)
            total_steps = len(trace_events)

            for idx, event in enumerate(trace_events):
                # Update progress
                progress.progress((idx + 1) / total_steps)

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

                    elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
                        st.info("🌐 Web search completed successfully")
                        with st.expander("View Web Search Results"):
                            st.code(event['details']['retrieved_content_summary'], language=None)

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

                    elif event['node_name'] == 'answer':
                        st.success("💡 Generating response based on gathered context...")

                    elif event['node_name'] == '__end__':
                        st.success("✅ Agent workflow completed successfully!")

                    # Show additional details if available
                    if event['details'] and event['node_name'] not in ['rag_lookup', 'web_search', 'router']:
                        with st.expander("📊 Raw Event Data"):
                            st.json(event['details'])

                    st.markdown("---")

            # Remove progress bar after completion
            progress.empty() 

