import streamlit as st
# Import functions that send requests to the FastAPI backend
from backend_api import upload_document_to_backend

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
    Includes file selection, upload action, and backend API integration.
    """

    # Section header for document upload
    st.header("Upload Your Document (PDF only)")

    # Collapsible section for uploading a new document
    with st.expander("Add a New Document"):

        # File uploader widget (PDF restricted)
        uploaded_file = st.file_uploader(
            "Select a PDF file to add to the knowledge base",
            type="pdf",
            key="pdf_uploader"
        )

        # Button to trigger the upload request
        if st.button("Upload PDF", key="upload_pdf_button"):

            # Proceed only if a file is selected
            if uploaded_file is not None:
                # Display spinner while processing
                with st.spinner(f"Uploading and processing '{uploaded_file.name}'..."):
                    try:
                        # Send the file to the FastAPI backend for processing
                        upload_data = upload_document_to_backend(
                            fastapi_base_url,
                            uploaded_file
                        )

                        # Display success message with backend response details
                        st.success(
                            f"Upload complete: '{upload_data.get('filename')}'. "
                        )

                    except Exception as e:
                        # Handle any errors from backend or network
                        st.error(f"Upload failed: {e}")

            else:
                # Warn the user if they clicked upload with no file selected
                st.warning("Please select a PDF file before clicking 'Upload PDF'.")
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
    for message in st.session_state.messages:
        # Create a chat bubble based on message role: "user" or "assistant"
        with st.chat_message(message["role"]):
            # Display the message content using markdown for formatting support
            st.markdown(message["content"])

def display_trace_events(trace_events: list):
    """
    Display the agent’s internal reasoning trace in a structured, expandable panel.
    Each trace event is rendered with descriptive labels and additional details when available.
    """

    if not trace_events:
        return

    # Expandable section to view agent workflow details
    with st.expander("Agent Workflow Trace"):

        # Mapping internal node names to human-friendly labels
        label_map = {
            "router": "Router",
            "rag_lookup": "RAG Lookup",
            "web_search": "Web Search",
            "answer": "Answer",
            "__end__": "End"
        }

        for event in trace_events:
            node_name = event.get("node_name")
            step_num = event.get("step")

            # Fallback: if node not in map, title-case the raw name
            label = label_map.get(node_name, (node_name or "Step").replace("_", " ").title())

            # Clean step header (no duplicate "router", no brackets)
            st.subheader(f"Step {step_num} • {label} ")

            # High-level description
            description = event.get("description", "No description provided.")
            st.write(f"**Description:** {description}")

            details = event.get("details", {})

            # RAG Lookup node
            if node_name == "rag_lookup" and "sufficiency_verdict" in details:
                verdict = details["sufficiency_verdict"]

                if verdict == "Sufficient":
                    st.success(f"RAG Verdict: {verdict} — Relevant knowledge base content identified.")
                else:
                    st.warning(f"RAG Verdict: {verdict} — Not enough information found; may fall back to Web Search.")

                if "retrieved_content_summary" in details:
                    st.markdown(f"**Retrieved Content Summary:** `{details['retrieved_content_summary']}`")

            # Web Search node
            elif node_name == "web_search" and "retrieved_content_summary" in details:
                st.markdown(f"**Web Search Content Summary:** `{details['retrieved_content_summary']}`")

            # Router node
            elif node_name == "router" and "router_override_reason" in details:
                st.info(f"Router Override: {details['router_override_reason']}")
                st.json({
                    "initial_decision": details.get("initial_decision"),
                    "final_decision": details.get("final_decision")
                })

            # Other nodes with details
            elif details:
                st.json(details)

            st.markdown("---")