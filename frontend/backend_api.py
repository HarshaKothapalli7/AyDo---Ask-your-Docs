import requests
import json

def upload_document_to_backend(fastapi_base_url: str, uploaded_file):
    """
    Upload a PDF file to the FastAPI backend so it can be stored, parsed,
    and indexed for retrieval. This function sends the file as a POST request
    to the backend’s document-upload endpoint and returns the server’s response.
    """
    # Prepare the file for a multipart/form-data request
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }
    
    # Send the file to the backend upload endpoint
    response = requests.post(
        f"{fastapi_base_url}/upload-document/",
        files=files
    )
    response.raise_for_status()

    return response.json()

def chat_with_backend_agent(fastapi_base_url: str, session_id: str, query: str, enable_web_search: bool):
    """
    Send a user query to the FastAPI backend and retrieve the agent’s response.
    This function contacts the backend’s chat/agent endpoint, passes along the
    current session ID, the user’s message, and whether web search should be
    allowed. The backend processes the request (RAG, routing, reasoning, etc.)
    and returns the assistant’s reply along with any trace information.
    """
    payload = {
        "session_id": session_id,
        "query": query,
        "enable_web_search": enable_web_search
    }

    response = requests.post(
        f"{fastapi_base_url}/chat/",
        json=payload,
        stream=False
    )
    response.raise_for_status()

    data = response.json()

    agent_response = data.get(
        "response",
        "Sorry, I couldn't get a response from the agent."
    )
    trace_events = data.get("trace_events", [])

    return agent_response, trace_events