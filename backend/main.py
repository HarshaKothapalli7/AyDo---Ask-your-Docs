
import os
import time
import logging
from typing import List, Dict, Any
import tempfile

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


from agent import rag_agent
from vectorstore import add_document_to_vectorstore, clear_all_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for the LangGraph-powered RAG agent with Pinecone and Groq.",
    version="1.0.0",
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session manager for LangGraph checkpoints (for demonstration)
memory = MemorySaver()

# --- Pydantic Models for API ---
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    enable_web_search: bool = True # NEW: Add web search toggle state

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int

class FileUploadResult(BaseModel):
    filename: str
    status: str  # "success" or "failed"
    processed_chunks: int = 0
    error_message: str = ""
    document_id: str = ""

class BatchUploadResponse(BaseModel):
    total_files: int
    successful_uploads: int
    failed_uploads: int
    results: List[FileUploadResult]

# --- Configuration Constants ---
MAX_FILE_SIZE_MB = 200  # Maximum file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Document Upload Endpoint ---
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")  # Limit to 5 uploads per minute
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Uploads a PDF document, extracts text, and adds it to the RAG knowledge base.
    """
    # Validate file extension
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    # Validate content type
    if file.content_type not in ["application/pdf", "application/x-pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Expected PDF but got {file.content_type}."
        )

    # Read file content and validate size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)."
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Write validated content to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    logger.info(f"Received PDF for upload: {file.filename} ({file_size / 1024 / 1024:.2f} MB). Saved to {temp_file_path}")

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        total_chunks_added = 0
        document_id = None
        if documents:
            full_text_content = "\n\n".join([doc.page_content for doc in documents])
            # Pass filename to add_document_to_vectorstore for metadata tracking
            document_id = add_document_to_vectorstore(full_text_content, filename=file.filename)
            total_chunks_added = len(documents)
            logger.info(f"Successfully indexed {total_chunks_added} chunks from {file.filename} with document_id: {document_id}")
        else:
            logger.warning(f"No content extracted from {file.filename}")

        return DocumentUploadResponse(
            message=f"PDF '{file.filename}' successfully uploaded and indexed. Document ID: {document_id}",
            filename=file.filename,
            processed_chunks=total_chunks_added
        )
    except Exception as e:
        logger.error(f"Error processing PDF document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process PDF. Please ensure the file is a valid PDF document."
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Cleaned up temporary file: {temp_file_path}")


# --- Batch Document Upload Endpoint ---
@app.post("/upload-documents-batch/", response_model=BatchUploadResponse, status_code=status.HTTP_200_OK)
@limiter.limit("3/minute")  # More restrictive for batch uploads
async def upload_documents_batch(request: Request, files: List[UploadFile] = File(...)):
    """
    Uploads multiple PDF documents in a single request.
    Processes each file independently and returns individual results.
    All files in the batch share the same batch_id for grouped retrieval.
    """
    MAX_BATCH_SIZE = 10

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum allowed: {MAX_BATCH_SIZE} files per batch."
        )

    # Generate a single batch_id for all files in this upload session
    from datetime import datetime
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
    logger.info(f"Batch upload started: {len(files)} files with batch_id: {batch_id}")

    results = []
    successful_count = 0
    failed_count = 0

    for file in files:
        file_result = FileUploadResult(filename=file.filename, status="failed")

        try:
            # Validate file extension
            if not file.filename.endswith(".pdf"):
                file_result.error_message = "Only PDF files are supported."
                file_result.status = "failed"
                failed_count += 1
                results.append(file_result)
                continue

            # Validate content type
            if file.content_type not in ["application/pdf", "application/x-pdf"]:
                file_result.error_message = f"Invalid content type: {file.content_type}"
                file_result.status = "failed"
                failed_count += 1
                results.append(file_result)
                continue

            # Read and validate file size
            file_content = await file.read()
            file_size = len(file_content)

            if file_size > MAX_FILE_SIZE_BYTES:
                file_result.error_message = f"File too large ({file_size / 1024 / 1024:.2f} MB). Max: {MAX_FILE_SIZE_MB} MB"
                file_result.status = "failed"
                failed_count += 1
                results.append(file_result)
                continue

            if file_size == 0:
                file_result.error_message = "File is empty."
                file_result.status = "failed"
                failed_count += 1
                results.append(file_result)
                continue

            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                temp_file_path = tmp_file.name

            try:
                # Process PDF
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()

                if documents:
                    full_text_content = "\n\n".join([doc.page_content for doc in documents])
                    document_id = add_document_to_vectorstore(full_text_content, filename=file.filename, batch_id=batch_id)
                    total_chunks = len(documents)

                    file_result.status = "success"
                    file_result.processed_chunks = total_chunks
                    file_result.document_id = document_id
                    successful_count += 1
                    logger.info(f"Successfully processed {file.filename}: {total_chunks} chunks, ID: {document_id}, Batch ID: {batch_id}")
                else:
                    file_result.error_message = "No content extracted from PDF."
                    file_result.status = "failed"
                    failed_count += 1
                    logger.warning(f"No content extracted from {file.filename}")

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            file_result.error_message = f"Processing error: {str(e)}"
            file_result.status = "failed"
            failed_count += 1
            logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)

        results.append(file_result)

    logger.info(f"Batch upload completed: {successful_count} successful, {failed_count} failed")

    return BatchUploadResponse(
        total_files=len(files),
        successful_uploads=successful_count,
        failed_uploads=failed_count,
        results=results
    )


# --- Chat Endpoint ---
@app.post("/chat/", response_model=AgentResponse)
@limiter.limit("20/minute")  # Limit to 20 chat requests per minute
async def chat_with_agent(request: Request, query_request: QueryRequest):
    trace_events_for_frontend: List[TraceEvent] = []
    
    try:
        # Pass enable_web_search into the config for the agent to access
        config = {
            "configurable": {
                "thread_id": query_request.session_id,
                "web_search_enabled": query_request.enable_web_search
            }
        }
        inputs = {"messages": [HumanMessage(content=query_request.query)]}

        final_message = ""

        logger.info(f"Starting Agent Stream for session {query_request.session_id}")
        logger.debug(f"Web Search Enabled: {query_request.enable_web_search}")

        for i, s in enumerate(rag_agent.stream(inputs, config=config)):
            current_node_name = None
            node_output_state = None

            if '__end__' in s:
                current_node_name = '__end__'
                node_output_state = s['__end__']
            else:
                current_node_name = list(s.keys())[0] 
                node_output_state = s[current_node_name]

            event_description = f"Executing node: {current_node_name}"
            event_details = {}
            event_type = "generic_node_execution"

            if current_node_name == "router":
                route_decision = node_output_state.get('route')
                # Check for overridden route if web search was disabled
                initial_decision = node_output_state.get('initial_router_decision', route_decision)
                override_reason = node_output_state.get('router_override_reason', None)

                if override_reason:
                    event_description = f"Router initially decided: '{initial_decision}'. Overridden to: '{route_decision}' because {override_reason}."
                    event_details = {"initial_decision": initial_decision, "final_decision": route_decision, "override_reason": override_reason}
                else:
                    event_description = f"Router decided: '{route_decision}'"
                    event_details = {"decision": route_decision, "reason": "Based on initial query analysis."}
                event_type = "router_decision"
            elif current_node_name == "rag_lookup":
                rag_content = node_output_state.get("rag", "")

                # Extract source document info from RAG content
                source_docs = []
                if "[Source:" in rag_content:
                    import re
                    sources = re.findall(r'\[Source: ([^,]+),', rag_content)
                    source_docs = list(set(sources))  # Get unique sources

                rag_content_summary = rag_content[:300] + "..." if len(rag_content) > 300 else rag_content
                rag_sufficient = node_output_state.get("route") == "answer"

                if rag_sufficient:
                    source_info = f" from {', '.join(source_docs)}" if source_docs else ""
                    event_description = f"RAG Lookup performed{source_info}. Content found and deemed sufficient. Proceeding to answer."
                    event_details = {
                        "retrieved_content_summary": rag_content_summary,
                        "sufficiency_verdict": "Sufficient",
                        "source_documents": source_docs
                    }
                else:
                    source_info = f" from {', '.join(source_docs)}" if source_docs else ""
                    event_description = f"RAG Lookup performed{source_info}. Content NOT sufficient. Diverting to web search."
                    event_details = {
                        "retrieved_content_summary": rag_content_summary,
                        "sufficiency_verdict": "Not Sufficient",
                        "source_documents": source_docs
                    }

                event_type = "rag_action"
            elif current_node_name == "web_search":
                web_content_summary = node_output_state.get("web", "")[:200] + "..."
                event_description = f"Web Search performed. Results retrieved. Proceeding to answer."
                event_details = {"retrieved_content_summary": web_content_summary}
                event_type = "web_action"
            elif current_node_name == "answer":
                event_description = "Generating final answer using gathered context."
                event_type = "answer_generation"
            elif current_node_name == "__end__":
                event_description = "Agent process completed."
                event_type = "process_end"

            trace_events_for_frontend.append(
                TraceEvent(
                    step=i + 1,
                    node_name=current_node_name,
                    description=event_description,
                    details=event_details,
                    event_type=event_type
                )
            )
            logger.debug(f"Streamed Event: Step {i+1} - Node: {current_node_name} - Desc: {event_description}")

        # Get the final state from the last yielded item in the stream
        final_actual_state_dict = None
        if s:
            if '__end__' in s:
                final_actual_state_dict = s['__end__']
            else:
                if list(s.keys()):
                    final_actual_state_dict = s[list(s.keys())[0]]

        if final_actual_state_dict and "messages" in final_actual_state_dict:
            for msg in reversed(final_actual_state_dict["messages"]):
                if isinstance(msg, AIMessage):
                    final_message = msg.content
                    break
        
        if not final_message:
            logger.error("Agent finished, but no final AIMessage found in the final state")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent did not return a valid response. Please try again."
            )

        logger.info(f"Agent Stream Ended for session {query_request.session_id}. Response length: {len(final_message)} chars")
        logger.debug(f"Final Response Preview: {final_message[:200]}...")

        return AgentResponse(response=final_message, trace_events=trace_events_for_frontend)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error during agent invocation for session {query_request.session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again."
        )
    

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Admin Endpoint to Clear All Documents ---
@app.delete("/admin/clear-database/")
@limiter.limit("2/hour")  # Very restrictive rate limit for destructive operation
async def clear_database(request: Request):
    """
    ADMIN ENDPOINT: Deletes all documents from the Pinecone vector database.
    Use with caution - this operation cannot be undone!
    """
    try:
        logger.warning("ADMIN: Clear database operation initiated")
        result = clear_all_documents()

        if result["status"] == "success":
            logger.info(f"ADMIN: Successfully cleared {result['vectors_deleted']} vectors from database")
            return {
                "status": "success",
                "message": result["message"],
                "vectors_deleted": result["vectors_deleted"]
            }
        else:
            logger.error(f"ADMIN: Failed to clear database - {result['message']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
    except Exception as e:
        logger.error(f"ADMIN: Error during clear database operation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear database: {str(e)}"
        )
