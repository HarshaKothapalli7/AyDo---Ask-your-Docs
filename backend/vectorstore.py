import os
from datetime import datetime
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import PINECONE_API_KEY

#set environment variables for pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

#initialize pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME = "langgraph-rag-index" #

# Global variable to track the most recent document ID
_most_recent_document_id = None

# Global variable to track the most recent batch ID
_most_recent_batch_id = None

def get_most_recent_document_id():
    """Returns the most recently uploaded document ID."""
    return _most_recent_document_id

def set_most_recent_document_id(document_id: str):
    """Sets the most recently uploaded document ID."""
    global _most_recent_document_id
    _most_recent_document_id = document_id
    print(f"Most recent document ID set to: {document_id}")

def get_most_recent_batch_id():
    """Returns the most recently uploaded batch ID."""
    return _most_recent_batch_id

def set_most_recent_batch_id(batch_id: str):
    """Sets the most recently uploaded batch ID."""
    global _most_recent_batch_id
    _most_recent_batch_id = batch_id
    print(f"Most recent batch ID set to: {batch_id}")

#retriever function to get relevant documents from pinecone
def get_retriever(k: int = 5, filter_dict: dict = None):
    """
    Initializes and returns the Pinecone vector store retriever.

    Args:
        k: Number of documents to retrieve (default: 5)
        filter_dict: Optional metadata filter (e.g., {"document_id": "doc_123"})

    Returns:
        A retriever instance
    """
    # Ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Changed dimension for 'sentence-transformers/all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region as per your Pinecone setup
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Configure search kwargs
    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
        print(f"Retriever configured with filter: {filter_dict}")

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(text_content: str, filename: str = "unknown.pdf", document_id: str = None, batch_id: str = None):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    Each chunk includes metadata: filename, document_id, batch_id, upload_timestamp, and chunk_index.

    Args:
        text_content: The text content to add
        filename: Name of the source file
        document_id: Unique identifier for this document (auto-generated if not provided)
        batch_id: Unique identifier for the upload batch/session (auto-generated if not provided)
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    # Generate document ID if not provided
    if document_id is None:
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"

    # Generate batch ID if not provided (single file upload gets its own batch)
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get current timestamp
    upload_timestamp = datetime.now().isoformat()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # Split text into chunks
    text_chunks = text_splitter.split_text(text_content)

    # Create Document objects with metadata for each chunk
    documents = []
    for idx, chunk in enumerate(text_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "filename": filename,
                "document_id": document_id,
                "batch_id": batch_id,
                "upload_timestamp": upload_timestamp,
                "chunk_index": idx,
                "total_chunks": len(text_chunks)
            }
        )
        documents.append(doc)

    print(f"Splitting document '{filename}' into {len(documents)} chunks for indexing...")
    print(f"Document ID: {document_id}")
    print(f"Batch ID: {batch_id}")
    print(f"Upload timestamp: {upload_timestamp}")

    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

    # Update the most recent document ID and batch ID
    set_most_recent_document_id(document_id)
    set_most_recent_batch_id(batch_id)

    return document_id

# --- Function to clear all documents from the vector store ---
def clear_all_documents():
    """
    Deletes all vectors from the Pinecone index.
    This is useful for starting fresh with a clean database.

    Returns:
        dict: Status information about the deletion
    """
    try:
        # Get the index
        index = pc.Index(INDEX_NAME)

        # Get index stats before deletion
        stats_before = index.describe_index_stats()
        total_vectors_before = stats_before.get('total_vector_count', 0)

        print(f"Index '{INDEX_NAME}' currently has {total_vectors_before} vectors.")

        if total_vectors_before == 0:
            print("Index is already empty.")
            return {
                "status": "success",
                "message": "Index is already empty",
                "vectors_deleted": 0
            }

        # Delete all vectors from all namespaces
        # Get all namespaces
        namespaces = stats_before.get('namespaces', {})

        if not namespaces:
            # If no namespaces info, delete from default namespace
            print("Deleting all vectors from default namespace...")
            index.delete(delete_all=True)
        else:
            # Delete from each namespace
            for namespace in namespaces.keys():
                print(f"Deleting all vectors from namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)

        # Reset the most recent document ID
        global _most_recent_document_id
        _most_recent_document_id = None

        print(f"Successfully deleted all vectors from index '{INDEX_NAME}'.")

        return {
            "status": "success",
            "message": f"Successfully cleared all documents from index '{INDEX_NAME}'",
            "vectors_deleted": total_vectors_before
        }

    except Exception as e:
        print(f"Error clearing documents: {e}")
        return {
            "status": "error",
            "message": f"Failed to clear documents: {str(e)}",
            "vectors_deleted": 0
        }