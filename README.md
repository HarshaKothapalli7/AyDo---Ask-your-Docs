#  AyDo — Ask Your Docs

AyDo is a Retrieval-Augmented Generation (RAG) system that allows users to query documents through natural language. It combines the power of embeddings, vector search (via Pinecone), and large language models (LLMs) to generate grounded, hallucination-free answers — all via a clean, interactive UI.

---

##  Features

- Upload PDF documents and build a document knowledge base
- Ask natural language questions based on the uploaded content
- Retrieval-Augmented Generation using vector search + LLMs
- Optional fallback to web search when documents don't suffice
- Agent Trace Viewer: Step-by-step insight into how answers are formed
- Secrets managed securely via `.env` (excluded from version control)

---

---

##  Tech Stack

| Layer         | Tools Used                                |
|---------------|--------------------------------------------|
|  LLM         | OpenAI GPT / HuggingFace Transformers      |
|  Retrieval   | Pinecone vector DB + MiniLM embeddings     |
|  Orchestration | LangChain + LangGraph                    |
|  Frontend    | Streamlit (with light/dark mode support)   |
|  Backend     | FastAPI (REST API for doc upload & chat)   |

---

##  Setup Instructions

 1. Clone and configure `.env`

```bash
git clone https://github.com/yourusername/aydo-rag.git
cd aydo-rag
cp .env.example .env  # Then manually add your API keys
```
    2. Create and activate a virtual environment
    
    ```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux


 3. Install dependencies

```bash
pip install -r requirements.txt
```


## Running the Application

1. Start the FastAPI backend:

```bash
cd backend
uvicorn main:app --reload

2. Start the Streamlit frontend:

```bash
cd frontend
streamlit run app.py





