import os, json, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag_chain import make_chain

# Load environment variables from .env file at startup
load_dotenv()

# Configuration: Get settings from environment with sensible defaults
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
LOGS_DIR = os.getenv("LOGS_DIR", "./data/logs")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Choose between "ollama" or "openai"

# Ensure logs directory exists for storing user feedback
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI application with descriptive title
app = FastAPI(title="RAG Helpdesk API")

# Build the RAG chain once at startup (expensive operation, do it early)
# This connects to the vector database and sets up the LLM
chain, retriever = make_chain(INDEX_DIR, EMB_MODEL, LLM_PROVIDER)

# === DATA MODELS (Pydantic schemas for API validation) ===

class QueryIn(BaseModel):
    """Input model for user questions"""
    question: str

class RAGOut(BaseModel):
    """Output model for RAG responses with metadata"""
    answer: str           # The AI's response
    sources: list[str]    # List of source documents referenced
    latency_ms: float     # How long the query took (for monitoring)

class FeedbackIn(BaseModel):
    """Input model for user feedback on answers"""
    question: str                    # Original question asked
    answer: str                      # Answer that was provided
    rating: int                      # User rating (e.g., 1-5 stars)
    comment: str | None = None       # Optional user comment

# === API ENDPOINTS ===

@app.get("/")
def home():
    return {"msg": "RAG Helpdesk API. See /docs or /health."}

@app.get("/health")
def health():
    """
    Health check endpoint - verify the API is running
    Used by load balancers, monitoring systems, etc.
    """
    return {"status": "ok"}

@app.post("/query", response_model=RAGOut)
async def query(payload: QueryIn):
    """
    Main RAG endpoint - answer user questions using the knowledge base
    """
    # Start timing for latency measurement
    t0 = time.time()
    
    try:
        # Invoke the RAG chain asynchronously for better performance
        resp = await chain.ainvoke(payload.question)
        
        # Extract answer text from LLM response (different LLMs return different formats)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        
        # Ensure answer is always a string (defensive programming)
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Parse sources from the answer (assumes sources are bullet points starting with "- ")
        sources = [
            line.replace("- ", "").strip() 
            for line in answer.splitlines() 
            if line.strip().startswith("- ")
        ]
        
        # Return structured response with timing information
        return RAGOut(
            answer=answer, 
            sources=sources, 
            latency_ms=(time.time() - t0) * 1000  # Convert to milliseconds
        )
        
    except Exception as e:
        # Convert any error to HTTP 500 with error details
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(fb: FeedbackIn):
    """
    Collect user feedback for answer quality evaluation and system improvement
    """
    # Convert Pydantic model to dict and add timestamp
    rec = fb.model_dump() | {"ts": time.time()}
    
    # Append feedback to JSONL file (one JSON object per line)
    # JSONL format makes it easy to process large feedback datasets
    with open(Path(LOGS_DIR)/"feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    return {"status": "logged"}