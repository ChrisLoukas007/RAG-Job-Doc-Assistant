import os, json, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag_chain import make_chain
import re
import time
from pathlib import Path
from typing import List, Any
from fastapi import HTTPException

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

import re
from pathlib import Path

BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")

def coerce_to_text(resp: Any) -> str:
    """Turn various LangChain/LLM outputs into a plain string."""
    # LangChain message types
    try:
        from langchain_core.messages import BaseMessage  # type: ignore
        if isinstance(resp, BaseMessage):
            return str(resp.content or "")
    except Exception:
        pass

    # Already a string?
    if isinstance(resp, str):
        return resp

    # Some models return dicts with content/text/answer fields
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        if "text" in resp:
            return str(resp["text"])
        if "answer" in resp:
            return str(resp["answer"])
        return str(resp)

    # Some return a list[dict|str] parts
    if isinstance(resp, list):
        parts: List[str] = []
        for x in resp:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                t = x.get("text") or x.get("content") or x.get("answer") or ""
                parts.append(str(t))
        return "\n".join(parts)

    # Fallback
    return str(resp)

@app.post("/query", response_model=RAGOut)
async def query(payload: QueryIn):
    t0 = time.time()
    try:
        # 1) Ask the chain
        raw = await chain.ainvoke(payload.question)
        answer_text: str = coerce_to_text(raw)

        # 2) Parse sources from the answer (accept '-' or '*')
        parsed: List[str] = []
        for line in answer_text.splitlines():
            m = BULLET_RE.match(line)
            if m:
                parsed.append(m.group(1).strip())

        # 3) Fallback: filenames from retriever if nothing parsed
        if not parsed:
            docs = await retriever.aget_relevant_documents(payload.question)
            for d in docs:
                src = d.metadata.get("source", "unknown")
                parsed.append(Path(src).name)

        # 4) Deduplicate while preserving order
        seen = set()
        sources: List[str] = []
        for s in parsed:
            if s not in seen and s:
                seen.add(s)
                # keep only filename if a path sneaks in
                sources.append(Path(s).name)

        return RAGOut(
            answer=answer_text,
            sources=sources,
            latency_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
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