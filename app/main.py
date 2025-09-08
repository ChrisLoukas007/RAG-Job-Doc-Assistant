import os # for reading env vars 
import json
import time
import re # for pattern matching in text 
from pathlib import Path  # for handling file paths safely
from typing import Any, List 
from contextlib import asynccontextmanager 

from dotenv import load_dotenv # for loading .env files
from fastapi import FastAPI, HTTPException, Query, Body  # web framework
from fastapi.middleware.cors import CORSMiddleware      # for handling CORS which is for browser security
from fastapi.responses import StreamingResponse  # for streaming responses
from pydantic import BaseModel  # fro data validation

from .rag_chain import make_chain, stream_chain_answer

# Env & basic configuration
load_dotenv() # Read .env file to get secret keys and settings

DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
LOGS_DIR = os.getenv("LOGS_DIR", "./data/logs")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

# Create logs folder if it doesn't exist
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# App startup / lifespan - Preparing the RAG chain when server strts. 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build chain once, share via app.state
    chain, retriever = make_chain(INDEX_DIR, EMB_MODEL, LLM_PROVIDER)     # This connects to the knowledge database and sets up the AI
    app.state.chain = chain  # The main RAG chain for answering questions
    app.state.retriever = retriever # The document retriever to find relevant info
    yield # app runs

# Create the FastAPI app with startup/shutdown management 
app = FastAPI(title="RAG Helpdesk API", lifespan=lifespan)

# CORS - allow browsers talk to our api from these origins
app.add_middleware(
    CORSMiddleware,                    # Add browser security handler
    allow_origins=CORS_ORIGINS,       # These websites can call our API
    allow_credentials=True,            # Allow cookies and auth tokens
    allow_methods=["*"],              # Allow all HTTP methods 
    allow_headers=["*"],              # Allow all headers 
)

# DATA MODELS - Defining what data looks like
class QueryIn(BaseModel):
    question: str

class RAGOut(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float

class FeedbackIn(BaseModel):
    question: str
    answer: str
    rating: int
    comment: str | None = None

# Utilities - helper functions 
BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")

def coerce_to_text(resp: Any) -> str:
    """Normalize various LangChain/LLM outputs into plain text."""
    try:
        from langchain_core.messages import BaseMessage  # type: ignore
        if isinstance(resp, BaseMessage):
            return str(resp.content or "")
    except Exception:
        pass

    # Already a string? Just return it
    if isinstance(resp, str):
        return resp

    # If it's a dict, look for common keys
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        if "text" in resp:
            return str(resp["text"])
        if "answer" in resp:
            return str(resp["answer"])
        return str(resp)

    # If it's a list, Join all text parts
    if isinstance(resp, list):
        parts: List[str] = []
        for x in resp:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                t = x.get("text") or x.get("content") or x.get("answer") or ""
                parts.append(str(t))
        return "\n".join(parts)

    # Fallback: just convert to string
    return str(resp)

# API ENDPOINTS - The actual web services
@app.get("/")
def home():
    return {"msg": "RAG Helpdesk API. See /docs or /health."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=RAGOut)
async def query(payload: QueryIn = Body(...)):
    t0 = time.time()
    try:
        # Get the AI components from app memory
        chain = app.state.chain # The RAG chain answe
        retriever = app.state.retriever # The document retriever

        # 1) Ask the chain for an answer
        raw = await chain.ainvoke(payload.question) 
        answer_text: str = coerce_to_text(raw) # Normalize to plain text

        # 2) Try to find source files mentioned in the answer
        parsed: List[str] = []
        for line in answer_text.splitlines():
            m = BULLET_RE.match(line)
            if m:
                parsed.append(m.group(1).strip())

        # 3) If no sources found in answer, get them from the search results
        if not parsed:
            docs = await retriever.aget_relevant_documents(payload.question)
            for d in docs:
                src = d.metadata.get("source", "unknown")
                parsed.append(Path(src).name)

        # 4) Remove duplicates while keeping order
        seen = set()
        sources: List[str] = []
        for s in parsed:
            if s and s not in seen:
                seen.add(s)
                sources.append(Path(s).name)

        # Return the complete response
        return RAGOut(
            answer=answer_text,
            sources=sources,
            latency_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream(q: str = Query(..., description="User question")):
    """
    STREAMING ENDPOINT: Get answers word-by-word as they're generated by the model. 
    This uses Server-Sent Events (SSE) to send data in real-time    
    Events sent:
    - 'meta': Status updates and citations
    - 'token': Each word/piece of the answer
    - 'done': Finished
    - 'error': If something went wrong
    """
    async def event_generator():
        """
        This function generates the streaming response
        """
        # Get AI components
        chain = app.state.chain
        retriever = app.state.retriever

        try:
            # Tell client we started
            yield "event: meta\ndata: " + json.dumps({"status": "started"}) + "\n\n"

            # Send citations early (which documents will be used)
            try:
                docs = await retriever.aget_relevant_documents(q)
                citations = [
                    {
                        "title": Path(d.metadata.get("source", "unknown")).name,
                        "url": d.metadata.get("source", "")
                    }
                    for d in docs
                ]
                yield "event: meta\ndata: " + json.dumps({"citations": citations}) + "\n\n"
            except Exception:
                # Don't break the stream if citations fail
                pass

            # Stream the answer token by token
            async for piece in stream_chain_answer(chain, q):
                # Send each piece of text as it's generated
                yield "event: token\ndata: " + json.dumps({"token": piece}) + "\n\n"

            # Signal that we're done
            yield "event: done\ndata: {}\n\n"
            
        except Exception as e:
            # If error occurs, tell client and close gracefully
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
            yield "event: done\ndata: {}\n\n"

    # Set headers for streaming
    headers = {
        "Cache-Control": "no-cache",      # Don't cache the stream
        "Connection": "keep-alive",       # Keep connection open
        "X-Accel-Buffering": "no",        # Tell nginx not to buffer
    }
    
    # Return streaming response
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream", 
        headers=headers
    )

# Feedback endpoint to collect user ratings/comments
@app.post("/feedback")
def feedback(fb: FeedbackIn):
    rec = fb.model_dump() | {"ts": time.time()}
    with open(Path(LOGS_DIR) / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "logged"}