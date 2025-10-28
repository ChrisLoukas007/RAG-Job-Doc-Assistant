# HTTP layer only - The FastAPI routes (endpoints) for the RAG API.
import json
import re  # for pattern matching in text
import time
from pathlib import Path  # for handling file paths safely
from typing import Any, List

from fastapi import (  # web framework components for building API routes
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
)
from fastapi.responses import StreamingResponse  # for streaming responses (SSE)

from ..core.logging import get_logger  # custom logging setup
from ..deps import (  # dependency injection for getting the RAG chain and retriever
    get_chain,
    get_retriever,
)
from ..models.schemas import QueryIn, RAGOut  # request/response data models
from ..pipelines.rag import (
    stream_chain_answer,  # function to stream answers from the chain
)

router = APIRouter()  # Create a router for our API endpoints
logger = get_logger(__name__)

# Utilities - helper functions
# This pattern finds lines that start with bullet points (- or *)
BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")


def coerce_to_text(resp: Any) -> str:
    """Convert any type of AI response (objects, dicts, lists, strings) into plain text string."""
    try:
        # Handle LangChain message objects
        from langchain_core.messages import BaseMessage  # type: ignore

        if isinstance(resp, BaseMessage):
            return str(resp.content or "")
    except Exception:
        pass

    # If it's already text, return it
    if isinstance(resp, str):
        return resp

    # If it's a dictionary, look for common text fields
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        if "text" in resp:
            return str(resp["text"])
        if "answer" in resp:
            return str(resp["answer"])
        return str(resp)

    # If it's a list, join all parts together
    if isinstance(resp, list):
        parts: List[str] = []
        for item in resp:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    # Fallback: convert anything else to string
    return str(resp)


# API ENDPOINTS - The actual web services


@router.get("/")
def home():
    """Simple welcome message for the API root."""
    return {"msg": "RAG Helpdesk API. See /docs or /health."}


@router.get("/health")
def health():
    """Health check endpoint - tells you if the API is running."""
    return {"status": "ok"}


@router.post("/query", response_model=RAGOut)
async def query(
    payload: QueryIn = Body(...),  # The user's question comes in the request body
    chain=Depends(get_chain),  # Get the AI chain (the "brain" of our system)
    retriever=Depends(get_retriever),  # Get the document searcher
):
    """
    Main endpoint: Ask a question and get an answer with sources.
    """
    t0 = time.time()  # Start timing how long this takes
    try:
        # 1) Ask the AI chain for an answer
        raw = await chain.ainvoke(
            payload.question
        )  # Get raw response from the AI chain
        answer_text: str = coerce_to_text(raw)  # Convert response to plain text

        # 2) Try to extract source files from the answer text
        # Look for bullet points that might contain file names
        parsed: List[str] = []
        for line in answer_text.splitlines():
            m = BULLET_RE.match(line)  # Check if line starts with - or *
            if m:
                parsed.append(m.group(1).strip())  # Extract the text after the bullet

        # 3) If no sources found in answer, get them from search results instead
        if not parsed:
            docs = await retriever.aget_relevant_documents(payload.question)
            for d in docs:
                src = d.metadata.get("source", "unknown")
                parsed.append(Path(src).name)  # Just the filename, not full path

        # 4) Remove duplicate sources but keep the order
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
            latency_ms=(time.time() - t0) * 1000.0,  # How long it took in milliseconds
        )
    except Exception as e:
        # If anything goes wrong, return a 500 error
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def stream(
    q: str = Query(..., description="User question"),  # Question comes as URL parameter
    chain=Depends(get_chain),  # Get the AI chain
    retriever=Depends(get_retriever),  # Get the document searcher
):
    """
    STREAMING ENDPOINT: Get answers word-by-word as they're generated.
    Uses Server-Sent Events (SSE) - like a live text feed.

    Events sent to client:
    - 'meta': Status updates and source citations
    - 'token': Each word/piece of the answer as it's generated
    - 'done': Signal that we're finished
    - 'error': If something went wrong
    """

    async def event_generator():
        """
        Generator function that produces the streaming response.
        Each 'yield' sends data to the client immediately.
        """
        try:
            # Tell client we started processing
            yield "event: meta\ndata: " + json.dumps({"status": "started"}) + "\n\n"

            # Send source citations early (before the answer starts)
            try:
                docs = await retriever.aget_relevant_documents(q)
                citations = [
                    {
                        "title": Path(d.metadata.get("source", "unknown")).name,
                        "url": d.metadata.get("source", ""),
                    }
                    for d in docs
                ]
                yield (
                    "event: meta\ndata: "
                    + json.dumps({"citations": citations})
                    + "\n\n"
                )
            except Exception:
                # Don't break the stream if getting citations fails
                pass

            # Stream the answer piece by piece as the AI generates it
            async for piece in stream_chain_answer(chain, q):
                # Send each chunk of text immediately
                yield "event: token\ndata: " + json.dumps({"token": piece}) + "\n\n"

            # Tell client we're completely done
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            # If error occurs, tell client what went wrong and close gracefully
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
            yield "event: done\ndata: {}\n\n"

    # Headers to make streaming work properly
    headers = {
        "Cache-Control": "no-cache",  # Don't cache the stream
        "Connection": "keep-alive",  # Keep connection open
        "X-Accel-Buffering": "no",  # Tell nginx not to buffer (for production)
    }
    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers=headers
    )
