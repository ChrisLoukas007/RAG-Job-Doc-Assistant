# deps.py — wiring helpers used by routes (get_llm, get_retriever)
from fastapi import Request
from .core.config import INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER
from .pipelines.rag import make_chain

def get_chain(request: Request):
    """
    Get the RAG chain (AI brain) for answering questions.
    Uses caching to avoid rebuilding it every time.
    """
    # Try to get prebuilt components from app memory (fastest option)
    chain = getattr(request.app.state, "chain", None)        # main RAG chain 
    retriever = getattr(request.app.state, "retriever", None) # document searcher
    
    # If not found in memory, build them fresh and save for next time
    if chain is None or retriever is None: 
        chain, retriever = make_chain(INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER)
        # Store in app memory so we don't have to rebuild next time
        request.app.state.chain = chain
        request.app.state.retriever = retriever
        
    return chain

def get_retriever(request: Request):
    """
    Get the document retriever (searches through your knowledge base).
    Makes sure the chain is built first, then returns the retriever part.
    """
    # This ensures both chain and retriever are available
    _ = get_chain(request)  # Build chain if needed (includes retriever)
    return request.app.state.retriever