# deps.py — wiring helpers used by routes (get_llm, get_retriever)
from fastapi import Request
from .core.config import INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER
from .pipelines.rag import make_chain

def get_chain(request: Request):
    # Prefer prebuilt on app.state (loaded in lifespan); fallback to lazy build
    chain = getattr(request.app.state, "chain", None)
    retriever = getattr(request.app.state, "retriever", None)
    if chain is None or retriever is None:
        chain, retriever = make_chain(INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER)
        request.app.state.chain = chain
        request.app.state.retriever = retriever
    return chain

def get_retriever(request: Request):
    # Ensure retriever is available (uses same lazy path)
    _ = get_chain(request)
    return request.app.state.retriever
