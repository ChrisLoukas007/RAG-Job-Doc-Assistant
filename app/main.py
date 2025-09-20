# tiny FastAPI entrypoint
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import CORS_ORIGINS, INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER
from .api.routes import router
from .pipelines.rag import make_chain

# App startup / lifespan - Preparing the RAG chain when server starts. 
@asynccontextmanager
async def lifespan(app: FastAPI): 
    # Build chain once, share via app.state
    chain, retriever = make_chain(INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER)  # This connects to the knowledge database and sets up the AI
    app.state.chain = chain  # The main RAG chain for answering questions
    app.state.retriever = retriever  # The document retriever to find relevant info
    yield  # app runs

# Create the FastAPI app with startup/shutdown management 
app = FastAPI(title="RAG Helpdesk API", lifespan=lifespan)

# CORS - allow browsers talk to our api from these origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP layer only  — mount routes
app.include_router(router)
