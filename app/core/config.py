# Env & basic configuration
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()  # Read .env file to get secret keys and settings

DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
LOGS_DIR = os.getenv("LOGS_DIR", "./data/logs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CORS_ORIGINS: List[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Create logs folder if it doesn't exist (kept behavior)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
