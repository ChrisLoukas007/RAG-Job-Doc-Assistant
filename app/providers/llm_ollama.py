# adapters/llm_ollama.py — thin wrapper around Ollama chat models
import os
from langchain_ollama import ChatOllama # Ollama chat model wrapper

def get_ollama_llm( 
    model: str | None = None, # default Ollama model if not set in env
    base_url: str | None = None, # default Ollama server URL if not set in env
    temperature: float = 0, # consistent answers 
):
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", model or "llama3.1:8b"), 
        base_url=os.getenv("OLLAMA_BASE_URL", base_url or "http://localhost:11434"), # Ollama server URL
        temperature=temperature,     # Consistent answers 
        num_predict=512,             # Reasonable length
        top_p=0.9,                   # Focused but natural
        repeat_penalty=1.1,          # No repetition
    )
# Note: You can add more Ollama-specific configurations here as needed