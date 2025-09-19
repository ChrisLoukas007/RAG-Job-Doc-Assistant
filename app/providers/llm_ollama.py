# adapters/llm_ollama.py — thin wrapper around Ollama chat models
# (keeps the same knobs you had set)
import os
from langchain_ollama import ChatOllama

def get_ollama_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
):
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", model or "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", base_url or "http://localhost:11434"),
        temperature=temperature,     # Consistent answers
        num_predict=512,             # Reasonable length
        top_p=0.9,                   # Focused but natural
        repeat_penalty=1.1,          # No repetition
    )
# Note: You can add more Ollama-specific configurations here as needed