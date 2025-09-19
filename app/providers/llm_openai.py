# adapters/llm_openai.py — thin wrapper around OpenAI chat models
# (keeps your "Choose which AI brain to use" split, now cleanly isolated)
import os
from langchain_openai import ChatOpenAI

def get_openai_llm(default_model: str | None = None, temperature: float = 0):
    model = default_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Use OpenAI's GPT models (requires API key in environment)
    return ChatOpenAI(model=model, temperature=temperature)
# Note: You can add more OpenAI-specific configurations here as needed