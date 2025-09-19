# pipelines/tools.py — optional LLM tools/skills
# Keep small & visible; you can register these with your LLM if needed.
# For now, a placeholder no-op tool to show structure.
from typing import Literal

def hello_tool(name: str) -> str:
    """A tiny example tool the LLM could call."""
    return f"Hello, {name}!"
