# DATA MODELS - Defining what data looks like
from typing import List

from pydantic import BaseModel


class QueryIn(BaseModel):
    question: str


class RAGOut(BaseModel):
    answer: str
    sources: List[str]
    latency_ms: float
