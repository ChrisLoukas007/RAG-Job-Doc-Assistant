# DATA MODELS - Defining what data looks like
from pydantic import BaseModel
from typing import List

class QueryIn(BaseModel):
    question: str

class RAGOut(BaseModel):
    answer: str
    sources: List[str]
    latency_ms: float

class FeedbackIn(BaseModel):
    rating: int | None = None
    comment: str | None = None
    question: str | None = None
    answer: str | None = None
# Note: You can add more data models here as needed