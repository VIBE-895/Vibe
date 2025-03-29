# Created by guxu at 3/28/25
from typing import List

from pydantic import BaseModel

class Summary(BaseModel):
    setting: str
    topic: str
    key_terms: List[str]
    summary: str

class ChunkSummary(BaseModel):
    summary: str

class QuestionsForRAG(BaseModel):
    questions: List[str]