from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: str = "#00D4FF"


class CategoryOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    color: str
    created_at: str


class DocumentOut(BaseModel):
    id: str
    category_id: int
    filename: str
    original_name: str
    chunk_count: int
    status: str
    error_msg: Optional[str]
    created_at: str


class QueryRequest(BaseModel):
    question: str
    category_id: Optional[int] = None
    top_k: int = 5
    use_rewrite: bool = False


class SourceChunk(BaseModel):
    filename: str
    original_name: str
    page_number: int
    snippet: str
