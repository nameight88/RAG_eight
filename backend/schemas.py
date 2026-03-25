"""Pydantic 요청/응답 스키마."""
from typing import Optional
from pydantic import BaseModel


class FilterParams(BaseModel):
    doc_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    institution: Optional[str] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    filters: FilterParams = FilterParams()


class SourceDoc(BaseModel):
    institution: str = ""
    date: str = ""
    doc_type: str = ""
    content: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: list[SourceDoc] = []


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: list[SourceDoc] = []
    filters: Optional[FilterParams] = None
    created_at: str
