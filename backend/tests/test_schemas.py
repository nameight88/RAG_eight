"""Pydantic 스키마 검증"""
import pytest
from pydantic import ValidationError


def test_chat_request_requires_message():
    from backend.schemas import ChatRequest
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_request_optional_conversation_id():
    from backend.schemas import ChatRequest
    req = ChatRequest(message="테스트 질문")
    assert req.conversation_id is None
    assert req.message == "테스트 질문"


def test_chat_request_with_filters():
    from backend.schemas import ChatRequest, FilterParams
    req = ChatRequest(
        message="테스트",
        filters=FilterParams(doc_type="sanctions", date_from="2023-01-01")
    )
    assert req.filters.doc_type == "sanctions"
    assert req.filters.date_from == "2023-01-01"
    assert req.filters.date_to is None


def test_source_schema():
    from backend.schemas import SourceDoc
    src = SourceDoc(
        institution="국민은행",
        date="2023-03-15",
        doc_type="제재결과",
        content="관련 내용",
        score=0.87
    )
    assert src.score == 0.87
