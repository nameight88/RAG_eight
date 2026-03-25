"""채팅 라우터 단위 테스트"""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_handle_chat_creates_new_conversation():
    """conversation_id 없이 요청 시 새 대화가 생성돼야 한다"""
    from backend.schemas import ChatRequest
    from backend.routers.chat import _handle_chat

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.query.return_value = {
        "answer": "테스트 답변",
        "sources": [],
    }

    request = ChatRequest(message="전자금융 제재 알려줘")
    result = await _handle_chat(request, mock_db, mock_rag)

    assert result["answer"] == "테스트 답변"
    assert result["conversation_id"] is not None
    assert len(result["conversation_id"]) > 0


@pytest.mark.asyncio
async def test_handle_chat_uses_existing_conversation():
    """기존 conversation_id가 있으면 그대로 사용한다"""
    from backend.schemas import ChatRequest
    from backend.routers.chat import _handle_chat

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.query.return_value = {"answer": "답변", "sources": []}

    existing_id = "test-conv-id-123"
    request = ChatRequest(message="추가 질문", conversation_id=existing_id)
    result = await _handle_chat(request, mock_db, mock_rag)

    assert result["conversation_id"] == existing_id


@pytest.mark.asyncio
async def test_handle_chat_normalizes_sources():
    """소스 메타데이터가 SourceDoc 형식으로 정규화돼야 한다"""
    from backend.schemas import ChatRequest
    from backend.routers.chat import _handle_chat

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.query.return_value = {
        "answer": "답변",
        "sources": [
            {
                "content": "긴 내용 " * 100,
                "metadata": {
                    "institution": "국민은행",
                    "date": "2023-01-15",
                    "doc_type": "제재결과",
                },
                "score": 0.89,
            }
        ],
    }

    request = ChatRequest(message="질문")
    result = await _handle_chat(request, mock_db, mock_rag)

    assert len(result["sources"]) == 1
    src = result["sources"][0]
    assert src["institution"] == "국민은행"
    assert src["score"] == 0.89
    assert len(src["content"]) <= 300  # content 300자 제한
