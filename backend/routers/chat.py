"""POST /api/chat 라우터."""
import uuid
from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.models import Conversation, Message
from backend.schemas import ChatRequest, ChatResponse, SourceDoc
from backend.rag_service import get_rag_service, RagService


router = APIRouter()


async def _handle_chat(
    request: ChatRequest,
    db: AsyncSession,
    rag: RagService,
) -> Dict[str, Any]:
    """채팅 처리 로직 (테스트 가능하도록 분리)."""
    # RAG 질의
    rag_result = rag.query(
        question=request.message,
        date_from=request.filters.date_from,
        date_to=request.filters.date_to,
        institution=request.filters.institution,
        doc_type=request.filters.doc_type,
    )

    answer = rag_result.get("answer", "답변을 생성하지 못했습니다.")
    raw_sources = rag_result.get("sources", [])

    # sources 정규화 (content 300자 제한)
    sources = [
        SourceDoc(
            institution=s.get("metadata", {}).get("institution", ""),
            date=s.get("metadata", {}).get("date", ""),
            doc_type=s.get("metadata", {}).get("doc_type", ""),
            content=s.get("content", "")[:300],
            score=float(s.get("score", 0.0)),
        )
        for s in raw_sources
    ]

    # 대화 생성 또는 기존 사용
    conv_id = request.conversation_id or str(uuid.uuid4())
    if not request.conversation_id:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(id=conv_id, title=title)
        db.add(conv)

    # 사용자 메시지 저장
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_id,
        role="user",
        content=request.message,
        filters=request.filters.model_dump(),
    )
    db.add(user_msg)

    # 봇 메시지 저장
    bot_msg_id = str(uuid.uuid4())
    bot_msg = Message(
        id=bot_msg_id,
        conversation_id=conv_id,
        role="assistant",
        content=answer,
        sources=[s.model_dump() for s in sources],
    )
    db.add(bot_msg)

    await db.commit()

    return {
        "conversation_id": conv_id,
        "message_id": bot_msg_id,
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    rag: RagService = Depends(get_rag_service),
):
    result = await _handle_chat(request, db, rag)
    return ChatResponse(**result)
