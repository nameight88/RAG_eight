"""GET /api/conversations 라우터."""
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.models import Conversation, Message
from backend.schemas import ConversationSummary, MessageOut, SourceDoc, FilterParams


router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation).order_by(Conversation.updated_at.desc()).limit(50)
    )
    conversations = result.scalars().all()

    summaries = []
    for conv in conversations:
        count_result = await db.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        )
        count = count_result.scalar() or 0
        summaries.append(
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat(),
                message_count=count,
            )
        )
    return summaries


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(conversation_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    output = []
    for msg in messages:
        sources = [SourceDoc(**s) for s in (msg.sources or [])]
        filters = FilterParams(**msg.filters) if msg.filters else None
        output.append(
            MessageOut(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=sources,
                filters=filters,
                created_at=msg.created_at.isoformat(),
            )
        )
    return output
