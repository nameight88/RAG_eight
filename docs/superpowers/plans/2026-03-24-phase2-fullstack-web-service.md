# Phase 2: 풀스택 웹 서비스 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI 백엔드 + React 프론트엔드 + PostgreSQL DB로 금융감독원 제재/경영유의사항 RAG 챗봇 웹 서비스를 구축한다.

**Architecture:** FastAPI가 RAG 엔진을 내장하고 `/api/chat`, `/api/conversations` 엔드포인트를 제공한다. React(Vite+TypeScript)가 챗봇 UI를 담당하며 대화 이력을 Supabase PostgreSQL에 저장한다. 배포는 Vercel(frontend) + Render(backend) + Supabase(DB) 무료 티어 조합.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, asyncpg, Pydantic v2, React 18, TypeScript, Vite, TailwindCSS, PostgreSQL (Supabase)

---

## 파일 구조

```
RAG_eight/
├── backend/
│   ├── main.py               # FastAPI 앱, CORS, lifespan
│   ├── config.py             # Pydantic Settings (env vars)
│   ├── database.py           # SQLAlchemy async engine + session
│   ├── models.py             # ORM 모델 (Conversation, Message)
│   ├── schemas.py            # Pydantic 요청/응답 스키마
│   ├── rag_service.py        # FSSRagSystem 싱글턴 래퍼
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py           # POST /api/chat
│   │   └── conversations.py  # GET /api/conversations, /messages
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_schemas.py
│   │   └── test_chat_router.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── types.ts
    │   ├── api/client.ts
    │   ├── hooks/useChat.ts
    │   └── components/
    │       ├── Sidebar.tsx
    │       ├── FilterBar.tsx
    │       ├── ChatWindow.tsx
    │       ├── MessageBubble.tsx
    │       └── SourceCard.tsx
    ├── package.json
    └── vite.config.ts
```

---

## Task 1: 백엔드 프로젝트 초기화 + 설정

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/config.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_config.py`

**배경:**
백엔드 의존성과 환경변수 설정을 먼저 확립한다. `config.py`는 Pydantic Settings로 env 파일을 읽어
모든 설정을 타입 안전하게 제공한다.

- [ ] **Step 1: requirements.txt 생성**

`backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy[asyncio]==2.0.36
asyncpg==0.30.0
pydantic-settings==2.5.2
python-dotenv==1.0.1
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 2: config.py 테스트 작성**

`backend/tests/test_config.py`:
```python
"""config.py 환경변수 로드 검증"""
import os
import pytest


def test_settings_load_from_env(monkeypatch):
    """환경변수에서 설정을 올바르게 읽어야 한다"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
    monkeypatch.setenv("VECTOR_DB_PATH", "data/vector_db/fss_sanctions")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")

    # 모듈 재임포트로 새 env 반영
    import importlib
    import backend.config as cfg_module
    importlib.reload(cfg_module)
    settings = cfg_module.get_settings()

    assert settings.openai_api_key == "test-key-123"
    assert "localhost" in settings.database_url
    assert settings.vector_db_path == "data/vector_db/fss_sanctions"


def test_allowed_origins_parsed():
    """ALLOWED_ORIGINS 쉼표 구분 문자열이 리스트로 파싱돼야 한다"""
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:5173,https://myapp.vercel.app"
    import importlib
    import backend.config as cfg_module
    importlib.reload(cfg_module)
    settings = cfg_module.get_settings()

    assert isinstance(settings.allowed_origins, list)
    assert len(settings.allowed_origins) == 2
```

- [ ] **Step 3: config.py 구현**

`backend/config.py`:
```python
"""환경변수 설정 모듈."""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./test.db"
    vector_db_path: str = "data/vector_db/fss_sanctions"
    vector_db_type: str = "FAISS"
    allowed_origins: List[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context):
        # ALLOWED_ORIGINS가 단일 문자열로 들어올 때 파싱
        if isinstance(self.allowed_origins, str):
            object.__setattr__(
                self,
                "allowed_origins",
                [o.strip() for o in self.allowed_origins.split(",")]
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: 테스트 실행**

```bash
cd backend
python -m pytest tests/test_config.py -v
```

Expected: PASSED

- [ ] **Step 5: 커밋**

```bash
git add backend/requirements.txt backend/config.py backend/tests/__init__.py backend/tests/test_config.py
git commit -m "feat: add backend project structure and config"
```

---

## Task 2: DB 모델 + 스키마 정의

**Files:**
- Create: `backend/database.py`
- Create: `backend/models.py`
- Create: `backend/schemas.py`
- Create: `backend/tests/test_schemas.py`

**배경:**
SQLAlchemy async 엔진과 ORM 모델, Pydantic 스키마를 정의한다. DB는 개발 시 SQLite를 사용하고
배포 시 Supabase PostgreSQL로 전환한다 (DATABASE_URL 변경만으로 호환).

- [ ] **Step 1: schemas.py 테스트 작성**

`backend/tests/test_schemas.py`:
```python
"""Pydantic 스키마 검증"""
import pytest
from uuid import uuid4


def test_chat_request_requires_message():
    """message 필드가 없으면 ValidationError"""
    from pydantic import ValidationError
    from backend.schemas import ChatRequest
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_request_optional_conversation_id():
    """conversation_id는 선택 필드다"""
    from backend.schemas import ChatRequest
    req = ChatRequest(message="테스트 질문")
    assert req.conversation_id is None
    assert req.message == "테스트 질문"


def test_chat_request_with_filters():
    """필터 필드가 올바르게 설정된다"""
    from backend.schemas import ChatRequest, FilterParams
    req = ChatRequest(
        message="테스트",
        filters=FilterParams(doc_type="sanctions", date_from="2023-01-01")
    )
    assert req.filters.doc_type == "sanctions"
    assert req.filters.date_from == "2023-01-01"
    assert req.filters.date_to is None


def test_source_schema():
    """SourceDoc 스키마 생성"""
    from backend.schemas import SourceDoc
    src = SourceDoc(
        institution="국민은행",
        date="2023-03-15",
        doc_type="제재결과",
        content="관련 내용",
        score=0.87
    )
    assert src.score == 0.87
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd backend
python -m pytest tests/test_schemas.py -v
```

Expected: ImportError (schemas.py 없음)

- [ ] **Step 3: database.py 구현**

`backend/database.py`:
```python
"""SQLAlchemy 비동기 DB 연결 및 세션 관리."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from backend.config import get_settings


class Base(DeclarativeBase):
    pass


def create_engine_and_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


engine, AsyncSessionLocal = create_engine_and_session()


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """테이블 생성 (앱 시작 시 호출)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

- [ ] **Step 4: models.py 구현**

`backend/models.py`:
```python
"""SQLAlchemy ORM 모델."""
import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


def _uuid():
    return str(uuid.uuid4())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200), default="새 대화")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(10))  # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    filters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
```

- [ ] **Step 5: schemas.py 구현**

`backend/schemas.py`:
```python
"""Pydantic 요청/응답 스키마."""
from typing import Optional
from pydantic import BaseModel


class FilterParams(BaseModel):
    doc_type: Optional[str] = None      # 'sanctions' | 'management' | None
    date_from: Optional[str] = None     # YYYY-MM-DD
    date_to: Optional[str] = None       # YYYY-MM-DD
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
```

- [ ] **Step 6: 테스트 실행**

```bash
cd backend
python -m pytest tests/test_schemas.py -v
```

Expected: PASSED

- [ ] **Step 7: 커밋**

```bash
git add backend/database.py backend/models.py backend/schemas.py backend/tests/test_schemas.py
git commit -m "feat: add DB models and Pydantic schemas"
```

---

## Task 3: RAG 서비스 래퍼 + FastAPI 앱 기반

**Files:**
- Create: `backend/rag_service.py`
- Create: `backend/main.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/tests/test_rag_service.py`

**배경:**
기존 `rag_system.py`의 `FSSRagSystem`을 싱글턴으로 래핑하는 `rag_service.py`를 만든다.
RAG 시스템은 앱 시작 시 한 번만 초기화된다. `main.py`는 FastAPI 앱, CORS, lifespan을 설정한다.

- [ ] **Step 1: rag_service.py 테스트 작성**

`backend/tests/test_rag_service.py`:
```python
"""RAG 서비스 래퍼 테스트"""
import pytest
from unittest.mock import MagicMock, patch


def test_rag_service_returns_answer_and_sources():
    """RAG 서비스가 answer와 sources를 반환해야 한다"""
    from backend.rag_service import RagService

    mock_result = {
        "answer": "전자금융 관련 제재 결과입니다.",
        "sources": [
            {
                "content": "전자금융 위반 사항",
                "metadata": {
                    "institution": "국민은행",
                    "date": "2023-01-15",
                    "doc_type": "제재결과",
                },
                "score": 0.89,
            }
        ],
    }

    service = RagService.__new__(RagService)
    service._rag_system = MagicMock()
    service._rag_system.answer_question.return_value = mock_result

    result = service.query("전자금융 제재 알려줘")

    assert result["answer"] == "전자금융 관련 제재 결과입니다."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["institution"] == "국민은행"


def test_rag_service_query_passes_filters():
    """필터 파라미터가 answer_question에 전달돼야 한다"""
    from backend.rag_service import RagService

    service = RagService.__new__(RagService)
    service._rag_system = MagicMock()
    service._rag_system.answer_question.return_value = {"answer": "결과", "sources": []}

    service.query(
        "질문",
        date_from="2023-01-01",
        institution="신한은행",
        doc_type="sanctions",
    )

    call_kwargs = service._rag_system.answer_question.call_args
    assert call_kwargs.kwargs.get("date_from") == "2023-01-01"
    assert call_kwargs.kwargs.get("institution") == "신한은행"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd backend
python -m pytest tests/test_rag_service.py -v
```

Expected: ImportError

- [ ] **Step 3: rag_service.py 구현**

`backend/rag_service.py`:
```python
"""FSSRagSystem 싱글턴 래퍼."""
import sys
import os
from typing import Any, Dict, Optional

# 프로젝트 루트를 sys.path에 추가 (rag_system.py 접근)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


class RagService:
    """FSSRagSystem을 싱글턴으로 래핑한 서비스."""

    def __init__(self, vector_db_path: str, use_faiss: bool = True):
        from rag_system import FSSRagSystem
        self._rag_system = FSSRagSystem(
            vector_db_path=vector_db_path,
            use_openai_embeddings=True,
            use_faiss=use_faiss,
        )

    def query(
        self,
        question: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        institution: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 질의 수행 후 answer와 sources 반환."""
        return self._rag_system.answer_question(
            question=question,
            date_from=date_from,
            date_to=date_to,
            institution=institution,
            doc_type=doc_type,
        )


# 싱글턴 인스턴스 (lifespan에서 초기화)
_rag_service: Optional[RagService] = None


def get_rag_service() -> RagService:
    if _rag_service is None:
        raise RuntimeError("RAG 서비스가 초기화되지 않았습니다.")
    return _rag_service


def init_rag_service(vector_db_path: str, use_faiss: bool = True) -> RagService:
    global _rag_service
    _rag_service = RagService(vector_db_path, use_faiss)
    return _rag_service
```

- [ ] **Step 4: main.py 구현**

`backend/main.py`:
```python
"""FastAPI 앱 진입점."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.database import init_db
from backend.rag_service import init_rag_service
from backend.routers import chat, conversations


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    init_rag_service(
        vector_db_path=settings.vector_db_path,
        use_faiss=(settings.vector_db_type == "FAISS"),
    )
    yield


app = FastAPI(title="FSS RAG API", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: routers/__init__.py 생성**

```bash
touch backend/routers/__init__.py
```

- [ ] **Step 6: 테스트 실행**

```bash
cd backend
python -m pytest tests/test_rag_service.py -v
```

Expected: PASSED

- [ ] **Step 7: 커밋**

```bash
git add backend/rag_service.py backend/main.py backend/routers/__init__.py backend/tests/test_rag_service.py
git commit -m "feat: add RAG service wrapper and FastAPI app skeleton"
```

---

## Task 4: 채팅 + 대화 라우터 구현

**Files:**
- Create: `backend/routers/chat.py`
- Create: `backend/routers/conversations.py`
- Create: `backend/tests/test_chat_router.py`

**배경:**
`POST /api/chat`은 RAG 질의 → DB 저장 → 응답 반환을 수행한다.
새 대화(`conversation_id=null`)이면 Conversation을 생성하고 첫 질문으로 title을 설정한다.

- [ ] **Step 1: 테스트 작성**

`backend/tests/test_chat_router.py`:
```python
"""채팅 라우터 단위 테스트"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_chat_creates_new_conversation():
    """conversation_id 없이 요청 시 새 대화가 생성돼야 한다"""
    from backend.schemas import ChatRequest, FilterParams
    from backend.routers.chat import _handle_chat

    mock_db = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.query.return_value = {
        "answer": "테스트 답변",
        "sources": [],
    }

    request = ChatRequest(message="전자금융 제재 알려줘")
    result = await _handle_chat(request, mock_db, mock_rag)

    assert result["answer"] == "테스트 답변"
    assert result["conversation_id"] is not None


@pytest.mark.asyncio
async def test_chat_uses_existing_conversation():
    """기존 conversation_id가 있으면 해당 대화에 메시지를 추가한다"""
    from backend.schemas import ChatRequest
    from backend.routers.chat import _handle_chat

    mock_db = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.query.return_value = {"answer": "답변", "sources": []}

    existing_id = "test-conv-id-123"
    request = ChatRequest(message="추가 질문", conversation_id=existing_id)
    result = await _handle_chat(request, mock_db, mock_rag)

    assert result["conversation_id"] == existing_id
```

- [ ] **Step 2: chat.py 구현**

`backend/routers/chat.py`:
```python
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

    # sources 정규화
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

    # 대화 생성 또는 조회
    conv_id = request.conversation_id or str(uuid.uuid4())
    if not request.conversation_id:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(id=conv_id, title=title)
        db.add(conv)

    # 사용자 메시지 저장
    user_msg_id = str(uuid.uuid4())
    user_msg = Message(
        id=user_msg_id,
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
```

- [ ] **Step 3: conversations.py 구현**

`backend/routers/conversations.py`:
```python
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
```

- [ ] **Step 4: 테스트 실행**

```bash
cd backend
python -m pytest tests/test_chat_router.py -v
```

Expected: PASSED

- [ ] **Step 5: 커밋**

```bash
git add backend/routers/chat.py backend/routers/conversations.py backend/tests/test_chat_router.py
git commit -m "feat: add chat and conversations routers"
```

---

## Task 5: React 프론트엔드 초기화 + 타입/API 클라이언트

**Files:**
- Create: `frontend/` (Vite 프로젝트)
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api/client.ts`

**배경:**
Vite + React + TypeScript로 프론트엔드를 초기화하고 TailwindCSS를 설정한다.
API 클라이언트는 환경변수 `VITE_API_URL`을 기반으로 백엔드와 통신한다.

- [ ] **Step 1: Vite 프로젝트 생성**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

- [ ] **Step 2: TailwindCSS 설정**

`frontend/tailwind.config.js`:
```js
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

`frontend/src/index.css` 상단:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 3: 환경변수 파일 생성**

`frontend/.env.development`:
```
VITE_API_URL=http://localhost:8000
```

`frontend/.env.production`:
```
VITE_API_URL=https://your-backend.onrender.com
```

- [ ] **Step 4: types.ts 작성**

`frontend/src/types.ts`:
```typescript
export interface FilterParams {
  doc_type?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  institution?: string | null;
}

export interface SourceDoc {
  institution: string;
  date: string;
  doc_type: string;
  content: string;
  score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: SourceDoc[];
  filters?: FilterParams | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatRequest {
  conversation_id?: string | null;
  message: string;
  filters: FilterParams;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  sources: SourceDoc[];
}
```

- [ ] **Step 5: API 클라이언트 작성**

`frontend/src/api/client.ts`:
```typescript
import type { ChatRequest, ChatResponse, Conversation, Message } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export const api = {
  chat: (body: ChatRequest): Promise<ChatResponse> =>
    request('/api/chat', { method: 'POST', body: JSON.stringify(body) }),

  listConversations: (): Promise<Conversation[]> =>
    request('/api/conversations'),

  getMessages: (conversationId: string): Promise<Message[]> =>
    request(`/api/conversations/${conversationId}/messages`),
};
```

- [ ] **Step 6: 빌드 확인**

```bash
cd frontend
npm run build
```

Expected: 빌드 성공 (dist/ 생성)

- [ ] **Step 7: 커밋**

```bash
git add frontend/
git commit -m "feat: initialize React frontend with Vite, Tailwind, and API client"
```

---

## Task 6: React 컴포넌트 구현

**Files:**
- Create: `frontend/src/hooks/useChat.ts`
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/FilterBar.tsx`
- Create: `frontend/src/components/ChatWindow.tsx`
- Create: `frontend/src/components/MessageBubble.tsx`
- Create: `frontend/src/components/SourceCard.tsx`
- Modify: `frontend/src/App.tsx`

**배경:**
Streamlit 인터페이스와 동일한 UX를 React로 구현한다.
2-컬럼 레이아웃: 좌측 대화 이력 사이드바 + 우측 채팅 영역.

- [ ] **Step 1: useChat.ts 구현**

`frontend/src/hooks/useChat.ts`:
```typescript
import { useState, useCallback } from 'react';
import { api } from '../api/client';
import type { Message, Conversation, FilterParams, ChatResponse } from '../types';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState<FilterParams>({});

  const loadConversations = useCallback(async () => {
    const data = await api.listConversations();
    setConversations(data);
  }, []);

  const selectConversation = useCallback(async (id: string) => {
    setActiveConversationId(id);
    const msgs = await api.getMessages(id);
    setMessages(msgs);
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;

    // 사용자 메시지 즉시 표시 (optimistic update)
    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: text,
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);
    setIsLoading(true);

    try {
      const response: ChatResponse = await api.chat({
        conversation_id: activeConversationId,
        message: text,
        filters,
      });

      if (!activeConversationId) {
        setActiveConversationId(response.conversation_id);
        await loadConversations();
      }

      const botMsg: Message = {
        id: response.message_id,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev.filter(m => m.id !== tempUserMsg.id), tempUserMsg, botMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [activeConversationId, filters, loadConversations]);

  const newConversation = useCallback(() => {
    setActiveConversationId(null);
    setMessages([]);
  }, []);

  return {
    conversations, messages, isLoading, filters,
    loadConversations, selectConversation, sendMessage, newConversation, setFilters,
  };
}
```

- [ ] **Step 2: Sidebar.tsx 구현**

`frontend/src/components/Sidebar.tsx`:
```tsx
import type { Conversation } from '../types';

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function Sidebar({ conversations, activeId, onSelect, onNew }: Props) {
  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col h-full">
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={onNew}
          className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium"
        >
          + 새 대화
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.map(conv => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`w-full text-left px-3 py-2 rounded-lg mb-1 text-sm truncate ${
              activeId === conv.id ? 'bg-gray-700' : 'hover:bg-gray-800'
            }`}
          >
            {conv.title}
          </button>
        ))}
      </div>
    </aside>
  );
}
```

- [ ] **Step 3: FilterBar.tsx 구현**

`frontend/src/components/FilterBar.tsx`:
```tsx
import type { FilterParams } from '../types';

interface Props {
  filters: FilterParams;
  onChange: (f: FilterParams) => void;
}

export function FilterBar({ filters, onChange }: Props) {
  return (
    <div className="flex gap-3 p-3 bg-gray-50 border-b items-center flex-wrap">
      <select
        value={filters.doc_type ?? ''}
        onChange={e => onChange({ ...filters, doc_type: e.target.value || null })}
        className="text-sm border rounded px-2 py-1"
      >
        <option value="">전체 문서</option>
        <option value="sanctions">제재결과</option>
        <option value="management">경영유의사항</option>
      </select>
      <input
        type="date"
        value={filters.date_from ?? ''}
        onChange={e => onChange({ ...filters, date_from: e.target.value || null })}
        className="text-sm border rounded px-2 py-1"
        placeholder="시작 날짜"
      />
      <span className="text-gray-400 text-sm">~</span>
      <input
        type="date"
        value={filters.date_to ?? ''}
        onChange={e => onChange({ ...filters, date_to: e.target.value || null })}
        className="text-sm border rounded px-2 py-1"
        placeholder="종료 날짜"
      />
      <input
        type="text"
        value={filters.institution ?? ''}
        onChange={e => onChange({ ...filters, institution: e.target.value || null })}
        placeholder="기관명"
        className="text-sm border rounded px-2 py-1 w-32"
      />
    </div>
  );
}
```

- [ ] **Step 4: SourceCard.tsx 구현**

`frontend/src/components/SourceCard.tsx`:
```tsx
import type { SourceDoc } from '../types';

export function SourceCard({ source }: { source: SourceDoc }) {
  return (
    <div className="bg-gray-50 border-l-4 border-blue-400 p-3 rounded text-sm mb-2">
      <div className="font-semibold text-blue-800">{source.institution}</div>
      <div className="text-gray-500 text-xs mb-1">
        {source.date} · {source.doc_type} · 유사도 {(source.score * 100).toFixed(0)}%
      </div>
      <div className="text-gray-700 line-clamp-2">{source.content}</div>
    </div>
  );
}
```

- [ ] **Step 5: MessageBubble.tsx 구현**

`frontend/src/components/MessageBubble.tsx`:
```tsx
import { useState } from 'react';
import type { Message } from '../types';
import { SourceCard } from './SourceCard';

export function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-2xl rounded-2xl px-4 py-3 ${
        isUser ? 'bg-blue-100 text-blue-900' : 'bg-white border text-gray-800'
      }`}>
        <p className="whitespace-pre-wrap">{message.content}</p>
        {!isUser && message.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="text-xs text-blue-600 hover:underline"
            >
              참고 문서 {message.sources.length}개 {showSources ? '접기' : '보기'}
            </button>
            {showSources && (
              <div className="mt-2">
                {message.sources.map((src, i) => (
                  <SourceCard key={i} source={src} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: ChatWindow.tsx 구현**

`frontend/src/components/ChatWindow.tsx`:
```tsx
import { useEffect, useRef, useState } from 'react';
import type { Message } from '../types';
import { MessageBubble } from './MessageBubble';

interface Props {
  messages: Message[];
  isLoading: boolean;
  onSend: (text: string) => void;
}

export function ChatWindow({ messages, isLoading, onSend }: Props) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">금융감독원 제재/경영유의사항을 질문해보세요</p>
            <p className="text-sm mt-2">예: "전자금융 관련 최근 제재 사례를 알려줘"</p>
          </div>
        )}
        {messages.map(msg => <MessageBubble key={msg.id} message={msg} />)}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border rounded-2xl px-4 py-3 text-gray-400 animate-pulse">
              답변 생성 중...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            전송
          </button>
        </div>
      </form>
    </div>
  );
}
```

- [ ] **Step 7: App.tsx 조립**

`frontend/src/App.tsx`:
```tsx
import { useEffect } from 'react';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar';
import { FilterBar } from './components/FilterBar';
import { ChatWindow } from './components/ChatWindow';

export default function App() {
  const {
    conversations, messages, isLoading, filters,
    loadConversations, selectConversation, sendMessage, newConversation, setFilters,
  } = useChat();

  useEffect(() => { loadConversations(); }, [loadConversations]);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar
        conversations={conversations}
        activeId={null}
        onSelect={selectConversation}
        onNew={newConversation}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="px-6 py-3 bg-blue-800 text-white font-semibold text-lg">
          금융감독원 제재정보 챗봇
        </header>
        <FilterBar filters={filters} onChange={setFilters} />
        <ChatWindow messages={messages} isLoading={isLoading} onSend={sendMessage} />
      </div>
    </div>
  );
}
```

- [ ] **Step 8: 빌드 확인**

```bash
cd frontend
npm run build
```

Expected: 빌드 성공

- [ ] **Step 9: 커밋**

```bash
git add frontend/src/
git commit -m "feat: implement React chatbot UI components"
```

---

## Task 7: 로컬 통합 테스트 + 배포 준비

**Files:**
- Create: `backend/Dockerfile` (Render 배포용)
- Create: `frontend/vercel.json`
- Modify: `frontend/.env.production`

**배경:**
백엔드와 프론트엔드를 로컬에서 함께 실행하여 통합 테스트를 수행하고,
Render/Vercel 배포 설정 파일을 추가한다.

- [ ] **Step 1: 백엔드 로컬 실행 확인**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

`http://localhost:8000/health` → `{"status": "ok"}` 확인

- [ ] **Step 2: 프론트엔드 로컬 실행 확인**

```bash
cd frontend
npm run dev
```

`http://localhost:5173` 에서 챗봇 UI 확인

- [ ] **Step 3: Dockerfile 작성**

`backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: vercel.json 작성**

`frontend/vercel.json`:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

- [ ] **Step 5: 최종 테스트 실행**

```bash
cd backend
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 6: 최종 커밋**

```bash
git add backend/Dockerfile frontend/vercel.json
git commit -m "feat: add deployment configuration for Render and Vercel"
```

---

## 완료 기준

- [ ] `backend/`: 모든 단위 테스트 통과
- [ ] `GET http://localhost:8000/health` → `{"status": "ok"}`
- [ ] `POST http://localhost:8000/api/chat` → 답변 + sources 반환
- [ ] `GET http://localhost:8000/api/conversations` → 대화 목록 반환
- [ ] `http://localhost:5173` → 챗봇 UI 정상 렌더링
- [ ] 메시지 전송 → 답변 수신 → 참고문서 표시 정상 동작
- [ ] `npm run build` 빌드 성공
