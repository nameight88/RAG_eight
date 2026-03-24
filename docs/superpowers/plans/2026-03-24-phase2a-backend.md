# Phase 2A: FastAPI 백엔드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 RAG 엔진을 FastAPI로 래핑해 챗봇 API 서버를 구축한다. PostgreSQL(Supabase)에 대화 이력을 저장하고 CORS 설정으로 Vercel 프론트와 연동한다.

**Architecture:** `backend/` 디렉토리를 루트에 신규 생성한다. `rag_service.py`가 기존 `rag_system.py`를 래핑하고, `routers/chat.py`가 API 엔드포인트를 담당한다. DB는 SQLAlchemy async + Supabase PostgreSQL을 사용한다.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (async), asyncpg, python-dotenv, python-docx, LangChain (기존), OpenAI API

**전제 조건:** Phase 1 완료 (날짜 정규화 + 메타데이터 강화 임베딩)

---

## 파일 맵

```
backend/
├── main.py                      # FastAPI 앱 + CORS 설정
├── requirements.txt             # 백엔드 의존성
├── .env.example                 # 환경변수 템플릿
├── routers/
│   ├── __init__.py
│   └── chat.py                  # POST /api/chat, GET /api/conversations
├── services/
│   ├── __init__.py
│   ├── rag_service.py           # 기존 RAG 엔진 래퍼
│   └── llm_service.py           # LLM API 추상화 (OpenAI/Claude 전환 가능)
├── models/
│   ├── __init__.py
│   └── schemas.py               # Pydantic 요청/응답 스키마
├── db/
│   ├── __init__.py
│   ├── database.py              # SQLAlchemy async 엔진 + 세션
│   ├── models.py                # ORM 테이블 정의
│   └── crud.py                  # DB CRUD 함수
└── tests/
    ├── __init__.py
    └── test_chat_router.py      # API 엔드포인트 테스트
```

---

## Task 1: 백엔드 프로젝트 초기화

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/main.py`

- [ ] **Step 1: backend 디렉토리 및 하위 구조 생성**

```bash
mkdir -p backend/routers backend/services backend/models backend/db backend/tests
touch backend/routers/__init__.py backend/services/__init__.py
touch backend/models/__init__.py backend/db/__init__.py backend/tests/__init__.py
```

- [ ] **Step 2: requirements.txt 작성**

`backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
sqlalchemy==2.0.35
asyncpg==0.29.0
aiosqlite>=0.19.0
python-dotenv==1.0.1
pydantic==2.9.0
httpx==0.27.0
openai==1.51.0
anthropic==0.34.0
python-docx==1.1.2
pytest==7.4.4
pytest-asyncio==0.23.8
langchain==0.3.26
langchain-community==0.3.26
langchain-openai==0.3.26
faiss-cpu==1.9.0
```

- [ ] **Step 3: .env.example 작성**

`backend/.env.example`:
```
# LLM 설정
LLM_PROVIDER=openai          # openai 또는 anthropic
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini        # gpt-4o-mini 또는 claude-haiku-4-5-20251001

# 데이터베이스 (Supabase PostgreSQL)
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname

# RAG 설정
VECTOR_DB_PATH=../data/vector_db
CORPUS_MANAGEMENT=../data/fss_management_parsed.json
CORPUS_SANCTIONS=../data/fss_sanctions_parsed.json

# CORS
ALLOWED_ORIGINS=http://localhost:5173,https://your-vercel-app.vercel.app
```

- [ ] **Step 4: main.py 작성**

`backend/main.py`:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from routers import chat
from db.database import create_tables

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(title="FSS RAG API", version="1.0.0", lifespan=lifespan)

# CORS 설정 — Vercel 프론트엔드 + 로컬 개발 허용
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(chat.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: 서버 기동 확인**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# 브라우저에서 http://localhost:8000/health → {"status": "ok"} 확인
```

- [ ] **Step 6: 커밋**

```bash
git add backend/
git commit -m "feat: initialize FastAPI backend project structure"
```

---

## Task 2: Pydantic 스키마 정의

**Files:**
- Create: `backend/models/schemas.py`

- [ ] **Step 1: 스키마 작성**

`backend/models/schemas.py`:
```python
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid


class ChatFilters(BaseModel):
    doc_type: Optional[str] = None          # "제재결과" | "경영유의사항"
    date_from: Optional[str] = None         # "YYYY-MM-DD"
    date_to: Optional[str] = None           # "YYYY-MM-DD"
    institution: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    filters: ChatFilters = ChatFilters()
    conversation_id: Optional[str] = None   # 기존 대화 이어받기


class SourceDocument(BaseModel):
    institution: str
    date: str
    doc_type: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime


class MessageResponse(BaseModel):
    id: str
    role: str                               # "user" | "assistant"
    content: str
    sources: Optional[List[SourceDocument]] = None
    created_at: datetime
```

- [ ] **Step 2: import 확인**

```bash
cd backend
python -c "from models.schemas import ChatRequest, ChatResponse; print('schemas OK')"
```

- [ ] **Step 3: 커밋**

```bash
git add backend/models/schemas.py
git commit -m "feat: add Pydantic request/response schemas"
```

---

## Task 3: 데이터베이스 설정 및 ORM 모델

**Files:**
- Create: `backend/db/database.py`
- Create: `backend/db/models.py`
- Create: `backend/db/crud.py`

- [ ] **Step 1: database.py 작성**

`backend/db/database.py`:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        from db import models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
```

- [ ] **Step 2: models.py 작성**

`backend/db/models.py`:
```python
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from db.database import Base
import uuid


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(Text, default="새 대화")
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
```

- [ ] **Step 3: crud.py 작성**

`backend/db/crud.py`:
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import Conversation, Message
import uuid


async def create_conversation(db: AsyncSession, title: str = "새 대화") -> Conversation:
    conv = Conversation(id=str(uuid.uuid4()), title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_conversations(db: AsyncSession) -> list[Conversation]:
    result = await db.execute(
        select(Conversation).order_by(Conversation.created_at.desc()).limit(50)
    )
    return result.scalars().all()


async def create_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    sources: list | None = None,
) -> Message:
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_messages(db: AsyncSession, conversation_id: str) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return result.scalars().all()
```

- [ ] **Step 4: 테이블 생성 확인 (SQLite로 로컬 테스트)**

```bash
cd backend
DATABASE_URL=sqlite+aiosqlite:///./test.db python -c "
import asyncio
from db.database import create_tables
asyncio.run(create_tables())
print('테이블 생성 완료')
"
```

- [ ] **Step 5: 커밋**

```bash
git add backend/db/
git commit -m "feat: add async SQLAlchemy DB models and CRUD"
```

---

## Task 4: LLM 서비스 추상화

**Files:**
- Create: `backend/services/llm_service.py`

- [ ] **Step 1: llm_service.py 작성**

`backend/services/llm_service.py`:
```python
"""
LLM API 추상화 레이어
- 환경변수 LLM_PROVIDER로 OpenAI / Anthropic 전환
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def build_rag_prompt(query: str, context_docs: List[Dict]) -> List[Dict]:
    """RAG 컨텍스트를 포함한 프롬프트 메시지 생성"""
    context = "\n\n".join([
        f"[{d.get('institution', '')} / {d.get('date', '')} / {d.get('doc_type', '')}]\n{d.get('content', '')}"
        for d in context_docs
    ])
    system_msg = (
        "당신은 금융감독원의 제재결과 및 경영유의사항을 분석하는 전문가입니다. "
        "아래 참고 문서를 바탕으로 질문에 한국어로 명확하고 정확하게 답변하세요. "
        "참고 문서에 없는 내용은 추측하지 마세요."
    )
    user_msg = f"참고 문서:\n{context}\n\n질문: {query}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


async def generate_answer(query: str, context_docs: List[Dict]) -> str:
    """LLM API 호출하여 답변 생성"""
    messages = build_rag_prompt(query, context_docs)

    if LLM_PROVIDER == "anthropic":
        import anthropic
        # AsyncAnthropic 사용 — sync 클라이언트는 이벤트 루프 차단 위험
        client = anthropic.AsyncAnthropic(api_key=LLM_API_KEY)
        response = await client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=messages[0]["content"],
            messages=messages[1:],
        )
        return response.content[0].text

    else:  # openai (기본)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=LLM_API_KEY)
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content
```

- [ ] **Step 2: import 확인**

```bash
cd backend
python -c "from services.llm_service import generate_answer; print('llm_service OK')"
```

- [ ] **Step 3: 커밋**

```bash
git add backend/services/llm_service.py
git commit -m "feat: add LLM service abstraction (OpenAI/Anthropic switchable)"
```

---

## Task 5: RAG 서비스 래퍼

**Files:**
- Create: `backend/services/rag_service.py`

- [ ] **Step 1: rag_service.py 작성**

`backend/services/rag_service.py`:
```python
"""
기존 rag_system.py를 FastAPI에서 사용하기 위한 래퍼
- RAGSystem을 싱글턴으로 초기화
- 검색 결과를 API 스키마 형식으로 변환
"""
import sys
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트(RAG_eight/)를 sys.path에 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

_rag_system = None


def get_rag_system():
    """RAGSystem 싱글턴 반환"""
    global _rag_system
    if _rag_system is None:
        from rag_system import RAGSystem
        _rag_system = RAGSystem()
        _rag_system.load_vector_store()  # 실제 메서드명: load_vector_store (복수 아님)
    return _rag_system


async def search_and_answer(
    query: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    institution: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Dict:
    """
    RAG 검색 + LLM 답변 생성
    반환: {"answer": str, "sources": list}
    """
    rag = get_rag_system()

    # Phase 1에서 추가한 search_with_filters 사용
    if hasattr(rag, 'search_with_filters'):
        result = rag.search_with_filters(
            query=query,
            date_from=date_from,
            date_to=date_to,
            institution=institution,
            doc_type=doc_type,
        )
    else:
        result = rag.answer_question(query)

    # sources 형식 통일
    sources = []
    for s in result.get("sources", []):
        meta = s.get("metadata", s)
        sources.append({
            "institution": meta.get("institution", ""),
            "date": meta.get("date", ""),
            "doc_type": meta.get("doc_type", ""),
            "excerpt": s.get("content", "")[:200],
        })

    return {
        "answer": result.get("answer", ""),
        "sources": sources,
    }
```

- [ ] **Step 2: import 확인**

```bash
cd backend
python -c "from services.rag_service import get_rag_system; print('rag_service OK')"
```

- [ ] **Step 3: 커밋**

```bash
git add backend/services/rag_service.py
git commit -m "feat: add RAG service wrapper for FastAPI"
```

---

## Task 5-B: pytest 설정 파일 추가

**Files:**
- Create: `backend/pytest.ini`

- [ ] **Step 1: pytest.ini 작성**

`backend/pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
```

`asyncio_mode = auto` 없이 `pytest-asyncio==0.23.8`에서 `@pytest.mark.asyncio`만 사용하면 경고 또는 실행 오류가 발생한다.

- [ ] **Step 2: 커밋**

```bash
git add backend/pytest.ini
git commit -m "chore: configure pytest asyncio_mode=auto"
```

---

## Task 6: Chat 라우터 구현

**Files:**
- Create: `backend/routers/chat.py`
- Create: `backend/tests/test_chat_router.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_chat_router.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_health_endpoint():
    from main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_returns_answer():
    """RAG 서비스를 모킹하여 /api/chat 응답 구조 검증"""
    mock_result = {
        "answer": "테스트 답변입니다.",
        "sources": [
            {"institution": "테스트은행", "date": "2023-01-01",
             "doc_type": "제재결과", "excerpt": "테스트 발췌"}
        ]
    }
    with patch("services.rag_service.search_and_answer", new=AsyncMock(return_value=mock_result)):
        with patch("db.crud.create_conversation", new=AsyncMock()) as mock_conv:
            mock_conv.return_value = type("C", (), {"id": "test-uuid", "title": "테스트"})()
            with patch("db.crud.create_message", new=AsyncMock()):
                from main import app
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post("/api/chat", json={"query": "카드사 제재 알려줘"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "conversation_id" in data
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

```bash
cd backend
pytest tests/test_chat_router.py -v
```

Expected: `ImportError` 또는 `404` (라우터 없음)

- [ ] **Step 3: chat.py 라우터 구현**

`backend/routers/chat.py`:
```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from models.schemas import ChatRequest, ChatResponse, ConversationSummary
from services.rag_service import search_and_answer
from db.database import get_db
from db import crud
import uuid

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 대화 생성 또는 기존 대화 사용
    if request.conversation_id:
        conv_id = request.conversation_id
    else:
        title = request.query[:20] + ("..." if len(request.query) > 20 else "")
        conv = await crud.create_conversation(db, title=title)
        conv_id = conv.id

    # 사용자 메시지 저장
    await crud.create_message(db, conv_id, "user", request.query)

    # RAG 검색 + 답변 생성
    f = request.filters
    result = await search_and_answer(
        query=request.query,
        date_from=f.date_from,
        date_to=f.date_to,
        institution=f.institution,
        doc_type=f.doc_type,
    )

    # 어시스턴트 메시지 저장
    await crud.create_message(
        db, conv_id, "assistant", result["answer"], sources=result["sources"]
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conv_id,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    convs = await crud.get_conversations(db)
    return [
        ConversationSummary(id=c.id, title=c.title, created_at=c.created_at)
        for c in convs
    ]


@router.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, db: AsyncSession = Depends(get_db)):
    msgs = await crud.get_messages(db, conv_id)
    return [
        {"id": m.id, "role": m.role, "content": m.content,
         "sources": m.sources, "created_at": m.created_at}
        for m in msgs
    ]
```

- [ ] **Step 4: 테스트 실행해 통과 확인**

```bash
cd backend
pytest tests/test_chat_router.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add backend/routers/chat.py backend/tests/test_chat_router.py
git commit -m "feat: implement /api/chat and /api/conversations endpoints"
```

---

## Task 7: 통합 확인 및 배포 설정

> DB 초기화는 Task 1의 `main.py`에 `lifespan` 컨텍스트 매니저로 이미 포함됨. 별도 추가 불필요.

- [ ] **Step 1: 서버 기동 후 API 수동 테스트**

```bash
cd backend
DATABASE_URL=sqlite+aiosqlite:///./test.db uvicorn main:app --reload --port 8000
```

별도 터미널에서:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "전자금융 관련 제재 알려줘"}'
```

Expected: `{"answer": "...", "sources": [...], "conversation_id": "uuid"}`

- [ ] **Step 3: 전체 테스트 실행**

```bash
cd backend
pytest tests/ -v
```

- [ ] **Step 4: 최종 커밋**

```bash
git add backend/main.py
git commit -m "feat: add DB startup initialization, phase2a backend complete"
```
