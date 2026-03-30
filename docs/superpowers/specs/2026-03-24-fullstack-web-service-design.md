# Phase 2: FastAPI + React + PostgreSQL 풀스택 웹 서비스 설계

**작성일:** 2026-03-24
**프로젝트:** RAG_eight — 금융감독원 문서 RAG 챗봇 웹 서비스
**목표:** Streamlit MVP를 FastAPI + React + PostgreSQL 3티어 서비스로 전환
**배포 목적:** 포트폴리오 (무료 티어 기반)

---

## 1. 개요

### 1.1 목적
- 포트폴리오용 풀스택 웹 서비스
- 금융 제재결과·경영유의사항 문서를 챗봇 형식으로 질의응답
- 무료 티어 배포로 비용 최소화

### 1.2 MVP 범위 (1단계)
- 챗봇 UI (채팅 인터페이스)
- RAG 기반 질의응답 (FastAPI → 기존 RAG 엔진 연동)
- 대화 이력 저장 (PostgreSQL)
- 기관명/날짜 필터

### 1.3 2단계 (MVP 이후)
- 요약 결과 Word(.docx) 파일 다운로드

---

## 2. 아키텍처

### 2.1 전체 구성

```
사용자 브라우저
    ↓ HTTPS
[React Frontend — Vercel 무료]
    ↓ REST API
[FastAPI Backend — Render 무료]
    ├── RAG 엔진 (기존 코드 재사용)
    ├── FAISS 벡터 DB (서버 내장 파일)
    └── LLM API (OpenAI 또는 Claude API)
    ↓
[PostgreSQL — Supabase 무료]
```

### 2.2 배포 환경

| 계층 | 서비스 | 무료 제한 | 비고 |
|------|--------|-----------|------|
| Frontend | Vercel | 무제한 | React 최적화 |
| Backend | Render | 15분 비활성 시 슬립 | cold start ~30초 |
| Database | Supabase | 500MB, 50K 요청/월 | PostgreSQL 14 |
| LLM | OpenAI API | 종량제 | 소량 사용 시 월 $1~5 |
| Vector DB | FAISS (파일) | 없음 | Git 저장소에 인덱스 파일 포함하여 배포 |

### 2.3 LLM 전략
- 기존 로컬 HuggingFace 모델(`beomi/llama-2-ko-7b`) → **OpenAI `gpt-4o-mini` 또는 Claude `claude-haiku-4-5-20251001`로 교체**
- 이유: 무료 클라우드 서버(512MB RAM)에서 7B 모델 실행 불가
- 환경변수(`LLM_PROVIDER`, `LLM_API_KEY`)로 LLM 제공자 전환 가능하도록 설계

### 2.4 FAISS 인덱스 배포 전략
- Render 무료 티어는 디스크가 에페메럴(ephemeral)이므로 재시작 시 파일 시스템 초기화 위험 존재
- **결정: FAISS 인덱스 파일을 Git 저장소에 포함하여 배포 시 함께 적용**
- 포트폴리오 목적이므로 인덱스 파일 크기가 허용 범위(수백 MB 이하) 내인 경우 이 방식이 현실적
- 인덱스 크기가 문제가 될 경우 Git LFS 사용

### 2.5 인증 정책
- **인증 없음 — 공개 데모로 운영**
- `conversations` 테이블에 `user_id` 없음, 모든 사용자가 전체 대화 이력 공유
- 포트폴리오 목적에 맞는 의도적 결정이며, 향후 인증 추가 시 `user_id` 컬럼 추가로 확장 가능

---

## 3. 백엔드 설계 (FastAPI)

### 3.0 CORS 설정

Vercel(프론트)과 Render(백엔드)는 도메인이 다르므로 `main.py`에 `CORSMiddleware` 설정이 **필수**:
- 허용 오리진: Vercel 배포 도메인 + `localhost:5173` (개발용)
- 허용 메서드: `GET`, `POST`
- 허용 헤더: `Content-Type`, `Authorization`

### 3.1 디렉토리 구조

```
backend/
├── main.py                 # FastAPI 앱 진입점 (CORSMiddleware 설정 포함)
├── routers/
│   ├── chat.py             # 채팅 엔드포인트
│   └── documents.py        # 문서 조회 엔드포인트
├── services/
│   ├── rag_service.py      # 기존 RAG 엔진 래퍼
│   └── llm_service.py      # LLM API 추상화 레이어
├── models/
│   └── schemas.py          # Pydantic 요청/응답 스키마
├── db/
│   ├── database.py         # PostgreSQL 연결 (SQLAlchemy)
│   └── crud.py             # DB CRUD 함수
└── requirements.txt
```

### 3.2 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/chat` | RAG 검색 + LLM 답변 생성 |
| `GET` | `/api/conversations` | 대화 이력 목록 |
| `GET` | `/api/conversations/{id}` | 특정 대화 조회 |
| `GET` | `/api/documents` | 문서 메타데이터 목록 (필터 가능) |
| `GET` | `/health` | 서버 상태 확인 (Render 슬립 방지용, 무료 티어는 슬립 완전 방지 불가) |

### 3.3 채팅 요청/응답 스키마

**요청 (`POST /api/chat`):**
```json
{
  "query": "2023년 이후 카드사 제재 내용을 알려줘",
  "filters": {
    "doc_type": "제재결과",
    "date_from": "2023-01-01",
    "date_to": null,
    "institution": null
  },
  "conversation_id": "uuid-or-null"
}
```

**응답:**
```json
{
  "answer": "2023년 이후 카드사 제재 건으로는...",
  "sources": [
    {
      "institution": "삼성카드",
      "date": "2023-05-12",
      "doc_type": "제재결과",
      "excerpt": "..."
    }
  ],
  "conversation_id": "uuid"
}
```

---

## 4. 데이터베이스 설계 (PostgreSQL)

### 4.1 테이블

**`conversations`**
```sql
id          UUID PRIMARY KEY DEFAULT gen_random_uuid()
created_at  TIMESTAMP DEFAULT NOW()
title       TEXT        -- 첫 질문 앞 20자 자동 생성
```

**`messages`**
```sql
id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE
role            TEXT NOT NULL   -- 'user' | 'assistant'
content         TEXT NOT NULL
sources         JSONB           -- 참고문서 메타데이터 배열
created_at      TIMESTAMP DEFAULT NOW()
```

---

## 5. 프론트엔드 설계 (React)

### 5.1 디렉토리 구조

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatWindow.tsx       # 채팅 메시지 표시 영역
│   │   ├── ChatInput.tsx        # 질문 입력 + 필터
│   │   ├── ConversationList.tsx # 사이드바 대화 목록
│   │   └── SourceCard.tsx       # 참고문서 카드
│   ├── pages/
│   │   └── ChatPage.tsx         # 메인 페이지 (단일 페이지)
│   ├── api/
│   │   └── client.ts            # FastAPI 호출 함수
│   └── App.tsx
├── package.json
└── .env.local                   # VITE_API_URL=...
```

### 5.2 UI 레이아웃

```
┌─────────────────────────────────────────────┐
│  [사이드바]          [채팅 영역]              │
│  ─────────          ──────────────────────── │
│  + 새 대화           [질문] 2023년 카드사...  │
│                                              │
│  대화 목록           [답변] 2023년 이후...    │
│  - 카드사 제재 조회                           │
│  - 경영유의사항...    [참고문서]              │
│                       ┌──────────────────┐  │
│                       │ 삼성카드         │  │
│                       │ 2023.05.12 제재  │  │
│                       └──────────────────┘  │
│                                              │
│                      [필터: 유형▼ 기간▼ ]    │
│                      [입력창___________↑]    │
└─────────────────────────────────────────────┘
```

### 5.3 기술 스택

| 항목 | 선택 | 이유 |
|------|------|------|
| 빌드 도구 | Vite | 빠른 개발 서버 |
| UI 컴포넌트 | shadcn/ui + Tailwind | 빠른 구현, 포트폴리오 퀄리티 |
| 상태 관리 | React Query (TanStack) | 서버 상태 캐싱 |
| HTTP 클라이언트 | Axios | 간결한 API 호출 |

---

## 6. 에러 처리

| 상황 | 처리 |
|------|------|
| Render 슬립 cold start | 프론트에서 로딩 스피너 + "서버 시작 중" 안내 |
| LLM API 오류 | 재시도 1회 후 "잠시 후 다시 시도" 메시지 |
| RAG 검색 결과 없음 | "관련 문서를 찾지 못했습니다" 명시적 응답 |
| DB 연결 실패 | 이력 저장 실패 시 답변은 제공, 저장만 skip |

---

## 7. 2단계: Word 문서 다운로드 (MVP 이후)

MVP 완성 후 추가:
- `GET /api/download/{conversation_id}` 엔드포인트
- `python-docx`로 대화 요약 + 참고문서 목록을 Word 형식으로 생성
- 프론트엔드에 "Word로 저장" 버튼 추가

---

## 8. 개발 순서

1. **Phase 1 완료** (날짜 정규화 + 메타데이터 임베딩)
2. **백엔드 골격** — FastAPI 프로젝트 생성, DB 연결, `/health` 엔드포인트
3. **RAG 연동** — 기존 `rag_system.py`를 `rag_service.py`로 래핑, `/api/chat` 구현
4. **프론트엔드** — React 프로젝트 생성, 채팅 UI 구현
5. **배포** — Supabase DB → Render 백엔드 → Vercel 프론트엔드
6. **2단계** — Word 다운로드 기능 추가
