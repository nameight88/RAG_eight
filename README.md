# FSS RAG — 금융감독원 제재/경영유의사항 챗봇

금융감독원(FSS) 제재결과 및 경영유의사항 문서를 기반으로 한 RAG(Retrieval-Augmented Generation) 질의응답 시스템입니다.

---

## 주요 기능

| 기능 | 내용 |
|------|------|
| **BM25 + Dense Hybrid Search** | BM25Okapi + FAISS 벡터 검색을 RRF(Reciprocal Rank Fusion)로 결합 |
| **HyDE** | 질문 → LLM 가상 문서 생성 → 임베딩 검색으로 의미 공간 정렬 |
| **BGE Reranker** | `BAAI/bge-reranker-v2-m3` Cross-encoder로 상위 30개 재정렬 |
| **메타데이터 enriched 청킹** | 기관명·날짜·조항을 각 청크 prefix에 포함 (256자, overlap 30) |
| **챗봇 UI** | ChatGPT 스타일 React 인터페이스, 마크다운 렌더링, 필터 사이드바 |
| **대화 이력** | SQLite 기반 대화 저장/조회 |

---

## 기술 스택

### Backend
- **Python 3.11**
- **FastAPI** + Uvicorn — REST API 서버
- **LangChain** — RAG 파이프라인
- **FAISS** — 벡터 유사도 검색
- **rank-bm25** — BM25 키워드 검색
- **FlagEmbedding** — BGE Cross-encoder Reranker
- **OpenAI** — `text-embedding-3-large` 임베딩, `gpt-4o` LLM
- **SQLite** (aiosqlite) — 로컬 대화 이력 저장

### Frontend
- **React 19** + TypeScript
- **Vite 8** + TailwindCSS v4
- **react-markdown** — AI 답변 마크다운 렌더링

---

## 프로젝트 구조

```
RAG_eight/
├── rag_system.py              # RAG 핵심 엔진 (검색 + LLM)
├── rag_filters.py             # 필터 전처리 유틸
├── date_normalizer.py         # 날짜 정규화 (YYYY-MM-DD)
├── fss_crawler_*.py           # FSS 웹사이트 크롤러
├── fss_doc_*_parser_*.py      # PDF/HWP 문서 파서
├── requirements.txt           # Python 의존성
│
├── backend/                   # FastAPI 서버
│   ├── main.py                # 앱 진입점, CORS 설정
│   ├── config.py              # 환경변수 (pydantic-settings)
│   ├── database.py            # SQLite 초기화
│   ├── models.py              # DB 모델
│   ├── schemas.py             # Pydantic 스키마
│   ├── rag_service.py         # RAG 시스템 싱글턴 관리
│   └── routers/
│       ├── chat.py            # POST /api/chat
│       └── conversations.py   # GET /api/conversations
│
├── frontend/                  # React 프론트엔드
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── Sidebar.tsx       # 대화 목록 + 필터
│       │   ├── ChatWindow.tsx    # 메시지 영역 + 입력창
│       │   ├── MessageBubble.tsx # 마크다운 렌더링
│       │   ├── FilterBar.tsx     # 문서유형/날짜/기관 필터
│       │   └── SourceCard.tsx    # 참고 문서 카드
│       └── hooks/
│           └── useChat.ts        # 채팅 상태 관리
│
└── data/
    ├── fss_sanctions_parsed.json      # 제재결과 파싱 데이터 (4,320건)
    ├── fss_management_parsed.json     # 경영유의사항 파싱 데이터 (64건)
    └── vector_db/
        ├── fss_sanctions/faiss/       # 제재 벡터 DB + BM25 인덱스
        └── fss_management/faiss/      # 경영유의 벡터 DB + BM25 인덱스
```

---

## 검색 파이프라인

```
사용자 질문
    │
    ▼
[HyDE] LLM → 가상 제재 문서 생성
    │
    ▼
[Hybrid Search] BM25 + FAISS Dense → RRF 합산
    │
    ▼
[Filter] 날짜 / 기관 / 문서유형 메타데이터 필터링
    │
    ▼
[BGE Reranker] 상위 30개 → Cross-encoder 재정렬 → top-k 반환
    │
    ▼
[GPT-4o] 컨텍스트 + 질문 → 최종 답변 생성
```

**점수 공식:** `cosine_similarity = 1 - L²/2` (단위 정규화 벡터 기준)

---

## 시작하기

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- OpenAI API Key

### 1. Python 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...
DATABASE_URL=sqlite+aiosqlite:///./test.db
VECTOR_DB_PATH=data/vector_db/fss_sanctions
VECTOR_DB_TYPE=FAISS
```

### 3. 데이터 크롤링 및 벡터 DB 구축 (최초 1회)

```bash
# FSS 제재결과 크롤링
python fss_crawler_section_eight.py

# FSS 경영유의사항 크롤링
python fss_crawler_management_eight.py

# 벡터 DB 구축 (제재결과)
python -c "
from rag_system import FSSRagSystem
rag = FSSRagSystem(vector_db_path='data/vector_db/fss_sanctions', use_openai_embeddings=True, use_faiss=True)
rag.create_vector_store_from_json('data/fss_sanctions_parsed.json', db_type='sanctions')
"

# 벡터 DB 구축 (경영유의사항)
python -c "
from rag_system import FSSRagSystem
rag = FSSRagSystem(vector_db_path='data/vector_db/fss_management', use_openai_embeddings=True, use_faiss=True)
rag.create_vector_store_from_json('data/fss_management_parsed.json', db_type='management')
"
```

### 4. 백엔드 실행

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 `http://localhost:5174` 접속

---

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/chat` | 질문 전송 및 답변 수신 |
| `GET` | `/api/conversations` | 대화 목록 조회 |
| `GET` | `/api/conversations/{id}/messages` | 특정 대화 메시지 조회 |
| `GET` | `/health` | 헬스 체크 |

### POST /api/chat 예시

```json
{
  "message": "2024년 은행권 제재 현황을 요약해줘",
  "conversation_id": null,
  "filters": {
    "doc_type": "sanctions",
    "date_from": "2024-01-01",
    "date_to": "2024-12-31",
    "institution": null
  }
}
```

---

## 질의 예시

- `2024년 은행권 제재 현황을 요약해줘`
- `전자금융거래법 위반 사례 알려줘`
- `최근 과태료 부과 사례는?`
- `내부통제 관련 경영유의사항은?`
- `신용정보법 위반으로 제재받은 기관은?`
