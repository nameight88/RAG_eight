"""FastAPI 앱 진입점."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.database import init_db
from backend.routers import chat, conversations


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    # RAG 서비스 초기화는 OPENAI_API_KEY가 있을 때만
    if settings.openai_api_key:
        from backend.rag_service import init_rag_service
        try:
            init_rag_service(
                vector_db_path=settings.vector_db_path,
                use_faiss=(settings.vector_db_type == "FAISS"),
            )
        except Exception as e:
            print(f"⚠️ RAG 서비스 초기화 실패 (계속 실행): {e}")
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
