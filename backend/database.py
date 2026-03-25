"""SQLAlchemy 비동기 DB 연결 및 세션 관리."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from backend.config import get_settings


class Base(DeclarativeBase):
    pass


def _create_engine_and_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


engine, AsyncSessionLocal = _create_engine_and_session()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """테이블 생성 (앱 시작 시 호출)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
