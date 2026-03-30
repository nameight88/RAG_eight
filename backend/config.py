"""환경변수 설정 모듈."""
import os
from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트 (.env 위치)
_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = str(_ROOT / ".env")


class Settings(BaseSettings):
    openai_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./test.db"
    vector_db_path: str = "data/vector_db/fss_sanctions"
    vector_db_type: str = "FAISS"
    allowed_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
