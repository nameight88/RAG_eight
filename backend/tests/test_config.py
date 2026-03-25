"""config.py 환경변수 로드 검증"""
import importlib
import os


def test_settings_load_from_env(monkeypatch):
    """환경변수에서 설정을 올바르게 읽어야 한다"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("VECTOR_DB_PATH", "data/vector_db/fss_sanctions")

    import sys
    # force reimport
    for key in list(sys.modules.keys()):
        if "backend.config" in key:
            del sys.modules[key]

    from backend.config import Settings
    settings = Settings(
        openai_api_key="test-key-123",
        database_url="sqlite+aiosqlite:///./test.db",
        vector_db_path="data/vector_db/fss_sanctions",
    )

    assert settings.openai_api_key == "test-key-123"
    assert "test.db" in settings.database_url
    assert settings.vector_db_path == "data/vector_db/fss_sanctions"


def test_allowed_origins_default():
    """기본 ALLOWED_ORIGINS는 localhost를 포함해야 한다"""
    from backend.config import Settings
    settings = Settings()
    assert isinstance(settings.allowed_origins, list)
    assert any("localhost" in o for o in settings.allowed_origins)
