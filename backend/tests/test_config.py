import os

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in [
        "GEMINI_API_KEY",
        "EMBEDDING_BACKEND",
        "DATABASE_URL",
        "EMBEDDING_DIM",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_empty_string_database_url_falls_back_to_sqlite(monkeypatch):
    # Same bug class that broke CV-Maxxing's Render deploy: Render stores a
    # blank dashboard field as "", not a missing var. os.getenv(key, default)
    # does NOT fall back on "" - only on a fully-missing key - so config.py
    # must use `os.getenv(key) or default` instead.
    monkeypatch.setenv("DATABASE_URL", "")
    settings = Settings()
    assert settings.database_url == "sqlite:///./recall.db"


def test_real_database_url_is_used_when_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    settings = Settings()
    assert settings.database_url == "postgresql://user:pass@host/db"


def test_validate_requires_gemini_key_even_for_local_embeddings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    settings = Settings()
    with pytest.raises(RuntimeError):
        settings.validate()


def test_validate_requires_gemini_key_for_gemini_embeddings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "gemini")
    settings = Settings()
    with pytest.raises(RuntimeError):
        settings.validate()


def test_validate_passes_with_key_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    settings = Settings()
    settings.validate()  # should not raise


def test_invalid_embedding_backend_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "nonsense")
    settings = Settings()
    with pytest.raises(RuntimeError):
        settings.validate()
