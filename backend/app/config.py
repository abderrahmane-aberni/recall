"""Application configuration, read fresh from environment variables on every
call (no caching) so tests can monkeypatch env vars per-test without fighting
stale state. Mirrors the pattern used in the CV-Maxxing project.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        # --- LLM (chat generation) ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

        # --- Embeddings ---
        # "local"  -> sentence-transformers, runs on CPU, no API key needed, 384 dims
        # "gemini" -> Google's gemini-embedding-001, needs GEMINI_API_KEY, configurable dims
        self.embedding_backend = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
        self.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        # Dimensionality actually stored in the vector column. Must match whichever
        # backend produced the vectors currently in the DB - switching backends
        # requires re-ingesting all documents (see README "Switching embedding backends").
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))

        # --- Database ---
        # `or`, not a .getenv default: an empty-string env var (what Render stores
        # when a dashboard field is left blank) needs to fall back too, not just a
        # fully-missing variable. SQLite has no pgvector support, so a local
        # fallback here is only useful for import-time sanity checks, never for
        # actually running ingestion/retrieval.
        self.database_url = os.getenv("DATABASE_URL") or "sqlite:///./recall.db"

        # --- Chunking ---
        self.chunk_size_chars = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
        self.chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

        # --- Retrieval ---
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))

        # --- Uploads ---
        self.max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "200"))

        # --- Rate limiting (protects the free-tier Gemini quota on a public demo) ---
        self.rate_limit_per_ip_per_hour = int(os.getenv("RATE_LIMIT_PER_IP_PER_HOUR", "20"))
        self.rate_limit_global_per_day = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "200"))

    def validate(self) -> None:
        if self.embedding_backend not in ("local", "gemini"):
            raise RuntimeError(
                f"EMBEDDING_BACKEND must be 'local' or 'gemini', got {self.embedding_backend!r}"
            )
        if self.embedding_backend == "gemini" and not self.gemini_api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND=gemini requires GEMINI_API_KEY. Copy backend/.env.example "
                "to backend/.env and add your key from https://aistudio.google.com/apikey"
            )
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (needed for chat answer generation even if "
                "embeddings run locally). Copy backend/.env.example to backend/.env and "
                "add your key from https://aistudio.google.com/apikey"
            )


def get_settings() -> Settings:
    return Settings()
