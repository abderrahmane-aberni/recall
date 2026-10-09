"""Database models and session handling.

Lesson learned on the CV-Maxxing project: never call create_engine() (or
anything else that touches DATABASE_URL) at module import time with a value
that might be an empty string - Render stores a dashboard field left blank
as "", not unset, and SQLAlchemy can't parse that. Everything that needs a
real connection here goes through get_engine()/get_session(), which are only
called once a request actually needs the DB, by which point config.validate()
has already run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import get_settings

_settings = get_settings()

# Fixed at import time because it defines the vector column's width - this is a
# schema decision, not a live connection, so it's safe to read eagerly. Switching
# EMBEDDING_DIM later requires dropping and re-ingesting (see README).
EMBEDDING_DIM = _settings.embedding_dim


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    embedding_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Best-effort source locator shown in citations, e.g. "page 3" or
    # "section 2". Extraction is page-based for PDFs; see chunking.py.
    source_label: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def init_db() -> None:
    """Create the pgvector extension (if missing) and all tables.

    Safe to call on every backend startup - CREATE EXTENSION IF NOT EXISTS and
    create_all() are both idempotent.
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session, always closes it."""
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
