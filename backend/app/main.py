from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Document, get_db, init_db
from .embeddings import get_embedding_backend
from .ingest import EmptyDocument, UnsupportedFileType, ingest_document
from .rag import generate_answer
from .rate_limit import RateLimitExceeded, RateLimiter
from .retrieval import retrieve
from .schemas import ChatRequest, ChatResponse, DocumentOut, IngestResponse

app = FastAPI(title="Recall", description="RAG study assistant over your own course notes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = get_settings()
_rate_limiter = RateLimiter(
    per_ip_per_hour=_settings.rate_limit_per_ip_per_hour,
    global_per_day=_settings.rate_limit_global_per_day,
)
# Built once at startup - whichever backend is configured is the one that must
# be used for every ingestion and every query for as long as this process runs.
_embedding_backend = get_embedding_backend(_settings)


@app.on_event("startup")
def on_startup() -> None:
    _settings.validate()
    init_db()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "embedding_backend": _embedding_backend.name}


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@app.post("/api/documents", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
) -> IngestResponse:
    settings = get_settings()
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB, which exceeds the {settings.max_upload_mb} MB limit.",
        )

    try:
        document = ingest_document(
            db,
            filename=file.filename or "upload",
            title=title or (file.filename or "Untitled document"),
            data=data,
            settings=settings,
            embedding_backend=_embedding_backend,
        )
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(
        document=DocumentOut.model_validate(document),
        message=f"Ingested {document.num_chunks} chunks from {document.filename}.",
    )


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()
    return {"status": "deleted", "document_id": document_id}


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ChatResponse:
    settings = get_settings()
    try:
        _rate_limiter.check(_client_ip(request))
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    retrieved = retrieve(
        db,
        body.message,
        _embedding_backend,
        top_k=settings.retrieval_top_k,
        document_ids=body.document_ids,
    )

    if not retrieved:
        return ChatResponse(
            answer=(
                "There's nothing to search yet - upload a document first, or check that the "
                "documents you selected still exist."
            ),
            citations=[],
            retrieved_chunk_count=0,
        )

    answer, citations = generate_answer(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        question=body.message,
        retrieved=retrieved,
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        retrieved_chunk_count=len(retrieved),
    )
