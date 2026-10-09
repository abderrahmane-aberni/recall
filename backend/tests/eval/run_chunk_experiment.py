"""Chunk-size experiment: re-ingests the sample document under several
(chunk_size, overlap) configurations and reports retrieval hit-rate@k for
each, to justify the CHUNK_SIZE_CHARS/CHUNK_OVERLAP_CHARS defaults chosen in
config.py. Retrieval-only (no Gemini chat calls) to keep this fast and free
of API quota usage - the embedding calls still use whichever EMBEDDING_BACKEND
is configured.

Must be run by a human with a real GEMINI_API_KEY (if EMBEDDING_BACKEND=gemini)
and a real pgvector DATABASE_URL - see run_eval.py for why.

Usage (from backend/):
    python tests/eval/run_chunk_experiment.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/

from app.config import get_settings  # noqa: E402
from app.db import Document, get_sessionmaker, init_db  # noqa: E402
from app.embeddings import get_embedding_backend  # noqa: E402
from app.ingest import ingest_document  # noqa: E402
from app.retrieval import retrieve  # noqa: E402
from sqlalchemy import select  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
SAMPLE_PDF = EVAL_DIR.parent.parent.parent / "lecture_notes_databases.pdf"

# (chunk_size_chars, chunk_overlap_chars) pairs to compare.
CONFIGS = [
    (400, 50),
    (800, 150),
    (1200, 200),  # the default
    (2000, 300),
]


def _parse_pages(source_label: str) -> set[int]:
    nums = [int(n) for n in re.findall(r"\d+", source_label)]
    if len(nums) == 1:
        return {nums[0]}
    if len(nums) == 2:
        return set(range(nums[0], nums[1] + 1))
    return set()


def main() -> None:
    base_settings = get_settings()
    base_settings.validate()
    init_db()

    if not SAMPLE_PDF.exists():
        raise SystemExit(
            f"Sample PDF not found at {SAMPLE_PDF}. Run make_sample_notes.py first."
        )

    eval_set = json.loads((EVAL_DIR / "eval_set.json").read_text())
    questions = [q for q in eval_set["questions"] if not q.get("expect_not_answerable")]

    SessionLocal = get_sessionmaker()
    embedding_backend = get_embedding_backend(base_settings)

    experiment_results = []

    for chunk_size, overlap in CONFIGS:
        # Settings is a plain class (not frozen/dataclass) - mutate the two
        # fields that matter for this experiment directly on the shared
        # instance rather than constructing a new Settings() (which would
        # re-read env vars and lose nothing here, but this is more explicit).
        settings = base_settings
        settings.chunk_size_chars = chunk_size
        settings.chunk_overlap_chars = overlap

        db = SessionLocal()
        try:
            existing = db.scalars(
                select(Document).where(Document.filename == SAMPLE_PDF.name)
            ).all()
            for doc in existing:
                db.delete(doc)
            db.commit()

            document = ingest_document(
                db,
                filename=SAMPLE_PDF.name,
                title="Introduction to Databases - Lecture Notes",
                data=SAMPLE_PDF.read_bytes(),
                settings=settings,
                embedding_backend=embedding_backend,
            )

            hits = 0
            for q in questions:
                retrieved = retrieve(
                    db, q["question"], embedding_backend, top_k=settings.retrieval_top_k
                )
                retrieved_pages: set[int] = set()
                for chunk in retrieved:
                    retrieved_pages |= _parse_pages(chunk.source_label)
                if set(q["expected_pages"]) & retrieved_pages:
                    hits += 1

            hit_rate = hits / len(questions) if questions else 0.0
            experiment_results.append(
                {
                    "chunk_size_chars": chunk_size,
                    "chunk_overlap_chars": overlap,
                    "num_chunks_produced": document.num_chunks,
                    "hit_rate_at_k": round(hit_rate, 3),
                }
            )
            print(
                f"chunk_size={chunk_size:<6} overlap={overlap:<5} "
                f"chunks={document.num_chunks:<4} hit_rate@{settings.retrieval_top_k}="
                f"{hit_rate:.1%}"
            )
        finally:
            db.close()

    # leave the DB in the default (last-run) config's state rather than some
    # intermediate experimental one
    db = SessionLocal()
    try:
        existing = db.scalars(
            select(Document).where(Document.filename == SAMPLE_PDF.name)
        ).all()
        for doc in existing:
            db.delete(doc)
        db.commit()
        base_settings.chunk_size_chars, base_settings.chunk_overlap_chars = 1200, 200
        ingest_document(
            db,
            filename=SAMPLE_PDF.name,
            title="Introduction to Databases - Lecture Notes",
            data=SAMPLE_PDF.read_bytes(),
            settings=base_settings,
            embedding_backend=embedding_backend,
        )
    finally:
        db.close()

    (EVAL_DIR / "chunk_experiment_results.json").write_text(
        json.dumps(experiment_results, indent=2)
    )
    print(f"\nWrote {EVAL_DIR / 'chunk_experiment_results.json'}")
    print("Re-ingested the sample doc with the default config (1200/200) afterwards.")


if __name__ == "__main__":
    main()
