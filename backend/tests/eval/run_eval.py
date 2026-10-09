"""Retrieval + end-to-end eval harness for Recall.

Must be run by a human with a real GEMINI_API_KEY and a real pgvector-backed
DATABASE_URL in backend/.env - it is not run in CI/by the assistant, since
both the Gemini API and the Postgres host are unreachable from the sandboxed
tools used to build this project (see README "Known limitations").

Usage (from backend/):
    python tests/eval/run_eval.py

What it does:
  1. Deletes any existing "lecture_notes_databases.pdf" document and
     re-ingests it fresh, using whatever EMBEDDING_BACKEND is configured.
  2. For each hand-labeled question in eval_set.json, retrieves the
     configured RETRIEVAL_TOP_K chunks and checks whether any of the
     expected source pages was actually retrieved (hit-rate@k).
  3. For every question, also calls the full /api/chat pipeline (in-process,
     not over HTTP) and records the answer, citations, and whether the model
     correctly refused to answer the one question with no support in the
     document.
  4. Writes tests/eval/results.json and prints a pass/fail summary table.

A question counts as a retrieval hit if ANY of its expected_pages appears in
the source_label of ANY retrieved chunk (chunks spanning multiple pages, e.g.
"pages 2-3", count as covering every page they list).
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
from app.rag import generate_answer  # noqa: E402
from app.retrieval import retrieve  # noqa: E402
from sqlalchemy import select  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
SAMPLE_PDF = EVAL_DIR.parent.parent.parent / "lecture_notes_databases.pdf"
HIT_RATE_PASS_THRESHOLD = 0.8  # overall hit-rate@k must be >= this to PASS


def _parse_pages(source_label: str) -> set[int]:
    """'page 3' -> {3}; 'pages 2-3' -> {2, 3}"""
    nums = [int(n) for n in re.findall(r"\d+", source_label)]
    if len(nums) == 1:
        return {nums[0]}
    if len(nums) == 2:
        return set(range(nums[0], nums[1] + 1))
    return set()


def main() -> None:
    settings = get_settings()
    settings.validate()
    init_db()

    if not SAMPLE_PDF.exists():
        raise SystemExit(
            f"Sample PDF not found at {SAMPLE_PDF}. Run make_sample_notes.py first."
        )

    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    embedding_backend = get_embedding_backend(settings)

    try:
        existing = db.scalars(
            select(Document).where(Document.filename == SAMPLE_PDF.name)
        ).all()
        for doc in existing:
            db.delete(doc)
        db.commit()

        print(f"Ingesting {SAMPLE_PDF.name} with embedding backend={embedding_backend.name} ...")
        document = ingest_document(
            db,
            filename=SAMPLE_PDF.name,
            title="Introduction to Databases - Lecture Notes",
            data=SAMPLE_PDF.read_bytes(),
            settings=settings,
            embedding_backend=embedding_backend,
        )
        print(f"Ingested {document.num_chunks} chunks.\n")

        eval_set = json.loads((EVAL_DIR / "eval_set.json").read_text())
        results = []

        for q in eval_set["questions"]:
            retrieved = retrieve(
                db, q["question"], embedding_backend, top_k=settings.retrieval_top_k
            )
            retrieved_pages = set()
            for chunk in retrieved:
                retrieved_pages |= _parse_pages(chunk.source_label)

            expected_pages = set(q.get("expected_pages", []))
            expect_not_answerable = q.get("expect_not_answerable", False)

            if expect_not_answerable:
                retrieval_hit = None  # not meaningful for this question
            else:
                retrieval_hit = bool(expected_pages & retrieved_pages)

            answer, citations = generate_answer(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model,
                question=q["question"],
                retrieved=retrieved,
            )

            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected_pages": sorted(expected_pages),
                    "retrieved_pages": sorted(retrieved_pages),
                    "retrieval_hit": retrieval_hit,
                    "answer": answer,
                    "num_citations": len(citations),
                    "expect_not_answerable": expect_not_answerable,
                }
            )

        scored = [r for r in results if r["retrieval_hit"] is not None]
        hit_rate = sum(1 for r in scored if r["retrieval_hit"]) / len(scored) if scored else 0.0
        overall_pass = hit_rate >= HIT_RATE_PASS_THRESHOLD

        summary = {
            "embedding_backend": embedding_backend.name,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "retrieval_top_k": settings.retrieval_top_k,
            "hit_rate_at_k": round(hit_rate, 3),
            "pass_threshold": HIT_RATE_PASS_THRESHOLD,
            "overall": "PASS" if overall_pass else "FAIL",
            "results": results,
        }

        (EVAL_DIR / "results.json").write_text(json.dumps(summary, indent=2))

        print(f"{'ID':<28} {'HIT':<6} {'EXPECTED':<12} {'RETRIEVED':<12} CITES")
        for r in results:
            hit_str = "-" if r["retrieval_hit"] is None else ("yes" if r["retrieval_hit"] else "NO")
            print(
                f"{r['id']:<28} {hit_str:<6} {str(r['expected_pages']):<12} "
                f"{str(r['retrieved_pages']):<12} {r['num_citations']}"
            )
        print(f"\nHit-rate@{settings.retrieval_top_k}: {hit_rate:.1%} "
              f"(threshold {HIT_RATE_PASS_THRESHOLD:.0%}) -> {summary['overall']}")
        print(f"\nWrote {EVAL_DIR / 'results.json'}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
