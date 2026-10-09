# Recall

A RAG (retrieval-augmented generation) study assistant. Upload course notes
or lecture slides as a PDF (or plain text/markdown), and ask questions about
them in a chat interface. Every answer is grounded in the actual uploaded
material and cites exactly which document and page it came from - if the
material doesn't contain the answer, Recall says so instead of guessing.

Live demo: _add the Render URL here once deployed (see DEPLOY.md)._

## Why this exists

Part 3 of a 5-project AI engineering portfolio. The point of this one is a
working, correctly-cited RAG pipeline: chunking, embeddings, vector search,
and an LLM call that's constrained enough that it can't silently fabricate
a source.

## Architecture

```
                     ┌─────────────────────┐
  PDF / .txt / .md → │  Ingestion          │
  upload             │  (ingest.py)         │
                     │  1. extract text/page│
                     │  2. chunk + overlap  │
                     │     (chunking.py)    │
                     │  3. embed each chunk │
                     │     (embeddings.py)  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Postgres + pgvector  │
                     │ documents / chunks   │
                     │ (db.py)              │
                     └──────────┬───────────┘
                                │ cosine_distance() top-k
                                ▼
  user question  →   ┌─────────────────────┐
  (Streamlit chat)   │ Retrieval            │
                     │ (retrieval.py)        │
                     └──────────┬───────────┘
                                │ top-k chunks, numbered [1]..[k]
                                ▼
                     ┌─────────────────────┐
                     │ Grounded generation  │
                     │ (rag.py)              │
                     │ Gemini + schema-      │
                     │ constrained output:   │
                     │ {answer,              │
                     │  used_chunk_indices}  │
                     └──────────┬───────────┘
                                │ server maps indices → real
                                │ document/page metadata
                                ▼
                     answer + verified citations
```

**Backend:** FastAPI (`backend/app/`). **Frontend:** Streamlit chat UI
(`frontend/streamlit_app.py`). **Storage:** Postgres with the `pgvector`
extension (Neon free tier).

## Citation safety: the part that actually matters in a RAG demo

The easy way to build this would be: "hey Gemini, answer the question and
also tell me which page you got it from." That's also the way most RAG demos
quietly lie, because the model has to *recall* a page number correctly, and
LLMs are bad at recalling exact metadata even when they got the content
right.

Recall never asks the model for a page number. The server already knows
exactly which document, page, and text block each retrieved chunk came from
- it built that list itself. The model is only ever asked to say *which of
the numbered passages (1, 2, 3...) it actually used* (`used_chunk_indices` in
`schemas.py`'s `RetrievalAnswer`), via Gemini's schema-constrained JSON
output. The server then looks up the real metadata for those indices. If the
model names an index that doesn't exist, it's silently dropped rather than
shown as a citation - see `rag.py::build_citations`, which is unit-tested
independently of the live API in `tests/test_rag_citations.py`.

The model is also told explicitly to set `answerable_from_context: false` and
say so plainly when the retrieved passages don't contain the answer, rather
than filling the gap with outside knowledge.

## Chunking

Course material is extracted per-page (one string per PDF page via `pypdf`;
a `.txt`/`.md` upload is treated as a single page). `chunking.py` then slides
a fixed-size overlapping window across the concatenated text, tracking which
page(s) each resulting chunk's characters came from - so a chunk that happens
to straddle a page break still gets an honest `"pages 3-4"` label instead of
guessing one page.

**Defaults: 1200 characters per chunk, 200 character overlap.** The
chunk-size experiment in `backend/tests/eval/run_chunk_experiment.py`
compares retrieval hit-rate@k across several (chunk_size, overlap) pairs on
the hand-labeled eval set - see the results below for why 1200/200 was kept
over the alternatives.

### Chunk-size experiment results

_Run `python tests/eval/run_chunk_experiment.py` from `backend/` (with a real
`GEMINI_API_KEY` and `DATABASE_URL` in `backend/.env`) and paste the printed
table here. This needs a live Gemini key and Postgres connection, so it has
to be run by a human - see "Known limitations" below for why._

| chunk_size_chars | overlap_chars | chunks produced | hit-rate@5 |
|---|---|---|---|
| 400 | 50 | — | — |
| 800 | 150 | — | — |
| 1200 (default) | 200 | — | — |
| 2000 | 300 | — | — |

## Embeddings: two interchangeable backends

- **`gemini`** (production default) - Google's `gemini-embedding-001` via the
  official `google-genai` SDK, 768 dimensions, uses `task_type` to optimize
  query vs. document vectors separately. Uses your Gemini API quota.
- **`local`** - `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, runs
  on CPU, free, no API key. Not used in production because the free Render
  tier's 512MB RAM can't comfortably fit torch + the model.

**Switching embedding backends requires re-ingesting every document.**
Vectors from two different models aren't comparable - cosine distance
between a 384-dim MiniLM vector and a 768-dim Gemini vector isn't even
computable, and even at matching dimensions the numbers mean different
things. `backend/app/db.py` fixes the vector column's width
(`EMBEDDING_DIM`) at table-creation time, so changing it on an existing
database means dropping and recreating the `chunks` table before ingesting
anything new.

Local dev and the deployed app are configured to share one Neon database
(see `DEPLOY.md`), which is why both default to `gemini`/768 rather than
`local`/384 - keeping them in sync avoids exactly that mismatch.

## Retrieval evaluation

A hand-labeled set of 11 questions against a fictional 5-page "Introduction
to Databases" lecture-notes PDF (`lecture_notes_databases.pdf`, generated by
`make_sample_notes.py`), each tagged with which source page(s) count as a
correct retrieval. One question has no answer anywhere in the document, to
check that Recall says so instead of guessing (`expect_not_answerable` in
`backend/tests/eval/eval_set.json`).

`backend/tests/eval/run_eval.py` ingests the sample document fresh, retrieves
the top-k chunks for every question, scores hit-rate@k (did any expected page
show up among the retrieved chunks?), and additionally runs each question
through the full chat pipeline to check the final answer and citations.

### Results

_Run `python tests/eval/run_eval.py` from `backend/` and paste the printed
table and summary line here (also written to
`backend/tests/eval/results.json`, which gets committed alongside this
README once it exists)._

```
(pending a live run - see "Known limitations")
```

## Project structure

```
backend/
  app/
    config.py       settings, read fresh from env vars every call
    db.py            SQLAlchemy models (Document, Chunk) + pgvector column
    chunking.py      pure text-chunking logic (fully unit-tested)
    embeddings.py    local + Gemini embedding backends
    ingest.py        PDF/text extraction -> chunk -> embed -> store
    retrieval.py     top-k cosine similarity search
    rag.py           grounded answer generation + citation mapping
    rate_limit.py    in-memory per-IP + global rate limiting
    main.py          FastAPI routes
  tests/             pure-logic unit tests (chunking, config, ingest, rate
                     limiting, citation mapping) - no live API/DB needed
  tests/eval/        hand-labeled retrieval eval + chunk-size experiment
                     (need a live Gemini key + Postgres - see below)
frontend/
  streamlit_app.py   chat UI + document upload/management
make_sample_notes.py generates the demo PDF
render.yaml          Render Blueprint (backend + frontend services)
DEPLOY.md            step-by-step Render deployment guide
```

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # or requirements-local.txt for the "local" embedding backend
cp .env.example .env                  # fill in GEMINI_API_KEY and DATABASE_URL
uvicorn app.main:app --reload
```

```bash
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run streamlit_app.py
```

Run the unit tests (no live API/DB needed):

```bash
cd backend
pip install pytest
pytest tests/ -v
```

## Known limitations

- **Pure-text extraction only.** Scanned/image-only PDFs with no selectable
  text aren't supported (no OCR step) - `ingest.py` raises a clear error
  instead of silently ingesting nothing.
- **In-memory rate limiting.** Resets on every redeploy/restart and doesn't
  share state across multiple instances - fine for one free-tier Render
  service, not a real distributed rate limiter.
- **Citations point to a page, not a highlighted span.** A citation says
  "page 3 of Lecture Notes," not the exact sentence within that page - you
  still have to skim the page yourself to find the exact line.
- **No OCR, no multi-document cross-referencing beyond what fits in top-k.**
  If an answer genuinely needs synthesizing facts from more than
  `RETRIEVAL_TOP_K` (default 5) chunks spread across a document, some of them
  won't make it into context.
- **This assistant (Claude) could not run the live eval or chunk-size
  experiment itself.** Both the Gemini API and the Neon Postgres host are
  blocked by network egress policy from the sandboxed tools used to build
  this project; every piece of pure logic (chunking, config fallbacks, the
  citation-mapping function, the rate limiter) has a unit test that *was*
  run and passes (`pytest tests/ -v` → 29 passed), but anything requiring a
  live LLM or database call needs to be run by a human with real credentials.
