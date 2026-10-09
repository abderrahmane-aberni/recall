# Recall

A RAG (retrieval-augmented generation) study assistant. Upload course notes
or lecture slides as a PDF (or plain text/markdown), and ask questions about
them in a chat interface. Every answer is grounded in the actual uploaded
material and cites exactly which document and page it came from - if the
material doesn't contain the answer, Recall says so instead of guessing.

Live demo: [recall-frontend-3yor.onrender.com](https://recall-frontend-3yor.onrender.com)
(backend: [recall-backend-mmgs.onrender.com](https://recall-backend-mmgs.onrender.com/api/health))

Both are free-tier Render services, so the first request after a quiet
period takes 30-60 seconds to wake up - that's expected, not broken.

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
the hand-labeled eval set.

### Chunk-size experiment results

| chunk_size_chars | overlap_chars | chunks produced | hit-rate@5 |
|---|---|---|---|
| 400 | 50 | 12 | 100% |
| 800 | 150 | 6 | 100% |
| 1200 (default) | 200 | 4 | 100% |
| 2000 | 300 | 3 | 100% |

**Honest reading of this result:** every configuration scored 100%, which
doesn't actually distinguish between them - and the reason is informative in
its own right. The demo document is only 5 pages, so even the largest chunk
size produces just 3 chunks, and `RETRIEVAL_TOP_K=5` retrieves effectively
the whole document regardless of how it's split. This experiment would only
discriminate between chunk sizes on a document long enough that top-k
retrieval misses chunks at some sizes but not others (tens of pages, not
five). 1200/200 is kept as the default anyway because it's a reasonable
middle ground for typical lecture-note-length documents - not because this
particular run proved it beats the alternatives.

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

Run with `EMBEDDING_BACKEND=gemini`, `CHUNK_SIZE_CHARS=1200`,
`CHUNK_OVERLAP_CHARS=200`, `RETRIEVAL_TOP_K=5` (full output in
`backend/tests/eval/results.json`):

| ID | Hit | Expected page | Retrieved pages | Citations |
|---|---|---|---|---|
| acid_definition | yes | [1] | [1,2,3,4,5] | 1 |
| durability_definition | yes | [1] | [1,2,3,4,5] | 1 |
| default_isolation_postgres | yes | [2] | [1,2,3,4,5] | 2 |
| dirty_reads | yes | [2] | [1,2,3,4,5] | 1 |
| btree_vs_hash | yes | [3] | [1,2,3,4,5] | 2 |
| default_index_type | yes | [3] | [1,2,3,4,5] | 1 |
| third_normal_form | yes | [4] | [1,2,3,4,5] | 1 |
| 1nf_example | yes | [4] | [1,2,3,4,5] | 1 |
| cap_tradeoff | yes | [5] | [1,2,3,4,5] | 1 |
| single_node_cap | yes | [5] | [1,2,3,4,5] | 1 |
| not_in_document | n/a | — | [1,2,3,4,5] | 0 |

**Hit-rate@5: 100% (threshold 80%) → PASS.**

Every answer cited the correct page(s) for its question - see the sample
answers below. The one question with no answer in the document
(`not_in_document`, asking about a sharding strategy the lecture never
covers) correctly came back with zero citations and this answer: *"The
provided context passages do not contain information about sharding
strategies for a 10-node cluster."* - rather than a confident guess, which is
exactly what the `answerable_from_context` schema field (see "Citation
safety" above) is there to prevent.

Two representative answers, citations included:

> **Q: What is the default transaction isolation level in PostgreSQL?**
> The default isolation level in PostgreSQL is Read Committed [1][2].

> **Q: Why can't a hash index be used for a range query like age BETWEEN 20 AND 30?**
> A hash index cannot be used for a range query because hashing destroys the
> original ordering of the values [1][2].

As with the chunk-size experiment, a perfect hit-rate on a 5-page document
with top-k=5 is partly a function of how little there is to miss - this
confirms the retrieval and citation-mapping code paths work correctly
end-to-end, not that retrieval is bulletproof on a large corpus.

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
- **The small demo document limits what the eval can prove.** Both the
  retrieval eval and the chunk-size experiment hit 100% on a 5-page document
  with `RETRIEVAL_TOP_K=5`, which mostly reflects that there's little for
  top-k retrieval to miss at that size - see the honest caveats in the
  "Chunking" and "Retrieval evaluation" sections above. A longer, messier
  real-world document set would be a better test of where this actually
  breaks.
- **Both the Gemini API and the Neon Postgres host were unreachable from
  Claude's own sandboxed tools while building this project** (blocked by
  network egress policy), so the live eval and chunk-size experiment had to
  be run by a human with real credentials rather than by the assistant. Every
  piece of pure logic (chunking, config fallbacks, the citation-mapping
  function, the rate limiter) has a unit test that *was* run by the
  assistant and passes (`pytest tests/ -v` → 29 passed).
