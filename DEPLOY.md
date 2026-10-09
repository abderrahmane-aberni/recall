# Deploying Recall

Two free-tier Render services, same pattern as the CV-Maxxing project: a
Python web service for the FastAPI backend, and a Docker web service for the
Streamlit frontend. Both are defined in `render.yaml` at the repo root, so
Render can create both from one Blueprint.

## 0. Before you deploy: one database, two environments

Local development and the deployed app are configured to point at the **same**
Neon Postgres database (one free Neon project is enough for this - it isn't
worth running two). That only works safely if both environments use the same
embedding backend and dimension, because the `chunks.embedding` column has a
fixed width set when the table is first created. `backend/.env.example` and
`render.yaml` both default to `EMBEDDING_BACKEND=gemini` / `EMBEDDING_DIM=768`
for exactly this reason - don't change one without the other, or ingestion
from whichever side you didn't update will fail.

## 1. Prerequisites already in hand

- A Gemini API key from https://aistudio.google.com/apikey (starts with `AQ.`)
- A Neon Postgres connection string (Neon dashboard → your project → Connect).
  Use the **pooled** connection string (the one with `-pooler` in the hostname)
  - Render's free tier opens a fresh connection per request under load, and
    the pooler handles that far better than a direct connection.

## 2. Create the pgvector extension once

Open the Neon SQL Editor and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

(The backend also runs this itself on startup via `init_db()`, so this step
is really just a sanity check that your Neon plan allows the extension - all
Neon free-tier projects do.)

## 3. Push this repo to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Recall RAG study assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/recall.git
git push -u origin main
```

## 4. Deploy via Render Blueprint

1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect the `recall` GitHub repo. Render reads `render.yaml` and proposes
   two services: `recall-backend` and `recall-frontend`.
3. Before clicking **Apply**, you won't be able to fill in the `sync: false`
   env vars yet - that happens after the services exist (step 5).
4. Click **Apply**. Both services will fail their first deploy until the
   required env vars are set - that's expected.

## 5. Fill in the secret env vars

### `recall-backend` → Environment tab

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your key from aistudio.google.com |
| `DATABASE_URL` | your Neon **pooled** connection string, including `?sslmode=require` |

Everything else (`GEMINI_MODEL`, `EMBEDDING_BACKEND`, `EMBEDDING_DIM`, chunk
size, rate limits) is already set with sane defaults in `render.yaml`.

### `recall-frontend` → Environment tab

| Key | Value |
|---|---|
| `BACKEND_URL` | the backend's `.onrender.com` URL, e.g. `https://recall-backend.onrender.com` (copy it from the backend service's page once it's deployed) |

After saving each env var, Render redeploys that service automatically.

## 6. Verify

- Backend health check: `https://recall-backend.onrender.com/api/health` should
  return `{"status": "ok", "embedding_backend": "gemini"}`.
- Frontend: open `https://recall-frontend.onrender.com`, upload
  `lecture_notes_databases.pdf` (included at the repo root), and ask it a
  question from `backend/tests/eval/eval_set.json`.

## Troubleshooting notes from the first deploy

Both of these are already fixed in this repo's `render.yaml` - documented
here in case a future change to the Blueprint reintroduces either.

- **Backend build failed with a Rust/maturin error compiling
  `pydantic-core`.** Render's default build image had moved to Python 3.14,
  which didn't yet have a prebuilt wheel for `pydantic-core==2.27.2` (pulled
  in by `pydantic==2.10.4`), so pip fell back to compiling it from source -
  which failed because the build sandbox's cargo cache directory is
  read-only. Fixed by pinning `PYTHON_VERSION=3.11.9` as an env var on
  `recall-backend`.
- **Frontend Docker build failed with `"streamlit_app.py": not found`.**
  `dockerContext` was set to the repo root, but `frontend/Dockerfile`'s
  `COPY` commands use bare filenames (`COPY streamlit_app.py .`), which only
  resolve correctly if the build context is `frontend/` itself. Fixed by
  setting `dockerContext: ./frontend`.

## Notes on the free tier

- Both services spin down after ~15 minutes of inactivity and take 30-60
  seconds to wake up on the next request - the first request after a quiet
  period will be slow, not broken.
- Render's free Postgres offering was dropped; Neon's free tier is used
  instead, and it has its own inactivity scale-to-zero behavior with a
  similar cold-start delay on the first query after a while.
- The free web service plan has 512MB RAM, which is why production uses the
  Gemini embedding API instead of loading a local sentence-transformers model
  (see `backend/requirements.txt` for the full reasoning).

## Why Render for both services (not Streamlit Community Cloud)

Same reasoning as CV-Maxxing: Streamlit Community Cloud overlays a
non-removable "created by [your GitHub avatar]" badge on every app it hosts,
confirmed by Streamlit staff to have no workaround. Running the Streamlit
frontend as a Docker service on Render instead avoids it entirely, since the
overlay is a Community Cloud hosting feature, not something baked into the
Streamlit library itself.
