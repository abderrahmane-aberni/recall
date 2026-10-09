"""Recall - RAG study assistant frontend.

UX follows the same rules the CV-Maxxing frontend settled on: no action
button is ever disabled - every button is always clickable, and missing
input is reported with a plain message after the click rather than by
greying the button out.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Recall", page_icon="📚", layout="wide")


def _get_backend_url() -> str:
    # Deployed on Render via env vars, not Streamlit Community Cloud, so
    # there's deliberately no st.secrets lookup here - touching st.secrets
    # with no secrets.toml present makes Streamlit show a "No secrets found"
    # banner on every page load, which is confusing for visitors even though
    # it's harmless. BACKEND_URL is always an env var in every environment
    # this app actually runs in (local dev, Render).
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


BACKEND_URL = _get_backend_url()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations}
if "documents" not in st.session_state:
    st.session_state.documents = None  # loaded lazily, refreshed after upload/delete


def _load_documents() -> list[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/documents", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(f"Couldn't load documents: {exc}")
        return []


if st.session_state.documents is None:
    st.session_state.documents = _load_documents()


# ---------------------------------------------------------------------------
# Sidebar: document library
# ---------------------------------------------------------------------------

st.sidebar.title("📚 Recall")
st.sidebar.caption("A study assistant that only answers from documents you upload.")

st.sidebar.subheader("Upload a document")
uploaded_file = st.sidebar.file_uploader(
    "Course notes or lecture slides (PDF, .txt, .md)", type=["pdf", "txt", "md"]
)
doc_title = st.sidebar.text_input("Title (optional)", placeholder="e.g. Week 3 - Cell Biology")

upload_clicked = st.sidebar.button("Ingest document", type="primary")

if upload_clicked:
    if uploaded_file is None:
        st.sidebar.warning("Choose a file to upload first.")
    else:
        with st.sidebar.status("Ingesting...", expanded=False):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"title": doc_title} if doc_title.strip() else {}
                resp = requests.post(
                    f"{BACKEND_URL}/api/documents", files=files, data=data, timeout=120
                )
                if resp.status_code == 200:
                    st.sidebar.success(resp.json().get("message", "Ingested."))
                    st.session_state.documents = _load_documents()
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.sidebar.error(f"Ingestion failed: {detail}")
            except Exception as exc:
                st.sidebar.error(f"Couldn't reach the backend: {exc}")

st.sidebar.divider()
st.sidebar.subheader("Your documents")

if not st.session_state.documents:
    st.sidebar.caption("No documents yet - upload one above to get started.")
else:
    for doc in st.session_state.documents:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"**{doc['title']}**  \n{doc['num_chunks']} chunks · {doc['embedding_backend']}")
        if col2.button("🗑️", key=f"del_{doc['id']}", help="Delete this document"):
            try:
                requests.delete(f"{BACKEND_URL}/api/documents/{doc['id']}", timeout=15)
                st.session_state.documents = _load_documents()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Couldn't delete: {exc}")


# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------

st.title("Ask about your course material")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        citations = msg.get("citations") or []
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for c in citations:
                    st.markdown(
                        f"**[{c['marker']}] {c['document_title']}** — {c['source_label']}"
                    )
                    st.caption(c["snippet"])

question = st.chat_input("Ask a question about your uploaded documents")

if question is not None:
    if not question.strip():
        st.warning("Type a question before sending.")
    elif not st.session_state.documents:
        st.warning("Upload a document first - there's nothing to search yet.")
    else:
        st.session_state.messages.append({"role": "user", "content": question, "citations": []})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/chat", json={"message": question}, timeout=60
                )
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result["answer"]
                    citations = result.get("citations", [])
                    placeholder.markdown(answer)
                    if citations:
                        with st.expander(f"Sources ({len(citations)})"):
                            for c in citations:
                                st.markdown(
                                    f"**[{c['marker']}] {c['document_title']}** — {c['source_label']}"
                                )
                                st.caption(c["snippet"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "citations": citations}
                    )
                elif resp.status_code == 429:
                    detail = resp.json().get("detail", "Rate limit hit.")
                    placeholder.warning(detail)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": detail, "citations": []}
                    )
                else:
                    detail = resp.json().get("detail", resp.text)
                    placeholder.error(f"Something went wrong: {detail}")
            except Exception as exc:
                placeholder.error(f"Couldn't reach the backend: {exc}")
