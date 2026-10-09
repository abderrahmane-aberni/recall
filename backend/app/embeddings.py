"""Two interchangeable embedding backends:

- "local":  sentence-transformers all-MiniLM-L6-v2, runs on CPU, free, no
  API key, fixed 384 dimensions. Good default for a public demo since it
  doesn't touch anyone's API quota.
- "gemini": Google's gemini-embedding-001 via the official google-genai SDK,
  configurable output dimensionality, uses task_type to optimise query vs.
  document vectors separately (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT).

Whichever backend produced the vectors in the DB is the one that must be used
for queries against them - mixing is silent garbage (cosine distance between
vectors from two different models is meaningless). See README "Switching
embedding backends" for the re-ingestion requirement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from .config import Settings, get_settings

TaskType = Literal["query", "document"]


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str], task: TaskType) -> list[list[float]]:
        """Return one embedding vector per input text."""


class LocalEmbeddingBackend(EmbeddingBackend):
    """sentence-transformers, loaded once and reused (model load is the slow part)."""

    name = "local"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # lazy: avoid paying the load cost at import time

    def _get_model(self):
        if self._model is None:
            # Imported lazily too - keeps `import app.embeddings` cheap for code
            # paths (e.g. tests) that only need the Gemini backend.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str], task: TaskType) -> list[list[float]]:
        model = self._get_model()
        # normalize_embeddings=True makes cosine distance (what pgvector's
        # cosine_distance() computes) behave consistently.
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


class GeminiEmbeddingBackend(EmbeddingBackend):
    """Google gemini-embedding-001 via the official google-genai SDK."""

    name = "gemini"

    _TASK_TYPE_MAP = {
        "query": "RETRIEVAL_QUERY",
        "document": "RETRIEVAL_DOCUMENT",
    }

    def __init__(self, api_key: str, model_name: str, output_dim: int) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._output_dim = output_dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str], task: TaskType) -> list[list[float]]:
        from google.genai import types

        client = self._get_client()
        config = types.EmbedContentConfig(
            task_type=self._TASK_TYPE_MAP[task],
            output_dimensionality=self._output_dim,
        )
        result = client.models.embed_content(
            model=self._model_name, contents=texts, config=config
        )
        return [e.values for e in result.embeddings]


def get_embedding_backend(settings: Settings | None = None) -> EmbeddingBackend:
    settings = settings or get_settings()
    if settings.embedding_backend == "local":
        return LocalEmbeddingBackend(settings.local_embedding_model)
    if settings.embedding_backend == "gemini":
        return GeminiEmbeddingBackend(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embedding_model,
            output_dim=settings.embedding_dim,
        )
    raise ValueError(f"Unknown embedding backend: {settings.embedding_backend!r}")
