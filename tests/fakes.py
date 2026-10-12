"""Test doubles (PROVIDED).

These stand in for the real model, database and gateway so each milestone can
be tested in isolation. They are deliberately simple and are NOT reference
implementations of the lab's components:

* ``HashingEmbedder`` only counts hashed words – it has no notion of meaning.
* ``RecordingStore`` stores what it is given and returns canned search results;
  it does no similarity search or filtering.
* ``StaticResponseProvider`` returns a fixed string.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from rag_lab.vector_store import SearchFilters, SearchResult


class HashingEmbedder:
    """Deterministic fake embedder: hashed bag of words, L2-normalised."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 384, model_name: str | None = None) -> None:
        self._dimension = dimension
        self._model_name = model_name or f"fake-hashing-{dimension}"
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in self._TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._embed(text)


class RecordingStore:
    """In-memory stand-in for ``PgVectorStore``.

    ``lose_rows`` simulates rows going missing during persistence.
    ``search_results`` is returned (truncated to ``top_k``) by :meth:`search`.
    """

    def __init__(self, lose_rows: int = 0, search_results: list[SearchResult] | None = None):
        self.lose_rows = lose_rows
        self.search_results = search_results or []
        self.tables: dict[str, list] = {}
        self.models: dict[str, set[str]] = {}
        self.search_calls: list[tuple[str, list[float], int, SearchFilters | None]] = []

    def replace_chunks(self, strategy, chunks, embeddings, embedding_model) -> int:  # type: ignore[no-untyped-def]
        self.tables[strategy] = list(zip(chunks, embeddings, strict=True))
        self.models[strategy] = {embedding_model}
        return len(chunks)

    def count(self, strategy: str) -> int:
        return max(len(self.tables.get(strategy, [])) - self.lose_rows, 0)

    def chunk_ids(self, strategy: str) -> list[str]:
        return sorted(c.chunk_id for c, _ in self.tables.get(strategy, []))

    def embedding_models(self, strategy: str) -> set[str]:
        return set(self.models.get(strategy, set()))

    def search(self, strategy, query_embedding, top_k, filters=None) -> list[SearchResult]:  # type: ignore[no-untyped-def]
        self.search_calls.append((strategy, list(query_embedding), top_k, filters))
        return list(self.search_results[:top_k])


class StaticResponseProvider:
    """Fake LLM provider that returns ``response`` and records every call."""

    name = "static"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


class FakeRetriever:
    """Returns canned results and records the filters it was called with."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, SearchFilters | None]] = []

    def retrieve(self, question: str, filters: SearchFilters | None = None) -> list[SearchResult]:
        self.calls.append((question, filters))
        return list(self.results)


def make_result(
    i: int,
    source: str,
    heading: str,
    text: str,
    score: float = 0.8,
    effective_date: str = "2026-04-01",
) -> SearchResult:
    return SearchResult(
        chunk_id=f"doc_test:structured:{i:04d}",
        text=text,
        score=score,
        metadata={
            "source": source,
            "doc_id": "doc_test",
            "title": "Test",
            "doc_type": "policy",
            "effective_date": effective_date,
            "section_heading": heading,
            "department": "Human Resources",
            "location": "All Locations",
        },
    )
