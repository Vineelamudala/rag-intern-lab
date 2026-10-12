"""Milestone M7 – embed the question and retrieve chunks from one strategy's table.

Retrieval needs no LLM, so it can be built, tested and evaluated before you
touch the gateway.

PROVIDED: the ``Retriever`` configuration and validation.
YOUR WORK: :meth:`Retriever.retrieve`.

Tests: ``python -m pytest -m m7``
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_lab import STRATEGIES, todo
from rag_lab.embeddings import Embedder
from rag_lab.vector_store import SearchFilters, SearchResult, VectorStore


class RetrievalError(RuntimeError):
    """Raised for invalid retrieval requests or a model/index mismatch."""


@dataclass
class Retriever:
    """Embeds a question and returns the most similar chunks above ``min_similarity``."""

    embedder: Embedder
    store: VectorStore
    strategy: str = "structured"
    top_k: int = 4
    min_similarity: float = 0.30

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise RetrievalError(
                f"Unknown strategy {self.strategy!r}; expected one of {STRATEGIES}"
            )
        if self.top_k <= 0:
            raise RetrievalError("top_k must be greater than 0")

    def retrieve(self, question: str, filters: SearchFilters | None = None) -> list[SearchResult]:
        """Return up to ``top_k`` chunks relevant to ``question``.

        Requirements:
        * Raise :class:`RetrievalError` for an empty or whitespace-only question.
        * Refuse to compare vectors from different models: if
          ``store.embedding_models(strategy)`` is non-empty and differs from
          ``{embedder.model_name}``, raise :class:`RetrievalError` explaining that
          ingestion must be re-run.
        * Embed the question with ``embedder.embed_query``.
        * Call ``store.search(strategy, vector, top_k, filters)``.
        * Drop results whose ``score`` is below ``min_similarity`` and keep the
          rest in the order returned (highest score first).
        """
        raise todo("M7", "Retriever.retrieve")
