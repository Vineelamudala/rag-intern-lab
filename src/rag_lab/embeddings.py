"""Milestone M5 – local embeddings with Sentence Transformers.

The lab uses ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions) running
locally. No API key is involved, and the LiteLLM gateway is NOT used for embeddings.

PROVIDED: the ``Embedder`` protocol, error class and factory.
YOUR WORK: every method/function that raises ``NotImplementedError``.

Tests: ``python -m pytest -m m5`` (the first run downloads the model, ~90 MB)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_lab import todo
from rag_lab.config import Settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding model cannot be loaded or returns bad vectors."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-length vectors.

    The unit tests use a fake implementation of this protocol, so the rest of
    the pipeline can be tested without downloading a model.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model implementing :class:`Embedder`."""

    def __init__(self, model_name: str, expected_dimension: int, batch_size: int = 32) -> None:
        """Load the model and check its output dimension.

        Requirements:
        * Import ``sentence_transformers`` inside this method; if it is not
          installed raise :class:`EmbeddingError` telling the user to run
          ``pip install -r requirements.txt``.
        * If loading fails (bad name, no network on first download) raise
          :class:`EmbeddingError` that includes the model name.
        * If the model's embedding dimension differs from ``expected_dimension``
          raise :class:`rag_lab.config.ConfigError` explaining that ``EMBEDDING_DIM``
          and the vector columns must match the model.
        """
        raise todo("M5", "SentenceTransformerEmbedder.__init__")

    @property
    def model_name(self) -> str:
        raise todo("M5", "SentenceTransformerEmbedder.model_name")

    @property
    def dimension(self) -> int:
        raise todo("M5", "SentenceTransformerEmbedder.dimension")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts in batches.

        Requirements:
        * Return one ``list[float]`` per input text, in input order.
        * Vectors must be L2-normalised (length 1.0) so cosine similarity equals
          the dot product. Check the ``encode`` options of sentence-transformers.
        * An empty input returns ``[]`` without calling the model.
        """
        raise todo("M5", "SentenceTransformerEmbedder.embed_documents")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single question. Must be consistent with :meth:`embed_documents`."""
        raise todo("M5", "SentenceTransformerEmbedder.embed_query")


def validate_vectors(vectors: Sequence[Sequence[float]], expected_dimension: int) -> None:
    """Raise :class:`EmbeddingError` if any vector has the wrong length or
    contains NaN/infinite values. The message should say which vector (index)
    is bad and why."""
    raise todo("M5", "validate_vectors")


def build_embedder(settings: Settings) -> Embedder:
    """Create the configured local embedder (PROVIDED)."""
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
