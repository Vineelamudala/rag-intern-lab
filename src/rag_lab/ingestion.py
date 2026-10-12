"""Milestone M6 – ingestion pipeline: load -> chunk -> embed -> store -> verify.

Ingestion is completely separate from question answering: it never calls the
LLM gateway, and the answering code never writes to the database.

PROVIDED: the report type, the multi-strategy driver.
YOUR WORK: :func:`summarise_chunks`, :func:`build_chunks`, :func:`ingest_strategy`.

Tests: ``python -m pytest -m m6``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab import STRATEGIES, todo
from rag_lab.chunkers import Chunk
from rag_lab.document_loader import Document
from rag_lab.embeddings import Embedder
from rag_lab.vector_store import VectorStore


class IngestionError(RuntimeError):
    """Raised when ingestion produces inconsistent or invalid results."""


@dataclass(frozen=True)
class StrategyReport:
    """Summary of one strategy's ingestion run (PROVIDED)."""

    strategy: str
    table: str
    document_count: int
    chunk_count: int
    persisted_count: int
    avg_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int

    @property
    def counts_match(self) -> bool:
        return self.chunk_count == self.persisted_count

    def format_row(self) -> str:
        status = "OK" if self.counts_match else "MISMATCH"
        return (
            f"{self.strategy:<11} {self.table:<23} docs={self.document_count:<3} "
            f"chunks={self.chunk_count:<4} persisted={self.persisted_count:<4} "
            f"avg_chars={self.avg_chunk_chars:7.1f} min={self.min_chunk_chars:<4} "
            f"max={self.max_chunk_chars:<5} [{status}]"
        )


def summarise_chunks(chunks: Sequence[Chunk]) -> tuple[float, int, int]:
    """Return ``(average, minimum, maximum)`` chunk length in characters.

    An empty sequence returns ``(0.0, 0, 0)``.
    """
    raise todo("M6", "summarise_chunks")


def build_chunks(
    documents: Sequence[Document], strategy: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk all documents with one strategy and validate the result.

    Requirements:
    * Use :func:`rag_lab.chunkers.chunk_documents`.
    * Raise :class:`IngestionError` if no chunks are produced.
    * Raise :class:`IngestionError` naming the chunk ID if any chunk is missing
      required metadata (see :func:`rag_lab.chunkers.missing_metadata`).
    """
    raise todo("M6", "build_chunks")


def ingest_strategy(
    documents: Sequence[Document],
    strategy: str,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> StrategyReport:
    """Run the full pipeline for ONE strategy and verify what was persisted.

    Steps:
    1. :func:`build_chunks`.
    2. Embed the chunk texts with ``embedder.embed_documents`` (one call, same
       embedder for every strategy) and check you got one valid vector per chunk
       (:func:`rag_lab.embeddings.validate_vectors`).
    3. ``store.replace_chunks(strategy, chunks, vectors, embedder.model_name)``.
    4. Read back ``store.count(strategy)``.
    5. Build and return a :class:`StrategyReport` (table name from
       :data:`rag_lab.TABLE_NAMES`, document count = distinct ``doc_id`` values).
    6. Raise :class:`IngestionError` if the generated, inserted and persisted
       counts are not all equal – the message must show all three numbers.
    """
    raise todo("M6", "ingest_strategy")


def run_ingestion(
    documents: Sequence[Document],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
    strategies: Sequence[str] = STRATEGIES,
) -> list[StrategyReport]:
    """Ingest ``documents`` once per strategy (PROVIDED driver)."""
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise IngestionError(f"Unknown strategies {unknown}; expected a subset of {STRATEGIES}")
    if not documents:
        raise IngestionError("No documents to ingest")
    return [
        ingest_strategy(documents, s, embedder, store, chunk_size, chunk_overlap)
        for s in strategies
    ]
