"""RAG intern training lab – starter package.

Modules marked PROVIDED are complete infrastructure. Functions that raise
``NotImplementedError`` are yours to implement, milestone by milestone
(see docs/milestones.md).
"""

__version__ = "0.1.0"

STRATEGIES: tuple[str, ...] = ("fixed", "recursive", "structured")
"""The three chunking strategies compared in this lab."""

TABLE_NAMES: dict[str, str] = {
    "fixed": "rag_chunks_fixed",
    "recursive": "rag_chunks_recursive",
    "structured": "rag_chunks_structured",
}
"""Mapping from chunking strategy to its pgvector table."""


def todo(milestone: str, what: str) -> NotImplementedError:
    """Build the error raised by unfinished functions.

    Usage inside a stub: ``raise todo("M2", "chunk_fixed")``.
    """
    return NotImplementedError(
        f"[{milestone}] {what} is not implemented yet - see docs/milestones.md#{milestone.lower()}"
    )
