"""Milestones M2–M4 – three chunking strategies.

1. ``fixed``      (M2) – fixed-size character windows with overlap, written by you
                         in plain Python (no library).
2. ``recursive``  (M3) – LangChain ``RecursiveCharacterTextSplitter``.
3. ``structured`` (M4) – LangChain ``MarkdownHeaderTextSplitter`` first, then split
                         sections that are still longer than ``chunk_size``.

Every strategy must return :class:`Chunk` objects with the SAME metadata keys
(:data:`REQUIRED_CHUNK_METADATA`) so the rest of the pipeline does not care which
strategy produced them.

Conventions every strategy must follow (tests check these):
* ``chunk_id`` is ``f"{doc_id}:{strategy}:{chunk_index:04d}"``; ``chunk_index``
  starts at 0 for each document.
* No chunk is empty or whitespace-only.
* No chunk is longer than ``chunk_size`` characters.
* ``metadata`` contains the document's :meth:`Document.metadata` plus
  ``section_heading``, ``strategy`` and ``chunk_index`` (as a string).
* ``section_heading`` is formatted by :func:`format_section_heading`.
* Each chunk gets its own ``metadata`` dict (never share one dict between chunks).

PROVIDED: types, constants, validation helper, the dispatcher.
YOUR WORK: every function that raises ``NotImplementedError``.

Tests: ``python -m pytest -m m2`` / ``-m m3`` / ``-m m4``
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from rag_lab import STRATEGIES, todo
from rag_lab.document_loader import Document

REQUIRED_CHUNK_METADATA: tuple[str, ...] = (
    "source",
    "doc_id",
    "title",
    "doc_type",
    "effective_date",
    "section_heading",
    "department",
    "location",
)


class ChunkingError(ValueError):
    """Raised for invalid chunking configuration or unknown strategies."""


@dataclass(frozen=True)
class Chunk:
    """A piece of a document ready to be embedded and stored (PROVIDED)."""

    chunk_id: str
    doc_id: str
    strategy: str
    chunk_index: int
    text: str
    metadata: dict[str, str]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def char_count(self) -> int:
        return len(self.text)


def missing_metadata(chunk: Chunk) -> list[str]:
    """Return required metadata keys that are absent or empty on ``chunk`` (PROVIDED)."""
    return [key for key in REQUIRED_CHUNK_METADATA if not chunk.metadata.get(key)]


# --------------------------------------------------------------------------- #
# Shared helpers – M2
# --------------------------------------------------------------------------- #


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Raise :class:`ChunkingError` unless ``chunk_size > 0`` and
    ``0 <= chunk_overlap < chunk_size``. Call it at the start of every strategy."""
    raise todo("M2", "validate_chunk_params")


def format_section_heading(path: tuple[str, ...] | list[str], fallback: str) -> str:
    """Render a heading path for citations.

    ``path`` is the chain of headings from the top-level ``#`` heading down to
    the heading that contains the chunk, e.g.
    ``("Leave Policy", "Annual Leave", "Carry Forward and Lapse")``.

    Rules:
    * Drop the first element (the H1 repeats the document title) when the
      path has more than one element.
    * Join the remaining elements with ``" > "``.
    * A single-element path returns that element; an empty path returns
      ``fallback`` (use the document title).

    Example: ``("Leave Policy", "Annual Leave", "Encashment")`` →
    ``"Annual Leave > Encashment"``.
    """
    raise todo("M2", "format_section_heading")


def heading_path_at(text: str, offset: int) -> tuple[str, ...]:
    """Return the heading path in effect at character ``offset`` of ``text``.

    Scan the Markdown headings (lines starting with 1–6 ``#`` characters followed
    by a space) that start at or before ``offset`` and keep a stack: a heading of
    level *n* replaces any previous heading of level >= *n*.

    Example – for ``"# A\\n## B\\ntext\\n## C\\nmore"`` the path at the offset of
    ``"more"`` is ``("A", "C")``; at offset 0 it is ``("A",)``; before any
    heading it is ``()``.

    Used by the fixed and recursive strategies, which do not know about sections
    themselves.
    """
    raise todo("M2", "heading_path_at")


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


def chunk_fixed(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """M2 – fixed-size character windows. Implement the windowing yourself.

    Requirements:
    * Window ``i`` starts at ``i * (chunk_size - chunk_overlap)`` and contains up to
      ``chunk_size`` characters of ``doc.text``. Stop once a window reaches the end.
    * Do NOT strip or otherwise modify window text – with ``chunk_overlap=0`` the
      chunks concatenated in order must reproduce ``doc.text`` exactly
      (except for skipped whitespace-only windows).
    * The last ``chunk_overlap`` characters of a chunk equal the first
      ``chunk_overlap`` characters of the next chunk.
    * ``section_heading`` = heading in effect at the window's FIRST character
      (:func:`heading_path_at` + :func:`format_section_heading`).

    This strategy is deliberately naive. Note in your report where it cuts
    sentences, rules or tables in half.
    """
    raise todo("M2", "chunk_fixed")


def chunk_recursive(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """M3 – LangChain ``RecursiveCharacterTextSplitter``.

    Hints:
    * ``from langchain_text_splitters import RecursiveCharacterTextSplitter``
    * Read the splitter's documentation for ``chunk_size``, ``chunk_overlap`` and
      ``add_start_index`` – the start index tells you where each piece came from,
      which you need for ``section_heading``.
    * Same metadata and ID conventions as every other strategy.
    """
    raise todo("M3", "chunk_recursive")


def chunk_structured(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """M4 – structure-aware splitting with ``MarkdownHeaderTextSplitter``.

    Requirements:
    * Split on ``#``, ``##`` and ``###`` headings. Keep the heading lines in the
      chunk text (look at the splitter's ``strip_headers`` option).
    * Every chunk belongs to exactly one section; ``section_heading`` comes from
      the splitter's header metadata (formatted with :func:`format_section_heading`).
    * Sections longer than ``chunk_size`` are split again (e.g. with a
      ``RecursiveCharacterTextSplitter``); every piece keeps its section heading.
    * Skip sections whose content is empty after stripping.
    """
    raise todo("M4", "chunk_structured")


CHUNKERS: dict[str, Callable[[Document, int, int], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "structured": chunk_structured,
}


def chunk_document(
    doc: Document, strategy: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk one document with the named strategy (PROVIDED dispatcher)."""
    try:
        chunker = CHUNKERS[strategy]
    except KeyError as exc:
        raise ChunkingError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}") from exc
    return chunker(doc, chunk_size, chunk_overlap)


def chunk_documents(
    docs: list[Document], strategy: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every document with the named strategy, preserving order (PROVIDED)."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, chunk_size, chunk_overlap))
    return chunks
