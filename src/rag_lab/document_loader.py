"""Milestone M1 – load source documents and their metadata from ``data/rag_docs/``.

Each document is a Markdown file that starts with a small front-matter block::

    ---
    title: Leave Policy
    doc_type: policy
    effective_date: 2026-04-01
    department: Human Resources
    location: All Locations
    version: 3.0
    ---

    # Leave Policy
    ...

PROVIDED: the ``Document`` type, constants and the error class.
YOUR WORK: every function that raises ``NotImplementedError``.

Tests: ``python -m pytest -m m1``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_lab import todo

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})

REQUIRED_METADATA: tuple[str, ...] = (
    "title",
    "doc_type",
    "effective_date",
    "department",
    "location",
)


class DocumentLoadError(RuntimeError):
    """Raised when documents are missing or have invalid metadata.

    Always include the file name and what is wrong in the message, e.g.
    ``"bad.md: missing required metadata ['department']"``.
    """


@dataclass(frozen=True)
class Document:
    """A source document plus its document-level metadata (PROVIDED)."""

    doc_id: str
    source: str  # file name only, e.g. "leave_policy.md"
    title: str
    doc_type: str  # lower-case, e.g. "policy"
    effective_date: str  # ISO format YYYY-MM-DD
    department: str
    location: str
    text: str  # body WITHOUT the front-matter block
    content_hash: str
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, str]:
        """Document-level metadata that must be copied onto every chunk."""
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "effective_date": self.effective_date,
            "department": self.department,
            "location": self.location,
        }


def make_doc_id(source: str) -> str:
    """Return a stable document ID for the file name ``source``.

    Requirements (checked by tests):
    * deterministic – the same file name always gives the same ID, on any machine
      and in any Python process (so do NOT use Python's built-in ``hash()``);
    * case-insensitive – ``"Leave_Policy.md"`` and ``"leave_policy.md"`` match;
    * unique for different file names;
    * independent of the file's content (content changes are tracked separately
      in ``Document.content_hash``).

    Hint: a short hex prefix of a cryptographic hash from ``hashlib`` works well.
    """
    raise todo("M1", "make_doc_id")


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split raw file text into ``(metadata, body)``.

    Requirements:
    * The front matter is the block between a first line ``---`` and the next
      line ``---``. Each line inside is ``key: value``.
    * Keys are lower-cased and stripped. Values are stripped, and surrounding
      single or double quotes are removed.
    * Only the FIRST colon separates key from value
      (``title: "Circular: Move"`` → ``{"title": "Circular: Move"}``).
    * If there is no front matter, return ``({}, raw)``.
    * A non-empty line without a colon raises :class:`DocumentLoadError`.
    * The returned body must not contain the front-matter block.

    You may NOT add a YAML dependency – a few lines of string handling or a
    regular expression is enough.
    """
    raise todo("M1", "parse_front_matter")


def load_document(path: Path) -> Document:
    """Load one file and validate its metadata.

    Steps to implement:
    1. Read the file as UTF-8 and call :func:`parse_front_matter`.
    2. If ``title`` is missing, you may fall back to the first ``# H1`` heading.
    3. Raise :class:`DocumentLoadError` naming the file if any key in
       :data:`REQUIRED_METADATA` is missing or empty.
    4. Validate ``effective_date`` is an ISO date (``datetime.date.fromisoformat``)
       and raise :class:`DocumentLoadError` otherwise.
    5. Normalise line endings to ``\\n``; lower-case ``doc_type``.
    6. Compute ``doc_id`` with :func:`make_doc_id` and ``content_hash`` (e.g. SHA-256
       of the body). Put any non-required keys (e.g. ``version``) into
       ``extra_metadata``.
    """
    raise todo("M1", "load_document")


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every supported document directly inside ``docs_dir``.

    Requirements:
    * Only files whose suffix is in :data:`SUPPORTED_EXTENSIONS` (case-insensitive);
      ignore everything else. Do not recurse into sub-directories.
    * Return documents sorted by file name so ingestion is deterministic.
    * Raise :class:`DocumentLoadError` with a clear message if the directory does
      not exist, is not a directory, or contains no supported files.
    * Raise :class:`DocumentLoadError` if two files produce the same ``doc_id``.
    """
    raise todo("M1", "load_documents")
