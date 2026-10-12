"""Milestone M11 – evaluate all three chunking strategies on the 15 test queries.

PROVIDED: parsing of ``data/rag_test_queries.md`` into :class:`EvalQuery` objects
(:func:`load_eval_queries`) and the record type.
YOUR WORK: scoring, aggregation and reporting functions that raise
``NotImplementedError``.

The expected facts in the query file are for YOU to judge answers with.
Never feed them into retrieval, prompts or answer generation.

Tests: ``python -m pytest -m m11``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_lab import todo
from rag_lab.answering import AnswerResult, LLMProvider
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import SearchFilters, SearchResult

_QUERY_HEADER_RE = re.compile(
    r"^### (?P<id>Q\d{2}): (?P<title>.+?) \((?P<cats>[A-Z, ]+)\)\s*$", re.MULTILINE
)
_FIELD_RE = re.compile(r"^- \*\*(?P<name>[^:*]+):\*\*\s*(?P<value>.*)$", re.MULTILINE)
_FILTER_RE = re.compile(r"`--(?P<flag>doc-type|department|location)\s+(?P<value>[^`]+)`")
_SOURCE_RE = re.compile(r"`(?P<file>[\w.-]+\.md)`")


class EvaluationError(RuntimeError):
    """Raised when the query file cannot be parsed or an evaluation fails."""


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query parsed from ``data/rag_test_queries.md`` (PROVIDED)."""

    query_id: str
    title: str
    categories: frozenset[str]
    question: str
    filters: SearchFilters
    expected_sources: tuple[str, ...]
    expected_facts: str

    @property
    def answerable(self) -> bool:
        """Unanswerable queries (category ``U``) have no expected source."""
        return "U" not in self.categories


@dataclass
class EvalRecord:
    """Outcome of one query against one strategy."""

    query_id: str
    strategy: str
    retrieved: list[tuple[str, str]] = field(default_factory=list)  # (source, section)
    retrieval_hit: bool | None = None  # None for unanswerable queries
    answer: AnswerResult | None = None  # None in retrieval-only mode
    notes: str = ""


def _parse_filters(raw: str) -> SearchFilters:
    if raw.strip().lower().startswith("none"):
        return SearchFilters()
    values = {m.group("flag"): m.group("value").strip() for m in _FILTER_RE.finditer(raw)}
    return SearchFilters(
        doc_type=values.get("doc-type"),
        department=values.get("department"),
        location=values.get("location"),
    )


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Parse the evaluation file into queries, in file order (PROVIDED)."""
    if not path.exists():
        raise EvaluationError(f"Query file not found: {path}")
    text = path.read_text(encoding="utf-8")
    headers = list(_QUERY_HEADER_RE.finditer(text))
    if not headers:
        raise EvaluationError(f"No '### Qnn: title (CATEGORY)' headings found in {path}")

    queries: list[EvalQuery] = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[header.end() : end]
        fields = {
            m.group("name").strip().lower(): m.group("value").strip()
            for m in _FIELD_RE.finditer(block)
        }
        if "question" not in fields:
            raise EvaluationError(f"{header.group('id')}: missing '- **Question:**' line")
        queries.append(
            EvalQuery(
                query_id=header.group("id"),
                title=header.group("title").strip(),
                categories=frozenset(c.strip() for c in header.group("cats").split(",")),
                question=fields["question"],
                filters=_parse_filters(fields.get("filters", "none")),
                expected_sources=tuple(_SOURCE_RE.findall(fields.get("expected source", ""))),
                expected_facts=fields.get("expected answer / key facts", ""),
            )
        )
    return queries


# --------------------------------------------------------------------------- #
# YOUR WORK – M11
# --------------------------------------------------------------------------- #


def is_retrieval_hit(query: EvalQuery, results: list[SearchResult]) -> bool | None:
    """Decide whether retrieval found what the query needs.

    * Unanswerable queries return ``None`` (there is nothing to find).
    * Otherwise return ``True`` if at least one expected source file appears in
      ``results``. You may make this stricter (e.g. also matching the expected
      section) – if you do, explain it in your report.
    """
    raise todo("M11", "is_retrieval_hit")


def evaluate_strategy(
    queries: list[EvalQuery],
    retriever: Retriever,
    provider: LLMProvider | None = None,
) -> list[EvalRecord]:
    """Run every query against ``retriever`` (one strategy).

    * Return one :class:`EvalRecord` per query, in query order, with
      ``strategy=retriever.strategy``.
    * Use each query's own ``filters``.
    * Always record the retrieved ``(source, section_heading)`` pairs and
      :func:`is_retrieval_hit`.
    * If ``provider`` is given, also call
      :func:`rag_lab.answering.answer_question` and store the result; with no
      provider run in retrieval-only mode (``answer=None``).
    * One failing query must not stop the run: catch the error and record it in
      ``notes``.
    """
    raise todo("M11", "evaluate_strategy")


def summarise(records: list[EvalRecord]) -> dict[str, dict[str, float | int]]:
    """Aggregate records per strategy.

    Return ``{strategy: {...}}`` with at least:
    * ``retrieval_hits`` and ``answerable`` – hits among answerable queries;
    * ``answered`` – answerable queries with ``found=True`` (0 in retrieval-only mode);
    * ``correct_refusals`` and ``unanswerable`` – unanswerable queries that were
      refused (``found=False``).

    Whether an answer's *content* is correct still needs a human: compare it
    with ``expected_facts`` and record that in the query file.
    """
    raise todo("M11", "summarise")


def render_markdown(records: list[EvalRecord], summary: dict[str, dict[str, float | int]]) -> str:
    """Render a Markdown report: one row per (query, strategy) and the summary table.

    Make it easy to copy rows into ``data/rag_test_queries.md``.
    """
    raise todo("M11", "render_markdown")
