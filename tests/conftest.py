"""Shared fixtures and milestone handling (PROVIDED).

* Unit tests need no database, no model download and no gateway key.
* A test that hits an unfinished function (``NotImplementedError``) is reported
  as SKIPPED with the milestone in the reason, so the summary tells you what is
  still TODO. Pass ``--strict-todos`` to see those as failures instead.
* Integration tests (``-m integration``) need ``docker compose up -d``; they use
  ``test_rag_chunks_*`` tables so your real tables are never touched.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOCS_DIR = ROOT / "data" / "rag_docs"
QUERIES_FILE = ROOT / "data" / "rag_test_queries.md"
EXPECTED_SOURCES = {
    "leave_policy.md",
    "travel_expense_policy.md",
    "remote_work_policy.md",
    "hr_circular.md",
    "employee_faq.md",
    "holiday_calendar.md",
}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DIM = 384


# --------------------------------------------------------------------------- #
# NotImplementedError -> "TODO" skip
# --------------------------------------------------------------------------- #


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--strict-todos",
        action="store_true",
        default=False,
        help="Report tests blocked by NotImplementedError as failures instead of skips.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--strict-todos"):
        return
    if (
        report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(NotImplementedError)
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"TODO {call.excinfo.value}")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@cache
def _load_documents():  # type: ignore[no-untyped-def]
    from rag_lab.document_loader import load_documents

    return load_documents(DOCS_DIR)


@pytest.fixture
def documents():  # type: ignore[no-untyped-def]
    """All six lab documents, loaded with YOUR loader (M1)."""
    return _load_documents()


@pytest.fixture
def chunks_for(documents):  # type: ignore[no-untyped-def]
    """``chunks_for("fixed")`` – all documents chunked with YOUR strategy (M2–M4)."""
    from rag_lab.chunkers import chunk_documents

    cache_: dict[str, list] = {}

    def _get(strategy: str) -> list:
        if strategy not in cache_:
            cache_[strategy] = chunk_documents(documents, strategy, CHUNK_SIZE, CHUNK_OVERLAP)
        return cache_[strategy]

    return _get


@pytest.fixture
def fake_embedder():  # type: ignore[no-untyped-def]
    from fakes import HashingEmbedder

    return HashingEmbedder(DIM)


@pytest.fixture(scope="session")
def database_url() -> str:
    """URL for integration tests. Skips them if PostgreSQL is not reachable."""
    from rag_lab.config import load_settings

    url = os.environ.get("RAG_TEST_DATABASE_URL") or load_settings().database_url
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=3):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"PostgreSQL not reachable ({exc.__class__.__name__}). "
            "Start it with `docker compose up -d` and check .env."
        )
    return url


@pytest.fixture
def test_store(database_url):  # type: ignore[no-untyped-def]
    """A PgVectorStore that writes to ``test_rag_chunks_*`` tables."""
    from rag_lab.vector_store import PgVectorStore

    with PgVectorStore(database_url, DIM, table_prefix="test_") as store:
        yield store
