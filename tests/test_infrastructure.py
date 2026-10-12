"""M0 – provided infrastructure. These tests should pass BEFORE you write any code.

Run: python -m pytest -m m0
"""

from __future__ import annotations

import re

import pytest

from rag_lab import TABLE_NAMES, todo
from rag_lab.config import ConfigError, load_settings
from rag_lab.evaluation import load_eval_queries
from rag_lab.vector_store import (
    DatabaseConnectionError,
    PgVectorStore,
    VectorStoreError,
    create_index_sql,
    create_table_sql,
    table_for,
)

from conftest import DOCS_DIR, EXPECTED_SOURCES, QUERIES_FILE

pytestmark = pytest.mark.m0

REQUIRED_KEYS = ("title", "doc_type", "effective_date", "department", "location")


# -- data files -------------------------------------------------------------


def test_six_documents_exist():
    names = {p.name for p in DOCS_DIR.iterdir() if p.is_file()}
    assert names == EXPECTED_SOURCES


@pytest.mark.parametrize("name", sorted(EXPECTED_SOURCES))
def test_documents_have_front_matter_and_training_label(name):
    raw = (DOCS_DIR / name).read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    header = raw.split("---", 2)[1]
    for key in REQUIRED_KEYS:
        assert re.search(rf"^{key}:\s*\S", header, re.MULTILINE), f"{name} lacks {key}"
    assert "TRAINING DATA ONLY" in raw


def test_queries_file_is_outside_docs_dir():
    assert QUERIES_FILE.exists()
    assert QUERIES_FILE.parent != DOCS_DIR
    assert not (DOCS_DIR / QUERIES_FILE.name).exists()


@pytest.mark.parametrize("docs_dir", ["data", "."])
def test_config_refuses_docs_dir_containing_query_file(docs_dir):
    with pytest.raises(ConfigError, match="test queries can never be ingested"):
        load_settings({"RAG_DOCS_DIR": docs_dir})


def test_exactly_fifteen_queries_are_parsed():
    queries = load_eval_queries(QUERIES_FILE)
    assert [q.query_id for q in queries] == [f"Q{i:02d}" for i in range(1, 16)]
    for q in queries:
        assert q.question.endswith("?")
        if q.answerable:
            assert q.expected_sources, q.query_id
            assert set(q.expected_sources) <= EXPECTED_SOURCES
            assert q.expected_facts
        else:
            assert q.expected_sources == ()


def test_query_categories_cover_every_type():
    queries = load_eval_queries(QUERIES_FILE)
    categories = set().union(*(q.categories for q in queries))
    assert categories == {"S", "M", "F", "D", "U"}
    assert sum(not q.answerable for q in queries) == 3


def test_query_filters_are_parsed():
    by_id = {q.query_id: q for q in load_eval_queries(QUERIES_FILE)}
    assert by_id["Q07"].filters.doc_type == "circular"
    assert by_id["Q08"].filters.department == "Finance"
    assert by_id["Q09"].filters.location == "Pune"
    assert by_id["Q12"].filters.location == "Hyderabad"
    assert by_id["Q01"].filters.is_empty()
    assert by_id["Q11"].filters.is_empty()  # "none (also try ...)" means no default filter


# -- configuration ----------------------------------------------------------


def test_default_settings_are_valid():
    settings = load_settings({})
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dim == 384
    assert settings.litellm_api_base == ""  # never invented


def test_settings_validation_messages():
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        load_settings({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"})
    with pytest.raises(ConfigError, match="integer"):
        load_settings({"CHUNK_SIZE": "big"})


def test_database_url_built_from_parts():
    settings = load_settings({"POSTGRES_PASSWORD": "pw", "POSTGRES_PORT": "6000"})
    assert settings.database_url == "postgresql://rag:pw@localhost:6000/rag_lab"


def test_gateway_settings_must_all_be_present():
    with pytest.raises(ConfigError, match="LITELLM_API_BASE, LITELLM_API_KEY, LITELLM_MODEL"):
        load_settings({}).require_gateway()
    ok = load_settings(
        {
            "LITELLM_API_BASE": "https://gw.test",
            "LITELLM_API_KEY": "fake-secret-value-123",
            "LITELLM_MODEL": "m",
        }
    )
    assert ok.require_gateway() is ok
    assert "fake-secret-value-123" not in repr(ok)  # the key is never shown in repr/logs


def test_todo_helper_names_milestone():
    err = todo("M3", "chunk_recursive")
    assert isinstance(err, NotImplementedError)
    assert "[M3]" in str(err) and "chunk_recursive" in str(err)


# -- database helpers -------------------------------------------------------


def test_table_whitelist():
    assert {table_for(s) for s in TABLE_NAMES} == set(TABLE_NAMES.values())
    with pytest.raises(VectorStoreError):
        table_for("rag_chunks_fixed; DROP TABLE x")


def test_schema_sql_uses_configured_dimension():
    pytest.importorskip("psycopg")
    text = create_table_sql("rag_chunks_fixed", 384).as_string(None)
    assert "vector(384)" in text and '"rag_chunks_fixed"' in text
    indexes = " ".join(s.as_string(None) for s in create_index_sql("rag_chunks_fixed"))
    assert "hnsw" in indexes and "vector_cosine_ops" in indexes


def test_store_rejects_bad_configuration():
    with pytest.raises(VectorStoreError, match="DATABASE_URL"):
        PgVectorStore("", 384)
    with pytest.raises(VectorStoreError, match="table_prefix"):
        PgVectorStore("postgresql://x", 384, table_prefix="x; drop")


def test_unreachable_database_gives_helpful_error():
    store = PgVectorStore("postgresql://nobody@127.0.0.1:1/none", 384, connect_timeout=1)
    with pytest.raises(DatabaseConnectionError, match="docker compose up -d"):
        store.connect()


@pytest.mark.integration
def test_init_schema_creates_three_tables(database_url):
    with PgVectorStore(database_url, 384, table_prefix="test_") as store:
        tables = store.init_schema()
        assert tables == [f"test_{t}" for t in TABLE_NAMES.values()]
        assert all(store.table_exists(s) for s in TABLE_NAMES)
