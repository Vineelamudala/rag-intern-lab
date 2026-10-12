"""M7 – similarity retrieval and metadata filtering.

Unit tests:        python -m pytest -m m7
Integration tests: python -m pytest -m "m7 and integration"   (needs Docker + M6)
"""

from __future__ import annotations

import pytest

from rag_lab import STRATEGIES
from rag_lab.ingestion import run_ingestion
from rag_lab.retrieval import RetrievalError, Retriever
from rag_lab.vector_store import (
    GLOBAL_LOCATION,
    SearchFilters,
    VectorStoreError,
    build_search_query,
)

from conftest import CHUNK_OVERLAP, CHUNK_SIZE, DIM
from fakes import HashingEmbedder, RecordingStore, make_result

pytestmark = pytest.mark.m7


def canned_store(model_name: str) -> RecordingStore:
    store = RecordingStore(
        search_results=[
            make_result(0, "a.md", "A", "alpha", score=0.9),
            make_result(1, "b.md", "B", "beta", score=0.5),
            make_result(2, "c.md", "C", "gamma", score=0.2),
        ]
    )
    store.models["structured"] = {model_name}
    return store


# -- Retriever (unit) ---------------------------------------------------------


def test_retrieve_embeds_question_and_passes_arguments(fake_embedder):
    store = canned_store(fake_embedder.model_name)
    filters = SearchFilters(doc_type="policy")
    Retriever(fake_embedder, store, "structured", top_k=3, min_similarity=0.0).retrieve(
        "carry forward", filters
    )
    strategy, vector, top_k, passed = store.search_calls[0]
    assert strategy == "structured" and top_k == 3 and passed == filters
    assert vector == pytest.approx(fake_embedder.embed_query("carry forward"))


def test_retrieve_applies_similarity_threshold(fake_embedder):
    store = canned_store(fake_embedder.model_name)
    results = Retriever(fake_embedder, store, "structured", 3, min_similarity=0.4).retrieve("q")
    assert [r.source for r in results] == ["a.md", "b.md"]


def test_retrieve_returns_nothing_when_all_below_threshold(fake_embedder):
    store = canned_store(fake_embedder.model_name)
    assert Retriever(fake_embedder, store, "structured", 3, min_similarity=0.95).retrieve("q") == []


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_empty_question_rejected(fake_embedder, question):
    store = canned_store(fake_embedder.model_name)
    with pytest.raises(RetrievalError):
        Retriever(fake_embedder, store).retrieve(question)


def test_embedding_model_mismatch_detected(fake_embedder):
    store = canned_store("some-other-model")
    with pytest.raises(RetrievalError, match="ingestion"):
        Retriever(fake_embedder, store).retrieve("leave")
    assert store.search_calls == []


def test_unknown_strategy_rejected(fake_embedder):
    with pytest.raises(RetrievalError):
        Retriever(fake_embedder, RecordingStore(), strategy="semantic")


# -- build_search_query (unit) ------------------------------------------------


def _params_values(params: dict) -> list:
    return [v for v in params.values() if isinstance(v, str | int)]


def test_search_query_is_parameterised():
    pytest.importorskip("psycopg")
    hostile = "policy'; DROP TABLE rag_chunks_fixed; --"
    query, params = build_search_query(
        "rag_chunks_fixed", [0.0] * DIM, 5, SearchFilters(doc_type=hostile, department="Finance")
    )
    text = query.as_string(None)
    assert hostile not in text and "Finance" not in text
    assert hostile in _params_values(params) and "Finance" in _params_values(params)
    assert 5 in _params_values(params)
    assert '"rag_chunks_fixed"' in text
    assert "<=>" in text


def test_location_filter_includes_global_documents_by_default():
    pytest.importorskip("psycopg")
    _, params = build_search_query("t", [0.0] * DIM, 5, SearchFilters(location="Pune"))
    assert "Pune" in _params_values(params) and GLOBAL_LOCATION in _params_values(params)
    _, exact = build_search_query(
        "t", [0.0] * DIM, 5, SearchFilters(location="Pune", include_all_locations=False)
    )
    assert GLOBAL_LOCATION not in _params_values(exact)


def test_search_query_rejects_bad_top_k():
    pytest.importorskip("psycopg")
    with pytest.raises(VectorStoreError):
        build_search_query("t", [0.0] * DIM, 0)


# -- integration: your PgVectorStore.search -----------------------------------


@pytest.fixture
def ingested_store(documents, test_store):
    embedder = HashingEmbedder(DIM)
    run_ingestion(documents, embedder, test_store, CHUNK_SIZE, CHUNK_OVERLAP)
    return test_store, embedder


def _retriever(store, embedder, strategy="structured", top_k=10):
    return Retriever(embedder, store, strategy, top_k=top_k, min_similarity=-1.0)


@pytest.mark.integration
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pg_search_sorted_limited_with_metadata(ingested_store, strategy):
    store, embedder = ingested_store
    results = _retriever(store, embedder, strategy, top_k=4).retrieve("maternity leave weeks")
    assert 0 < len(results) <= 4
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    for r in results:
        assert r.source.endswith(".md") and r.section_heading
        assert isinstance(r.metadata["effective_date"], str)


@pytest.mark.integration
def test_pg_finds_exact_chunk_text(ingested_store, chunks_for):
    store, embedder = ingested_store
    target = chunks_for("structured")[5]
    top = _retriever(store, embedder, top_k=1).retrieve(target.text)
    assert top[0].chunk_id == target.chunk_id
    assert top[0].score == pytest.approx(1.0, abs=1e-4)


@pytest.mark.integration
def test_pg_doc_type_filter(ingested_store):
    store, embedder = ingested_store
    results = _retriever(store, embedder).retrieve(
        "office address", SearchFilters(doc_type="circular")
    )
    assert results and {r.source for r in results} == {"hr_circular.md"}


@pytest.mark.integration
def test_pg_department_filter_is_case_insensitive(ingested_store):
    store, embedder = ingested_store
    results = _retriever(store, embedder).retrieve("claims", SearchFilters(department="finance"))
    assert results and {r.metadata["department"] for r in results} == {"Finance"}


@pytest.mark.integration
def test_pg_location_filter(ingested_store):
    store, embedder = ingested_store
    pune = _retriever(store, embedder, top_k=100).retrieve(
        "anchor days", SearchFilters(location="Pune")
    )
    assert pune and {r.metadata["location"] for r in pune} <= {"Pune", GLOBAL_LOCATION}
    exact = _retriever(store, embedder, top_k=100).retrieve(
        "anchor days", SearchFilters(location="Hyderabad", include_all_locations=False)
    )
    assert exact and {r.source for r in exact} == {"hr_circular.md"}


@pytest.mark.integration
def test_pg_filter_without_matches(ingested_store):
    store, embedder = ingested_store
    assert _retriever(store, embedder).retrieve("leave", SearchFilters(department="Legal")) == []
