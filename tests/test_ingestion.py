"""M6 – ingestion pipeline and persistence validation.

Unit tests (fake embedder + RecordingStore):   python -m pytest -m m6
Integration tests (real pgvector, needs Docker): python -m pytest -m "m6 and integration"
"""

from __future__ import annotations

import pytest

from rag_lab import STRATEGIES, TABLE_NAMES
from rag_lab.chunkers import Chunk
from rag_lab.embeddings import EmbeddingError
from rag_lab.ingestion import (
    IngestionError,
    build_chunks,
    ingest_strategy,
    run_ingestion,
    summarise_chunks,
)
from rag_lab.vector_store import VectorStoreError

from conftest import CHUNK_OVERLAP, CHUNK_SIZE, DIM
from fakes import HashingEmbedder, RecordingStore

pytestmark = pytest.mark.m6


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(f"doc_x:fixed:{i:04d}", "doc_x", "fixed", i, text, {})


# -- unit tests ---------------------------------------------------------------


def test_summarise_chunks():
    assert summarise_chunks([]) == (0.0, 0, 0)
    assert summarise_chunks([_chunk(0, "ab"), _chunk(1, "abcd")]) == (3.0, 2, 4)


def test_build_chunks_validates_metadata(documents):
    chunks = build_chunks(documents, "fixed", CHUNK_SIZE, CHUNK_OVERLAP)
    assert chunks and all(c.strategy == "fixed" for c in chunks)


def test_ingestion_reports_counts_and_sizes(documents, fake_embedder):
    store = RecordingStore()
    reports = run_ingestion(documents, fake_embedder, store, CHUNK_SIZE, CHUNK_OVERLAP)
    assert [r.strategy for r in reports] == list(STRATEGIES)
    for report in reports:
        assert report.table == TABLE_NAMES[report.strategy]
        assert report.chunk_count == report.persisted_count == store.count(report.strategy)
        assert report.counts_match
        assert report.document_count == len(documents)
        assert 0 < report.min_chunk_chars <= report.avg_chunk_chars <= report.max_chunk_chars


def test_one_vector_per_chunk_and_same_model(documents, fake_embedder):
    store = RecordingStore()
    run_ingestion(documents, fake_embedder, store, CHUNK_SIZE, CHUNK_OVERLAP)
    for strategy in STRATEGIES:
        rows = store.tables[strategy]
        assert all(len(vec) == DIM for _, vec in rows)
        assert store.embedding_models(strategy) == {fake_embedder.model_name}


def test_count_mismatch_is_detected(documents, fake_embedder):
    store = RecordingStore(lose_rows=1)
    with pytest.raises(IngestionError, match="persisted|count"):
        ingest_strategy(documents, "recursive", fake_embedder, store, CHUNK_SIZE, CHUNK_OVERLAP)


def test_wrong_embedding_dimension_is_rejected(documents):
    class BadEmbedder(HashingEmbedder):
        @property
        def dimension(self) -> int:
            return DIM + 1  # claims a dimension its vectors do not have

    with pytest.raises((EmbeddingError, IngestionError, VectorStoreError)):
        ingest_strategy(
            documents, "fixed", BadEmbedder(DIM), RecordingStore(), CHUNK_SIZE, CHUNK_OVERLAP
        )


def test_repeated_ingestion_is_deterministic(documents, fake_embedder):
    store = RecordingStore()
    run_ingestion(documents, fake_embedder, store, CHUNK_SIZE, CHUNK_OVERLAP)
    first = {s: store.chunk_ids(s) for s in STRATEGIES}
    run_ingestion(documents, fake_embedder, store, CHUNK_SIZE, CHUNK_OVERLAP)
    assert {s: store.chunk_ids(s) for s in STRATEGIES} == first


# -- integration tests: your PgVectorStore ------------------------------------


@pytest.mark.integration
def test_pgvector_counts_and_idempotency(documents, fake_embedder, test_store):
    first = run_ingestion(documents, fake_embedder, test_store, CHUNK_SIZE, CHUNK_OVERLAP)
    ids_first = {s: test_store.chunk_ids(s) for s in STRATEGIES}
    second = run_ingestion(documents, fake_embedder, test_store, CHUNK_SIZE, CHUNK_OVERLAP)
    ids_second = {s: test_store.chunk_ids(s) for s in STRATEGIES}

    assert all(r.counts_match for r in first + second)
    assert [r.persisted_count for r in second] == [r.persisted_count for r in first]
    assert ids_first == ids_second
    for ids in ids_second.values():
        assert len(ids) == len(set(ids)) and ids == sorted(ids)


@pytest.mark.integration
def test_pgvector_replace_removes_stale_rows(documents, fake_embedder, test_store):
    run_ingestion(documents, fake_embedder, test_store, 300, 30, ["recursive"])
    many = test_store.count("recursive")
    run_ingestion(documents, fake_embedder, test_store, 1200, 30, ["recursive"])
    assert test_store.count("recursive") < many


@pytest.mark.integration
def test_pgvector_records_embedding_model(documents, fake_embedder, test_store):
    run_ingestion(documents, fake_embedder, test_store, CHUNK_SIZE, CHUNK_OVERLAP, ["fixed"])
    assert test_store.embedding_models("fixed") == {fake_embedder.model_name}


@pytest.mark.integration
def test_pgvector_rejects_mismatched_inputs(documents, fake_embedder, test_store):
    chunks = build_chunks(documents, "fixed", CHUNK_SIZE, CHUNK_OVERLAP)
    vectors = fake_embedder.embed_documents([c.text for c in chunks])
    with pytest.raises(VectorStoreError):
        test_store.replace_chunks("fixed", chunks, vectors[:-1], fake_embedder.model_name)
    with pytest.raises(VectorStoreError):
        test_store.replace_chunks("fixed", chunks[:1], [[0.0] * 3], fake_embedder.model_name)
    with pytest.raises(VectorStoreError):
        test_store.replace_chunks(
            "fixed", [chunks[0], chunks[0]], vectors[:2], fake_embedder.model_name
        )
