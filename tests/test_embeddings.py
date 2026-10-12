"""M5 – local embeddings with sentence-transformers.

Run: python -m pytest -m m5
Tests marked ``model`` load the real model (first run downloads ~90 MB, needs
internet once). They are skipped if sentence-transformers is not installed.
"""

from __future__ import annotations

import math

import pytest

from rag_lab.config import ConfigError
from rag_lab.embeddings import EmbeddingError, SentenceTransformerEmbedder, validate_vectors

pytestmark = pytest.mark.m5

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_validate_vectors_accepts_good_vectors():
    validate_vectors([[0.0, 1.0], [1.0, 0.0]], 2)
    validate_vectors([], 2)


def test_validate_vectors_rejects_wrong_dimension():
    with pytest.raises(EmbeddingError, match="1"):
        validate_vectors([[0.0, 1.0], [1.0]], 2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_validate_vectors_rejects_non_finite(bad):
    with pytest.raises(EmbeddingError):
        validate_vectors([[0.0, bad]], 2)


@pytest.fixture(scope="module")
def model_embedder():
    pytest.importorskip("sentence_transformers")
    try:
        return SentenceTransformerEmbedder(MODEL, 384)
    except EmbeddingError as exc:
        pytest.skip(f"Model could not be loaded (offline?): {exc}")


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.model
def test_model_metadata(model_embedder):
    assert model_embedder.model_name == MODEL
    assert model_embedder.dimension == 384


@pytest.mark.model
def test_vectors_are_normalised_and_ordered(model_embedder):
    texts = ["annual leave", "hotel cap", "shuttle timings"]
    vectors = model_embedder.embed_documents(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        assert math.isclose(math.sqrt(_dot(v, v)), 1.0, rel_tol=1e-3)
    assert model_embedder.embed_documents([]) == []


@pytest.mark.model
def test_query_matches_document_embedding(model_embedder):
    q = model_embedder.embed_query("How many sick days do I get?")
    d = model_embedder.embed_documents(["How many sick days do I get?"])[0]
    assert _dot(q, d) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.model
def test_semantic_similarity(model_embedder):
    q = model_embedder.embed_query("How many unused leave days roll over to next year?")
    related, unrelated = model_embedder.embed_documents(
        [
            "A maximum of 10 days of unused annual leave may be carried forward.",
            "A free shuttle runs from Raidurg Metro Station every 20 minutes.",
        ]
    )
    assert _dot(q, related) > _dot(q, unrelated)


@pytest.mark.model
def test_dimension_mismatch_is_a_config_error():
    pytest.importorskip("sentence_transformers")
    with pytest.raises(ConfigError, match="EMBEDDING_DIM"):
        SentenceTransformerEmbedder(MODEL, 512)
