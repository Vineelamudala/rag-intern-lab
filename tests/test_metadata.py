"""M2–M4 (with M1) – metadata on every chunk of the real lab documents.

Run: python -m pytest -m m2 tests/test_metadata.py   (and m3 / m4)
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from rag_lab.chunkers import REQUIRED_CHUNK_METADATA, missing_metadata

STRATEGIES = [
    pytest.param("fixed", marks=pytest.mark.m2, id="fixed"),
    pytest.param("recursive", marks=pytest.mark.m3, id="recursive"),
    pytest.param("structured", marks=pytest.mark.m4, id="structured"),
]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_chunk_has_all_required_metadata(strategy, chunks_for):
    for chunk in chunks_for(strategy):
        assert missing_metadata(chunk) == [], chunk.chunk_id
        assert set(REQUIRED_CHUNK_METADATA) <= set(chunk.metadata)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_document_is_chunked(strategy, documents, chunks_for):
    assert {c.doc_id for c in chunks_for(strategy)} == {d.doc_id for d in documents}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_metadata_matches_parent_document(strategy, documents, chunks_for):
    by_id = {d.doc_id: d for d in documents}
    for chunk in chunks_for(strategy):
        for key, value in by_id[chunk.doc_id].metadata().items():
            assert chunk.metadata[key] == value, (chunk.chunk_id, key)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_effective_dates_are_iso(strategy, chunks_for):
    for chunk in chunks_for(strategy):
        date.fromisoformat(chunk.metadata["effective_date"])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_section_headings_exist_in_source_document(strategy, documents, chunks_for):
    headings = {
        d.doc_id: {h.strip() for h in re.findall(r"^#{1,6}\s+(.+)$", d.text, flags=re.MULTILINE)}
        for d in documents
    }
    for chunk in chunks_for(strategy):
        for part in chunk.metadata["section_heading"].split(" > "):
            assert part in headings[chunk.doc_id], (chunk.chunk_id, part)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_ids_unique_across_corpus(strategy, chunks_for):
    ids = [c.chunk_id for c in chunks_for(strategy)]
    assert len(ids) == len(set(ids))


@pytest.mark.m4
def test_pune_rule_has_exact_heading_in_structured(chunks_for):
    pune = [
        c
        for c in chunks_for("structured")
        if c.metadata["source"] == "remote_work_policy.md" and "Monday and Wednesday" in c.text
    ]
    assert pune
    assert pune[0].metadata["section_heading"] == "Anchor Days > Pune Office Anchor Days"
