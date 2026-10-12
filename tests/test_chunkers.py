"""M2–M4 – chunking strategies, tested on a small hand-written document.

Run: python -m pytest -m m2   (fixed + shared helpers)
     python -m pytest -m m3   (recursive)
     python -m pytest -m m4   (structured)

These tests do not depend on your M1 loader.
"""

from __future__ import annotations

import pytest

from rag_lab.chunkers import (
    ChunkingError,
    chunk_document,
    chunk_fixed,
    chunk_recursive,
    chunk_structured,
    format_section_heading,
    heading_path_at,
    validate_chunk_params,
)
from rag_lab.document_loader import Document

M2, M3, M4 = pytest.mark.m2, pytest.mark.m3, pytest.mark.m4

SAMPLE = (
    """# Sample Policy

Intro paragraph for the whole policy.

## Annual Leave

Employees receive 18 days of annual leave per year.

### Carry Forward

Up to 10 days may be carried forward. The rest lapses on 31 March.

## Sick Leave

Employees receive 10 days of sick leave. A certificate is needed after 2 days.
"""
    + "\n".join(f"Sick leave rule number {i} applies to everyone." for i in range(30))
    + "\n"
)


@pytest.fixture
def doc() -> Document:
    return Document(
        doc_id="doc_sample",
        source="sample.md",
        title="Sample Policy",
        doc_type="policy",
        effective_date="2026-04-01",
        department="Human Resources",
        location="All Locations",
        text=SAMPLE,
        content_hash="x",
    )


STRATEGY_FUNCS = [
    pytest.param(chunk_fixed, "fixed", marks=M2, id="fixed"),
    pytest.param(chunk_recursive, "recursive", marks=M3, id="recursive"),
    pytest.param(chunk_structured, "structured", marks=M4, id="structured"),
]


# -- M2: shared helpers -------------------------------------------------------


@M2
@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ChunkingError):
        validate_chunk_params(size, overlap)


@M2
def test_valid_parameters_pass():
    validate_chunk_params(100, 0)
    validate_chunk_params(100, 99)


@M2
def test_format_section_heading():
    assert format_section_heading(("Leave Policy", "Annual Leave", "Encashment"), "x") == (
        "Annual Leave > Encashment"
    )
    assert format_section_heading(["Leave Policy", "Sick Leave"], "x") == "Sick Leave"
    assert format_section_heading(("Leave Policy",), "x") == "Leave Policy"
    assert format_section_heading((), "Fallback") == "Fallback"


@M2
def test_heading_path_at():
    text = "# A\n## B\nbody one\n### B1\nbody two\n## C\nbody three"
    assert heading_path_at(text, 0) == ("A",)
    assert heading_path_at(text, text.index("body one")) == ("A", "B")
    assert heading_path_at(text, text.index("body two")) == ("A", "B", "B1")
    assert heading_path_at(text, text.index("body three")) == ("A", "C")
    assert heading_path_at("no headings here", 5) == ()


# -- M2: fixed-size specifics -------------------------------------------------


@M2
def test_fixed_windows_overlap_exactly(doc):
    chunks = chunk_fixed(doc, 200, 50)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text[-50:] == b.text[:50]


@M2
def test_fixed_without_overlap_reproduces_text(doc):
    chunks = chunk_fixed(doc, 150, 0)
    assert "".join(c.text for c in chunks) == doc.text


@M2
def test_fixed_heading_is_heading_at_window_start(doc):
    for chunk in chunk_fixed(doc, 120, 0):
        start = doc.text.index(chunk.text)
        expected = format_section_heading(heading_path_at(doc.text, start), doc.title)
        assert chunk.metadata["section_heading"] == expected


# -- conventions shared by all strategies -------------------------------------


@pytest.mark.parametrize(("func", "strategy"), STRATEGY_FUNCS)
def test_chunk_conventions(func, strategy, doc):
    chunks = func(doc, 200, 40)
    assert chunks
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc_sample:{strategy}:{i:04d}"
        assert chunk.chunk_index == i
        assert chunk.strategy == strategy
        assert chunk.doc_id == "doc_sample"
        assert chunk.text.strip()
        assert len(chunk.text) <= 200
        assert chunk.metadata["strategy"] == strategy
        assert chunk.metadata["chunk_index"] == str(i)
        for key, value in doc.metadata().items():
            assert chunk.metadata[key] == value


@pytest.mark.parametrize(("func", "strategy"), STRATEGY_FUNCS)
def test_chunks_do_not_share_metadata_dicts(func, strategy, doc):
    chunks = func(doc, 200, 40)
    assert len({id(c.metadata) for c in chunks}) == len(chunks)


@pytest.mark.parametrize(("func", "strategy"), STRATEGY_FUNCS)
def test_chunking_is_deterministic(func, strategy, doc):
    first = [(c.chunk_id, c.text) for c in func(doc, 200, 40)]
    assert first == [(c.chunk_id, c.text) for c in func(doc, 200, 40)]


@pytest.mark.parametrize(("func", "strategy"), STRATEGY_FUNCS)
def test_chunk_size_is_configurable(func, strategy, doc):
    assert len(func(doc, 150, 20)) > len(func(doc, 600, 20))


@pytest.mark.parametrize(("func", "strategy"), STRATEGY_FUNCS)
def test_strategies_validate_parameters(func, strategy, doc):
    with pytest.raises(ChunkingError):
        func(doc, 100, 100)


# -- M3: recursive specifics --------------------------------------------------


@M3
def test_recursive_prefers_natural_boundaries(doc):
    """Most recursive chunks should end at a line break or sentence end, not mid-word."""
    chunks = chunk_recursive(doc, 200, 0)
    clean_endings = sum(c.text.rstrip()[-1] in ".:!?" or c.text.endswith("\n") for c in chunks)
    assert clean_endings >= len(chunks) * 0.7


@M3
def test_recursive_section_headings(doc):
    chunks = chunk_recursive(doc, 200, 0)
    rule = next(c for c in chunks if "Sick leave rule number 20" in c.text)
    assert rule.metadata["section_heading"] == "Sick Leave"


# -- M4: structured specifics -------------------------------------------------


@M4
def test_structured_chunks_stay_in_one_section(doc):
    for chunk in chunk_structured(doc, 300, 30):
        heading_lines = [ln for ln in chunk.text.splitlines() if ln.startswith("#")]
        assert len(heading_lines) <= 1, chunk.text


@M4
def test_structured_exact_section_headings(doc):
    chunks = chunk_structured(doc, 300, 30)
    by_text = {
        c.text.strip().splitlines()[0].strip(): c.metadata["section_heading"] for c in chunks
    }
    assert by_text["### Carry Forward"] == "Annual Leave > Carry Forward"
    assert by_text["## Annual Leave"] == "Annual Leave"
    carry = next(c for c in chunks if "carried forward" in c.text)
    assert carry.text.lstrip().startswith("### Carry Forward")  # heading kept in text


@M4
def test_structured_splits_long_sections_and_keeps_heading(doc):
    chunks = chunk_structured(doc, 300, 30)
    sick = [c for c in chunks if c.metadata["section_heading"] == "Sick Leave"]
    assert len(sick) > 1
    assert all(len(c.text) <= 300 for c in sick)


@M4
def test_dispatcher_rejects_unknown_strategy(doc):
    with pytest.raises(ChunkingError, match="Unknown strategy"):
        chunk_document(doc, "semantic", 500, 50)
