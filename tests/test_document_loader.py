"""M1 – document loading and metadata extraction.

Run: python -m pytest -m m1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.document_loader import (
    REQUIRED_METADATA,
    DocumentLoadError,
    load_document,
    load_documents,
    make_doc_id,
    parse_front_matter,
)

from conftest import DOCS_DIR, EXPECTED_SOURCES, QUERIES_FILE

pytestmark = pytest.mark.m1

VALID = (
    "---\ntitle: A\ndoc_type: Policy\neffective_date: 2026-01-01\n"
    "department: HR\nlocation: All Locations\nversion: 1.0\n---\n# A\n\nbody\n"
)


# -- make_doc_id --------------------------------------------------------------


def test_doc_id_is_deterministic_and_case_insensitive():
    assert make_doc_id("leave_policy.md") == make_doc_id("leave_policy.md")
    assert make_doc_id("Leave_Policy.md") == make_doc_id("leave_policy.md")
    assert make_doc_id("a.md") != make_doc_id("b.md")
    assert isinstance(make_doc_id("a.md"), str) and make_doc_id("a.md")


# -- parse_front_matter -------------------------------------------------------


def test_parse_front_matter_basic():
    meta, body = parse_front_matter(VALID)
    assert meta["title"] == "A"
    assert meta["version"] == "1.0"
    assert body.lstrip().startswith("# A")
    assert "doc_type:" not in body


def test_parse_front_matter_quotes_and_colons():
    meta, body = parse_front_matter('---\ntitle: "Circular: Move"\nDoc_Type: circular\n---\nBody\n')
    assert meta == {"title": "Circular: Move", "doc_type": "circular"}
    assert body.strip() == "Body"


def test_parse_front_matter_absent():
    assert parse_front_matter("# Just a heading\n") == ({}, "# Just a heading\n")


def test_parse_front_matter_malformed_line():
    with pytest.raises(DocumentLoadError):
        parse_front_matter("---\ntitle A\n---\nbody\n")


# -- load_document / load_documents ------------------------------------------


def test_load_document_fields(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text(VALID, encoding="utf-8")
    doc = load_document(path)
    assert doc.source == "a.md"
    assert doc.doc_id == make_doc_id("a.md")
    assert doc.doc_type == "policy"  # lower-cased
    assert doc.effective_date == "2026-01-01"
    assert doc.extra_metadata == {"version": "1.0"}
    assert not doc.text.startswith("---")
    assert doc.content_hash


def test_content_hash_tracks_content_but_id_does_not(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text(VALID, encoding="utf-8")
    first = load_document(path)
    path.write_text(VALID.replace("body", "changed body"), encoding="utf-8")
    second = load_document(path)
    assert first.doc_id == second.doc_id
    assert first.content_hash != second.content_hash


def test_loads_all_six_documents(documents):
    assert {d.source for d in documents} == EXPECTED_SOURCES
    assert [d.source for d in documents] == sorted(d.source for d in documents)


def test_every_document_has_required_metadata(documents):
    for doc in documents:
        meta = doc.metadata()
        for key in REQUIRED_METADATA:
            assert meta[key], f"{doc.source} missing {key}"


def test_metadata_values_support_filtering(documents):
    by_source = {d.source: d for d in documents}
    assert by_source["hr_circular.md"].doc_type == "circular"
    assert by_source["hr_circular.md"].location == "Hyderabad"
    assert by_source["hr_circular.md"].title.startswith("HR Circular HR/2026/014")
    assert by_source["travel_expense_policy.md"].department == "Finance"
    assert by_source["remote_work_policy.md"].effective_date == "2026-07-01"
    assert {d.doc_type for d in documents} == {"policy", "circular", "faq", "calendar"}


def test_doc_ids_unique_and_stable(documents):
    again = load_documents(DOCS_DIR)
    assert [d.doc_id for d in documents] == [d.doc_id for d in again]
    assert len({d.doc_id for d in documents}) == len(documents)


def test_query_file_is_never_loaded(documents):
    assert QUERIES_FILE.name not in {d.source for d in documents}


def test_ignores_unsupported_files(tmp_path: Path):
    (tmp_path / "a.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.docx").write_bytes(b"PK")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text(VALID, encoding="utf-8")
    assert [d.source for d in load_documents(tmp_path)] == ["a.md"]


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(DocumentLoadError):
        load_documents(tmp_path / "nope")


def test_empty_directory_raises(tmp_path: Path):
    with pytest.raises(DocumentLoadError):
        load_documents(tmp_path)


def test_missing_metadata_names_file_and_key(tmp_path: Path):
    (tmp_path / "bad.md").write_text("---\ntitle: Bad\n---\n# Bad\n", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match=r"bad\.md.*department|department.*bad\.md"):
        load_documents(tmp_path)


def test_invalid_date_raises(tmp_path: Path):
    (tmp_path / "bad.md").write_text(VALID.replace("2026-01-01", "01/04/2026"), encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="bad.md"):
        load_documents(tmp_path)
