"""M11 – evaluation across chunking strategies (with fakes – no DB or gateway).

Run: python -m pytest -m m11
"""

from __future__ import annotations

import pytest

from rag_lab.evaluation import (
    EvalRecord,
    evaluate_strategy,
    is_retrieval_hit,
    load_eval_queries,
    render_markdown,
    summarise,
)

from conftest import QUERIES_FILE
from fakes import FakeRetriever, StaticResponseProvider, make_result

pytestmark = pytest.mark.m11


@pytest.fixture
def queries():
    return {q.query_id: q for q in load_eval_queries(QUERIES_FILE)}


def test_retrieval_hit(queries):
    hit = [make_result(0, "leave_policy.md", "Annual Leave > Entitlement", "18 days")]
    miss = [make_result(0, "holiday_calendar.md", "About", "calendar")]
    assert is_retrieval_hit(queries["Q01"], hit) is True
    assert is_retrieval_hit(queries["Q01"], miss) is False
    assert is_retrieval_hit(queries["Q01"], []) is False
    assert is_retrieval_hit(queries["Q13"], hit) is None  # unanswerable


class _StrategyRetriever(FakeRetriever):
    strategy = "fixed"


def test_evaluate_strategy_retrieval_only(queries):
    retriever = _StrategyRetriever([make_result(0, "leave_policy.md", "Annual Leave", "18 days")])
    selected = [queries["Q01"], queries["Q07"], queries["Q13"]]
    records = evaluate_strategy(selected, retriever)
    assert [r.query_id for r in records] == ["Q01", "Q07", "Q13"]
    assert all(r.strategy == "fixed" and r.answer is None for r in records)
    assert records[0].retrieved == [("leave_policy.md", "Annual Leave")]
    assert [r.retrieval_hit for r in records] == [True, False, None]
    assert retriever.calls[1][1] == queries["Q07"].filters  # each query's own filters


def test_evaluate_strategy_with_provider(queries):
    retriever = _StrategyRetriever([make_result(0, "leave_policy.md", "Annual Leave", "18 days")])
    provider = StaticResponseProvider("18 days [S1].")
    records = evaluate_strategy([queries["Q01"]], retriever, provider)
    assert records[0].answer is not None and records[0].answer.found


def test_one_failing_query_does_not_stop_the_run(queries):
    class Flaky(_StrategyRetriever):
        def retrieve(self, question, filters=None):
            if "annual leave" in question.lower():
                raise RuntimeError("boom")
            return super().retrieve(question, filters)

    records = evaluate_strategy([queries["Q01"], queries["Q02"]], Flaky([]))
    assert len(records) == 2
    assert "boom" in records[0].notes


def _record(qid, strategy, hit, found=None):
    from rag_lab.answering import NOT_FOUND_MESSAGE, AnswerResult

    answer = None
    if found is not None:
        answer = AnswerResult("q", "a" if found else NOT_FOUND_MESSAGE, found)
    return EvalRecord(qid, strategy, [], hit, answer)


def test_summarise_counts_per_strategy():
    records = [
        _record("Q01", "fixed", True, True),
        _record("Q02", "fixed", False, False),
        _record("Q13", "fixed", None, False),
        _record("Q14", "fixed", None, True),
        _record("Q01", "structured", True, True),
    ]
    summary = summarise(records)
    assert summary["fixed"]["retrieval_hits"] == 1
    assert summary["fixed"]["answerable"] == 2
    assert summary["fixed"]["answered"] == 1
    assert summary["fixed"]["correct_refusals"] == 1
    assert summary["fixed"]["unanswerable"] == 2
    assert summary["structured"]["retrieval_hits"] == 1


def test_render_markdown_mentions_every_query_and_strategy():
    records = [_record("Q01", "fixed", True), _record("Q01", "recursive", False)]
    text = render_markdown(records, summarise(records))
    assert "Q01" in text and "fixed" in text and "recursive" in text
    assert "|" in text  # Markdown table
