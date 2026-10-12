"""M8–M10 – prompt construction, LiteLLM gateway, citations and refusals.

Run: python -m pytest -m m8   (format_context, build_prompt)
     python -m pytest -m m9   (LiteLLMProvider with a mocked litellm module)
     python -m pytest -m m10  (citations, unanswerable questions, answer_question)
Live gateway smoke test (needs LITELLM_* in .env): python -m pytest -m gateway
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag_lab.answering import (
    NOT_FOUND_MESSAGE,
    NOT_FOUND_SENTINEL,
    LiteLLMProvider,
    LLMError,
    answer_question,
    build_prompt,
    build_provider,
    extract_citations,
    format_context,
    is_not_found,
    sources_for,
)
from rag_lab.config import ConfigError, load_settings
from rag_lab.vector_store import SearchFilters

from fakes import FakeRetriever, StaticResponseProvider, make_result

M8, M9, M10 = pytest.mark.m8, pytest.mark.m9, pytest.mark.m10

RESULTS = [
    make_result(0, "leave_policy.md", "Sick Leave", "Employees receive 10 days of sick leave."),
    make_result(1, "employee_faq.md", "Leave > Where do I apply for leave?", "Apply in PeopleHub."),
    make_result(2, "leave_policy.md", "Sick Leave", "A certificate is needed after 2 days."),
]


# -- M8: context and prompt ---------------------------------------------------


@M8
def test_format_context_numbers_blocks_with_metadata():
    context = format_context(RESULTS)
    for i, r in enumerate(RESULTS, start=1):
        assert f"[S{i}]" in context
        assert r.text in context
        assert r.source in context
        assert r.section_heading in context
    assert "2026-04-01" in context
    assert context.index("[S1]") < context.index("[S2]") < context.index("[S3]")


@M8
def test_build_prompt_contains_context_and_question_only():
    system, user = build_prompt("How many sick days?", RESULTS[:2])
    assert "How many sick days?" in user
    assert RESULTS[0].text in user and RESULTS[1].text in user
    assert RESULTS[2].text not in user  # only what was passed in
    assert "How many sick days?" not in system  # question belongs in the user message


@M8
def test_system_prompt_states_grounding_rules():
    system, _ = build_prompt("q", RESULTS)
    assert NOT_FOUND_SENTINEL in system
    assert "[S1]" in system or "[S" in system  # tells the model how to cite
    assert len(system) > 100  # a real instruction, not a placeholder


# -- M9: LiteLLM gateway (mocked) ---------------------------------------------


def _fake_litellm(monkeypatch, reply="Answer [S1]", error: Exception | None = None):
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    return calls


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model="team-chat-model",
        api_base="https://gateway.example.invalid",
        api_key="fake-key-for-tests",
        max_tokens=256,
        temperature=0.0,
        timeout=30.0,
    )


@M9
def test_litellm_call_uses_configured_gateway(monkeypatch):
    calls = _fake_litellm(monkeypatch)
    assert _provider().generate("SYSTEM TEXT", "USER TEXT") == "Answer [S1]"
    kwargs = calls[0]
    assert kwargs["model"].endswith("team-chat-model")
    assert kwargs["api_base"] == "https://gateway.example.invalid"
    assert kwargs["api_key"] == "fake-key-for-tests"
    assert [m["role"] for m in kwargs["messages"]] == ["system", "user"]
    assert kwargs["messages"][0]["content"] == "SYSTEM TEXT"
    assert kwargs["messages"][1]["content"] == "USER TEXT"
    assert kwargs["max_tokens"] == 256 and kwargs["temperature"] == 0.0


@M9
def test_litellm_empty_content_returns_empty_string(monkeypatch):
    _fake_litellm(monkeypatch, reply=None)
    assert _provider().generate("s", "u") == ""


@M9
def test_litellm_errors_are_wrapped_without_leaking_key(monkeypatch):
    _fake_litellm(monkeypatch, error=RuntimeError("401 Unauthorized"))
    with pytest.raises(LLMError) as info:
        _provider().generate("s", "u")
    assert "fake-key-for-tests" not in str(info.value)


@M9
def test_build_provider_requires_gateway_settings():
    with pytest.raises(ConfigError, match="LITELLM"):
        build_provider(load_settings({}))


@pytest.mark.gateway
def test_live_gateway_smoke():
    """Real call through the organisation's gateway. Uses your .env settings."""
    settings = load_settings()
    try:
        provider = build_provider(settings)
    except ConfigError as exc:
        pytest.skip(str(exc))
    reply = provider.generate("Reply with the single word OK.", "Say OK.")
    assert reply.strip()


# -- M10: citations and unanswerable questions --------------------------------


@M10
def test_extract_citations():
    assert extract_citations("A [S2]. B [S1][S2]. C [S3]") == [2, 1, 3]
    assert extract_citations("no citations here") == []
    assert extract_citations("[S10] and [S1]") == [10, 1]


@M10
@pytest.mark.parametrize("reply", ["NOT_FOUND", " NOT_FOUND. ", "not_found", ""])
def test_is_not_found_variants(reply):
    assert is_not_found(reply)


@M10
def test_is_not_found_rejects_real_answers():
    assert not is_not_found("Employees get 10 days [S1].")


@M10
def test_sources_for_dedupes_in_citation_order():
    refs = sources_for(RESULTS, [3, 2, 1])
    assert [(r.source, r.section_heading) for r in refs] == [
        ("leave_policy.md", "Sick Leave"),
        ("employee_faq.md", "Leave > Where do I apply for leave?"),
    ]
    assert refs[0].chunk_id == RESULTS[2].chunk_id


@M10
def test_cited_answer_returns_sources_from_metadata():
    provider = StaticResponseProvider("You get 10 days [S1]; apply in PeopleHub [S2].")
    result = answer_question("Sick days?", FakeRetriever(RESULTS), provider)
    assert result.found
    assert result.answer.startswith("You get 10 days")
    assert [s.source for s in result.sources] == ["leave_policy.md", "employee_faq.md"]
    assert result.retrieved == RESULTS
    assert len(provider.calls) == 1


@M10
def test_sources_only_include_cited_blocks():
    result = answer_question(
        "Where?", FakeRetriever(RESULTS), StaticResponseProvider("PeopleHub [S2].")
    )
    assert [s.chunk_id for s in result.sources] == [RESULTS[1].chunk_id]


@M10
def test_filters_are_passed_to_retriever():
    retriever = FakeRetriever(RESULTS)
    filters = SearchFilters(location="Pune")
    answer_question("q", retriever, StaticResponseProvider("x [S1]"), filters)
    assert retriever.calls[0][1] == filters


@M10
def test_no_retrieval_means_no_llm_call():
    provider = StaticResponseProvider("Invented answer [S1].")
    result = answer_question("Stock options?", FakeRetriever([]), provider)
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert provider.calls == []
    assert result.reason


@M10
@pytest.mark.parametrize("reply", [NOT_FOUND_SENTINEL, "NOT_FOUND.", "  ", ""])
def test_model_not_found_reply_is_respected(reply):
    result = answer_question("Gym?", FakeRetriever(RESULTS), StaticResponseProvider(reply))
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


@M10
def test_uncited_answer_is_rejected():
    """An LLM answering from its own knowledge must not be shown as sourced."""
    result = answer_question(
        "Vesting schedule?",
        FakeRetriever(RESULTS),
        StaticResponseProvider("Options vest over four years with a one-year cliff."),
    )
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.reason


@M10
def test_citation_to_missing_block_is_rejected():
    result = answer_question(
        "q", FakeRetriever(RESULTS[:2]), StaticResponseProvider("Yes [S1], also [S7].")
    )
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
