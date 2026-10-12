"""Milestones M8–M10 – context construction, LiteLLM generation, grounding.

Answer generation goes ONLY through the organisation-provided LiteLLM gateway.
Its URL, key and model alias are read from the environment (see
``.env.example`` and :meth:`rag_lab.config.Settings.require_gateway`).

Overall contract of :func:`answer_question`:
* It never returns an answer that is not supported by retrieved chunks.
* When the answer is not in the documents it returns :data:`NOT_FOUND_MESSAGE`
  with ``found=False`` and no sources.
* Sources shown to the user come from stored chunk metadata (filename +
  section heading), never from text the model wrote.

PROVIDED: constants, result types, the provider protocol, provider factory.
YOUR WORK:
* M8  – :func:`format_context`, :func:`build_prompt` (prompt engineering)
* M9  – :meth:`LiteLLMProvider.generate`
* M10 – :func:`extract_citations`, :func:`is_not_found`, :func:`sources_for`,
  :func:`answer_question`

Tests: ``python -m pytest -m m8`` / ``-m m9`` / ``-m m10``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_lab import todo
from rag_lab.config import Settings
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import SearchFilters, SearchResult

NOT_FOUND_SENTINEL = "NOT_FOUND"
"""What the model must reply when the context is insufficient."""

NOT_FOUND_MESSAGE = "I could not find the answer in the available documents."
"""What the user sees when no supported answer exists."""


class LLMError(RuntimeError):
    """Raised when the LLM gateway call fails."""


class LLMProvider(Protocol):
    """Minimal provider interface: one system prompt + one user message -> text.

    Unit tests use a fake provider that returns a fixed string.
    """

    name: str

    def generate(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class SourceRef:
    """A citation shown to the user (PROVIDED)."""

    source: str
    section_heading: str
    chunk_id: str
    score: float

    def format(self) -> str:
        return f"{self.source} | {self.section_heading} (score {self.score:.3f})"


@dataclass(frozen=True)
class AnswerResult:
    """Result of :func:`answer_question` (PROVIDED).

    ``reason`` explains why an answer was not given (useful while debugging and
    in your evaluation notes); it is empty when ``found`` is true.
    """

    question: str
    answer: str
    found: bool
    sources: list[SourceRef] = field(default_factory=list)
    retrieved: list[SearchResult] = field(default_factory=list)
    reason: str = ""


# --------------------------------------------------------------------------- #
# M8 – context construction and prompt engineering
# --------------------------------------------------------------------------- #


def format_context(results: list[SearchResult]) -> str:
    """Render retrieved chunks as numbered context blocks.

    Requirements:
    * Block *i* (1-based) is labelled ``[S{i}]`` and includes the chunk's source
      filename, section heading and effective date, followed by the chunk text.
    * Blocks appear in the order of ``results``.
    * Use a clear separator between blocks so the model can tell them apart.

    ``scripts/ask.py --show-context`` prints this, which helps with debugging.
    """
    raise todo("M8", "format_context")


def build_prompt(question: str, results: list[SearchResult]) -> tuple[str, str]:
    """Return ``(system_prompt, user_message)`` for the LLM.

    Requirements:
    * The user message contains ONLY the numbered context blocks
      (:func:`format_context`) and the question – no outside material.
    * The system prompt instructs the model to:
      - answer only from the context blocks, without outside knowledge;
      - cite the supporting block(s) for every fact as ``[S1]``, ``[S2]`` ...;
      - reply with exactly :data:`NOT_FOUND_SENTINEL` when the context does not
        contain the answer;
      - prefer the block with the later effective date if blocks disagree.

    Iterate on the wording – record what you changed and why in your report.
    """
    raise todo("M8", "build_prompt")


# --------------------------------------------------------------------------- #
# M9 – LiteLLM gateway
# --------------------------------------------------------------------------- #


class LiteLLMProvider:
    """Chat completions through the organisation's LiteLLM gateway."""

    name = "litellm"

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        """Send one chat completion request and return the reply text.

        Requirements:
        * ``import litellm`` INSIDE this method (so the rest of the package and
          the unit tests work without contacting the gateway) and call
          ``litellm.completion(...)`` with:
          - ``model`` – the configured gateway model alias (``self.model``).
            When the litellm SDK talks to a LiteLLM *proxy*, the alias usually
            needs a provider prefix such as ``litellm_proxy/``; confirm the
            exact form in the LiteLLM docs and with your mentor;
          - ``api_base`` and ``api_key`` from this object (never hard-code them);
          - ``messages`` – a ``system`` message followed by a ``user`` message;
          - ``max_tokens``, ``temperature`` and ``timeout``.
        * Return the text of the first choice (``response.choices[0].message.content``),
          or ``""`` if it is empty/None.
        * Catch exceptions from litellm and raise :class:`LLMError` with a short,
          useful message (model, gateway URL, original error). Never print or
          include the API key in messages.
        """
        raise todo("M9", "LiteLLMProvider.generate")


def build_provider(settings: Settings) -> LiteLLMProvider:
    """Create the gateway provider from settings (PROVIDED).

    Raises :class:`rag_lab.config.ConfigError` naming every missing
    ``LITELLM_*`` variable.
    """
    settings.require_gateway()
    return LiteLLMProvider(
        model=settings.litellm_model,
        api_base=settings.litellm_api_base,
        api_key=settings.litellm_api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )


# --------------------------------------------------------------------------- #
# M10 – citations and unanswerable questions
# --------------------------------------------------------------------------- #


def extract_citations(answer: str) -> list[int]:
    """Return the block numbers cited as ``[S<n>]`` in ``answer``.

    Numbers are returned in order of first appearance, without duplicates.
    ``"A [S2]. B [S1][S2]."`` → ``[2, 1]``; no citations → ``[]``.
    """
    raise todo("M10", "extract_citations")


def is_not_found(answer: str) -> bool:
    """True if the model's reply means "not in the context".

    Must accept :data:`NOT_FOUND_SENTINEL` with surrounding whitespace and an
    optional trailing period, case-insensitively. An empty reply also counts as
    not found.
    """
    raise todo("M10", "is_not_found")


def sources_for(results: list[SearchResult], cited: list[int]) -> list[SourceRef]:
    """Build citations for the cited block numbers (1-based indexes into ``results``).

    Only cited blocks are included, in citation order, with duplicate
    ``(source, section_heading)`` pairs removed.
    """
    raise todo("M10", "sources_for")


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    filters: SearchFilters | None = None,
) -> AnswerResult:
    """Retrieve, prompt, generate and validate.

    Required behaviour (each point has a unit test):
    1. Retrieve with ``retriever.retrieve(question, filters)``. If nothing is
       returned, do NOT call the provider; return the not-found result.
    2. Build the prompt with :func:`build_prompt` and call ``provider.generate``.
    3. If :func:`is_not_found` → not-found result.
    4. If the reply cites no blocks → not-found result (an uncited answer may come
       from the model's own knowledge and must never be shown as sourced).
    5. If the reply cites a block number that does not exist → not-found result.
    6. Otherwise return ``found=True``, the reply text, and :func:`sources_for`.

    Every not-found result uses :data:`NOT_FOUND_MESSAGE`, ``found=False``, no
    sources, the retrieved chunks, and a short ``reason``.
    """
    raise todo("M10", "answer_question")
