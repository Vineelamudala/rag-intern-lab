"""Environment-driven configuration (PROVIDED – complete infrastructure).

All settings come from environment variables, optionally loaded from a local
``.env`` file. Nothing in this module talks to the network or the database.
You should not need to change this file to complete the lab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


def load_dotenv_if_available(path: Path | None = None) -> None:
    """Load ``.env`` into ``os.environ`` if python-dotenv is installed.

    Existing environment variables always win over values in the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
        return
    load_dotenv(path or PROJECT_ROOT / ".env", override=False)


def _get_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(env: dict[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _build_database_url(env: dict[str, str]) -> str:
    if env.get("DATABASE_URL"):
        return env["DATABASE_URL"]
    user = env.get("POSTGRES_USER", "rag")
    password = env.get("POSTGRES_PASSWORD", "")
    host = env.get("POSTGRES_HOST", "localhost")
    port = env.get("POSTGRES_PORT", "5433")
    db = env.get("POSTGRES_DB", "rag_lab")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


@dataclass(frozen=True)
class Settings:
    """Typed view of all lab configuration."""

    docs_dir: Path = PROJECT_ROOT / "data" / "rag_docs"
    queries_file: Path = PROJECT_ROOT / "data" / "rag_test_queries.md"
    database_url: str = "postgresql://rag@localhost:5433/rag_lab"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 4
    min_similarity: float = 0.30

    # LiteLLM gateway (organisation-provided). Values come ONLY from the environment.
    litellm_api_base: str = ""
    litellm_api_key: str = field(default="", repr=False)
    litellm_model: str = ""
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0

    def validate(self) -> Settings:
        """Check cross-field invariants and return ``self`` for chaining."""
        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be greater than 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.embedding_dim <= 0:
            raise ConfigError("EMBEDDING_DIM must be greater than 0")
        if self.top_k <= 0:
            raise ConfigError("TOP_K must be greater than 0")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ConfigError("MIN_SIMILARITY must be between -1 and 1 (cosine similarity)")
        if self.llm_max_tokens <= 0:
            raise ConfigError("LLM_MAX_TOKENS must be greater than 0")
        if self.queries_file.resolve().is_relative_to(self.docs_dir.resolve()):
            raise ConfigError(
                f"RAG_DOCS_DIR ({self.docs_dir}) contains the evaluation file "
                f"{self.queries_file.name}. Point RAG_DOCS_DIR at data/rag_docs so the "
                "test queries can never be ingested."
            )
        return self

    def require_gateway(self) -> Settings:
        """Raise a helpful error unless all LiteLLM gateway settings are present."""
        missing = [
            name
            for name, value in (
                ("LITELLM_API_BASE", self.litellm_api_base),
                ("LITELLM_API_KEY", self.litellm_api_key),
                ("LITELLM_MODEL", self.litellm_model),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                f"LiteLLM gateway is not configured: {', '.join(missing)} missing. "
                "Ask your mentor for the organisation's gateway URL, a personal key and the "
                "model alias, and put them in .env (never commit .env)."
            )
        return self

    def with_overrides(self, **changes: object) -> Settings:
        """Return a copy with selected fields replaced (``None`` values are ignored)."""
        filtered = {k: v for k, v in changes.items() if v is not None}
        return replace(self, **filtered).validate()  # type: ignore[arg-type]


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env: dict[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    """Build :class:`Settings` from environment variables.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``. Tests pass a dict.
        use_dotenv: Load ``.env`` first (only when reading ``os.environ``).
    """
    if env is None:
        if use_dotenv:
            load_dotenv_if_available()
        env = dict(os.environ)

    return Settings(
        docs_dir=_resolve_path(env.get("RAG_DOCS_DIR", "data/rag_docs")),
        queries_file=_resolve_path(env.get("RAG_QUERIES_FILE", "data/rag_test_queries.md")),
        database_url=_build_database_url(env),
        embedding_model=env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_dim=_get_int(env, "EMBEDDING_DIM", 384),
        chunk_size=_get_int(env, "CHUNK_SIZE", 800),
        chunk_overlap=_get_int(env, "CHUNK_OVERLAP", 120),
        top_k=_get_int(env, "TOP_K", 4),
        min_similarity=_get_float(env, "MIN_SIMILARITY", 0.30),
        litellm_api_base=env.get("LITELLM_API_BASE", "").strip(),
        litellm_api_key=env.get("LITELLM_API_KEY", "").strip(),
        litellm_model=env.get("LITELLM_MODEL", "").strip(),
        llm_max_tokens=_get_int(env, "LLM_MAX_TOKENS", 1024),
        llm_temperature=_get_float(env, "LLM_TEMPERATURE", 0.0),
        llm_timeout_seconds=_get_float(env, "LLM_TIMEOUT_SECONDS", 60.0),
    ).validate()
