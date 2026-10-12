"""Milestones M6–M7 – PostgreSQL + pgvector storage and similarity search.

PROVIDED (working infrastructure):
* :class:`PgVectorStore` connection management with helpful errors,
  ``CREATE EXTENSION vector`` and pgvector type registration.
* Schema helpers :func:`create_table_sql` / :func:`create_index_sql` and
  :meth:`PgVectorStore.init_schema` (used by ``scripts/ingest.py --init-db``).
* Table-name whitelist (:func:`table_for`) and the :class:`SearchFilters` /
  :class:`SearchResult` types.

YOUR WORK (raise ``NotImplementedError`` until you implement them):
* M6 – :meth:`PgVectorStore.replace_chunks`, :meth:`count`, :meth:`chunk_ids`,
  :meth:`embedding_models` (persistence + validation, idempotent re-ingestion).
* M7 – :func:`build_search_query` and :meth:`PgVectorStore.search`.

SQL rules for this lab:
* Values (filters, vectors, top_k) are ALWAYS query parameters (``%s`` or
  ``%(name)s``) – never f-strings or ``+`` concatenation.
* Identifiers (table/index names) come only from :func:`table_for` /
  ``self._table(...)`` and are composed with ``psycopg.sql.Identifier``.

Tests: ``python -m pytest -m m6`` and ``-m m7`` (unit), plus
``python -m pytest -m "integration and m6"`` with the database running.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_lab import STRATEGIES, TABLE_NAMES, todo
from rag_lab.chunkers import Chunk

_PREFIX_RE = re.compile(r"[a-z0-9_]*")

GLOBAL_LOCATION = "All Locations"
"""Documents tagged with this location apply to every office."""


class VectorStoreError(RuntimeError):
    """Raised for storage problems (missing tables, SQL errors, bad input)."""


class DatabaseConnectionError(VectorStoreError):
    """Raised when PostgreSQL cannot be reached."""


@dataclass(frozen=True)
class SearchFilters:
    """Optional metadata filters applied together with similarity search.

    Semantics your implementation must follow (all comparisons case-insensitive):
    * ``doc_type`` / ``department`` – exact match.
    * ``location`` – match chunks for that office; when ``include_all_locations``
      is true ALSO match company-wide chunks whose location is
      :data:`GLOBAL_LOCATION`.
    * ``None`` means "do not filter on this field".
    """

    doc_type: str | None = None
    department: str | None = None
    location: str | None = None
    include_all_locations: bool = True

    def is_empty(self) -> bool:
        return not (self.doc_type or self.department or self.location)


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by similarity search.

    ``score`` is cosine SIMILARITY (higher is better, 1.0 = identical direction).
    pgvector's ``<=>`` operator returns cosine DISTANCE, so similarity = 1 - distance.
    ``metadata`` holds at least the keys of ``REQUIRED_CHUNK_METADATA`` as strings.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def section_heading(self) -> str:
        return self.metadata.get("section_heading", "")


class VectorStore(Protocol):
    """Interface used by ingestion and retrieval (PROVIDED).

    The unit tests use an in-memory fake that implements this protocol.
    """

    def replace_chunks(
        self,
        strategy: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> int: ...

    def count(self, strategy: str) -> int: ...

    def chunk_ids(self, strategy: str) -> list[str]: ...

    def embedding_models(self, strategy: str) -> set[str]: ...

    def search(
        self,
        strategy: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]: ...


def table_for(strategy: str) -> str:
    """Return the whitelisted table name for ``strategy`` (PROVIDED)."""
    try:
        return TABLE_NAMES[strategy]
    except KeyError as exc:
        raise VectorStoreError(
            f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}"
        ) from exc


# --------------------------------------------------------------------------- #
# Schema (PROVIDED)
# --------------------------------------------------------------------------- #

COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "content",
    "content_hash",
    "source",
    "title",
    "doc_type",
    "effective_date",
    "section_heading",
    "department",
    "location",
    "embedding_model",
    "embedding",
)
"""Insertable columns, in table order (``ingested_at`` has a default)."""


def create_table_sql(table: str, dimension: int, if_not_exists: bool = False) -> Any:
    """Return the ``CREATE TABLE`` statement for a chunk table."""
    from psycopg import sql

    if dimension <= 0:
        raise VectorStoreError("Vector dimension must be positive")
    template = """
        CREATE TABLE {exists}{table} (
            chunk_id        TEXT PRIMARY KEY,
            doc_id          TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            content         TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            source          TEXT NOT NULL,
            title           TEXT NOT NULL,
            doc_type        TEXT NOT NULL,
            effective_date  DATE NOT NULL,
            section_heading TEXT NOT NULL,
            department      TEXT NOT NULL,
            location        TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding       vector({dim}) NOT NULL,
            ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (doc_id, chunk_index)
        )
    """
    return sql.SQL(template).format(
        exists=sql.SQL("IF NOT EXISTS " if if_not_exists else ""),
        table=sql.Identifier(table),
        dim=sql.Literal(int(dimension)),
    )


def create_index_sql(table: str) -> list[Any]:
    """Return index statements: HNSW (cosine) on ``embedding``, B-tree on filter columns."""
    from psycopg import sql

    statements = [
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding vector_cosine_ops)"
        ).format(name=sql.Identifier(f"{table}_embedding_hnsw_idx"), table=sql.Identifier(table))
    ]
    for column in ("doc_type", "department", "location"):
        statements.append(
            sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} (lower({col}))").format(
                name=sql.Identifier(f"{table}_{column}_idx"),
                table=sql.Identifier(table),
                col=sql.Identifier(column),
            )
        )
    return statements


def to_pgvector(values: Sequence[float]) -> Any:
    """Convert a Python vector to the type registered by ``pgvector.psycopg``."""
    import numpy as np

    return np.asarray(values, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Search SQL – M7
# --------------------------------------------------------------------------- #


def build_search_query(
    table: str, query_embedding: Any, top_k: int, filters: SearchFilters | None = None
) -> tuple[Any, dict[str, Any]]:
    """M7 – build a parameterised cosine-similarity query.

    Returns ``(query, params)`` where ``query`` is a ``psycopg.sql.Composed`` and
    ``params`` is a dict for named placeholders (``%(name)s``).

    Requirements:
    * Select ``chunk_id``, ``content``, the metadata columns and the similarity
      score (``1 - (embedding <=> query)``).
    * Add a ``WHERE`` clause only for the filters that are set, following the
      semantics documented on :class:`SearchFilters`.
    * Order by distance (closest first) with a deterministic tie-breaker, and
      limit to ``top_k`` rows.
    * Filter values, the query vector and ``top_k`` must appear ONLY in
      ``params`` – never inside the SQL text. A unit test feeds an SQL-injection
      string as a filter value.
    * Raise :class:`VectorStoreError` if ``top_k <= 0``.

    Useful API: ``psycopg.sql.SQL``, ``sql.Identifier``, ``sql.SQL(" AND ").join(...)``.
    """
    raise todo("M7", "build_search_query")


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class PgVectorStore:
    """pgvector-backed :class:`VectorStore`. Use as a context manager.

    ``table_prefix`` lets integration tests use ``test_rag_chunks_*`` tables so
    your real tables are never overwritten.
    """

    def __init__(
        self,
        database_url: str,
        dimension: int,
        connect_timeout: int = 5,
        table_prefix: str = "",
    ) -> None:
        if not database_url:
            raise VectorStoreError("DATABASE_URL is not configured. Copy .env.example to .env.")
        if not _PREFIX_RE.fullmatch(table_prefix):
            raise VectorStoreError("table_prefix may contain only lowercase letters, digits and _")
        self.database_url = database_url
        self.table_prefix = table_prefix
        self.dimension = dimension
        self.connect_timeout = connect_timeout
        self._conn: Any = None

    # -- connection management (PROVIDED) -------------------------------------

    def connect(self) -> Any:
        """Open (once) and return an autocommit ``psycopg`` connection.

        Autocommit means each standalone statement commits immediately, while
        ``with conn.transaction():`` blocks run as one real transaction –
        use that for anything that must be all-or-nothing.
        """
        if self._conn is not None:
            return self._conn
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise VectorStoreError(
                "psycopg / pgvector are not installed. Run `pip install -r requirements.txt`."
            ) from exc
        try:
            conn = psycopg.connect(
                self.database_url, connect_timeout=self.connect_timeout, autocommit=True
            )
        except psycopg.OperationalError as exc:
            raise DatabaseConnectionError(
                "Could not connect to PostgreSQL. Is the database running? "
                "Start it with `docker compose up -d` and check DATABASE_URL / POSTGRES_* in .env. "
                f"Driver error: {exc}"
            ) from exc
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
        except psycopg.Error as exc:
            conn.close()
            raise VectorStoreError(
                "The pgvector extension is not available in this database. "
                "Use the pgvector/pgvector Docker image from compose.yaml. "
                f"Driver error: {exc}"
            ) from exc
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PgVectorStore:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- helpers (PROVIDED) ---------------------------------------------------

    def _table(self, strategy: str) -> str:
        """Whitelisted table name for ``strategy``, including ``table_prefix``."""
        return f"{self.table_prefix}{table_for(strategy)}"

    def table_exists(self, strategy: str) -> bool:
        row = self.connect().execute("SELECT to_regclass(%s)", (self._table(strategy),)).fetchone()
        return row is not None and row[0] is not None

    def require_table(self, strategy: str) -> str:
        """Return the table name or raise a helpful error if it does not exist."""
        table = self._table(strategy)
        if not self.table_exists(strategy):
            raise VectorStoreError(
                f"Table {table} does not exist. Run `python scripts/ingest.py` first."
            )
        return table

    def init_schema(self, strategies: Sequence[str] = STRATEGIES) -> list[str]:
        """Create empty tables and indexes if they do not exist. Returns table names."""
        conn = self.connect()
        tables = []
        with conn.transaction():
            for strategy in strategies:
                table = self._table(strategy)
                conn.execute(create_table_sql(table, self.dimension, if_not_exists=True))
                for stmt in create_index_sql(table):
                    conn.execute(stmt)
                tables.append(table)
        return tables

    # -- persistence – M6 -----------------------------------------------------

    def replace_chunks(
        self,
        strategy: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> int:
        """M6 – store ``chunks`` and their vectors as the complete content of the
        strategy's table. Returns the number of rows inserted.

        Requirements:
        * Validate inputs first and raise :class:`VectorStoreError` if
          ``len(chunks) != len(embeddings)``, if chunk IDs repeat, or if any
          vector's length differs from ``self.dimension``.
        * Re-running with the same chunks must leave EXACTLY the same rows –
          no duplicates, no leftovers from an earlier run with more chunks.
        * A failure part-way must not leave a half-written table (use
          ``with conn.transaction():``).
        * Choose and document your replacement strategy (e.g. drop + recreate,
          delete + insert, or upsert + delete stale) in your pull request.
        * Use :data:`COLUMNS`, :func:`to_pgvector`, ``psycopg.sql`` composition and
          parameter placeholders. ``cursor.executemany`` is useful.
        * Wrap ``psycopg.Error`` in :class:`VectorStoreError` with the table name.
        """
        raise todo("M6", "PgVectorStore.replace_chunks")

    def count(self, strategy: str) -> int:
        """M6 – number of rows in the strategy's table (use :meth:`require_table`)."""
        raise todo("M6", "PgVectorStore.count")

    def chunk_ids(self, strategy: str) -> list[str]:
        """M6 – all chunk IDs in the strategy's table, sorted ascending."""
        raise todo("M6", "PgVectorStore.chunk_ids")

    def embedding_models(self, strategy: str) -> set[str]:
        """M6 – distinct ``embedding_model`` values stored in the strategy's table."""
        raise todo("M6", "PgVectorStore.embedding_models")

    # -- retrieval – M7 -------------------------------------------------------

    def search(
        self,
        strategy: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """M7 – run :func:`build_search_query` and convert rows to :class:`SearchResult`.

        Requirements:
        * Raise :class:`VectorStoreError` if the query vector's length differs from
          ``self.dimension`` or the table does not exist.
        * Results are sorted by ``score`` descending, at most ``top_k`` of them.
        * ``metadata`` values are strings (convert ``effective_date`` to ISO format).
        * Wrap ``psycopg.Error`` in :class:`VectorStoreError`.
        """
        raise todo("M7", "PgVectorStore.search")
