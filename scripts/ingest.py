"""Ingest data/rag_docs/ into the three pgvector tables.

Examples (PowerShell, from the repository root):

    python scripts/ingest.py --check-db           # M0: connectivity only
    python scripts/ingest.py --init-db            # M0: create extension + empty tables
    python scripts/ingest.py --dry-run            # M1-M4: load + chunk, no model/DB
    python scripts/ingest.py                      # M6: full ingestion, all strategies
    python scripts/ingest.py --strategies structured --chunk-size 600 --chunk-overlap 80

Exit codes: 0 success, 1 error, 2 a required milestone is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab import STRATEGIES  # noqa: E402
from rag_lab.config import ConfigError, Settings, load_settings  # noqa: E402
from rag_lab.document_loader import DocumentLoadError, load_documents  # noqa: E402
from rag_lab.embeddings import EmbeddingError, build_embedder  # noqa: E402
from rag_lab.ingestion import IngestionError, build_chunks, run_ingestion, summarise_chunks  # noqa: E402
from rag_lab.vector_store import PgVectorStore, VectorStoreError  # noqa: E402

from _common import configure_console, report_error, report_todo  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk, embed and store the lab documents.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Chunking strategies to ingest (default: all three).",
    )
    parser.add_argument("--chunk-size", type=int, help="Override CHUNK_SIZE (characters).")
    parser.add_argument("--chunk-overlap", type=int, help="Override CHUNK_OVERLAP (characters).")
    parser.add_argument("--docs-dir", type=Path, help="Override RAG_DOCS_DIR.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-db", action="store_true", help="Only test the database connection.")
    mode.add_argument(
        "--init-db",
        action="store_true",
        help="Create the pgvector extension and empty tables/indexes, then exit.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only; print statistics without embedding or touching the database.",
    )
    return parser.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> Settings:
    return load_settings().with_overrides(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        docs_dir=args.docs_dir.resolve() if args.docs_dir else None,
    )


def dry_run(settings: Settings, strategies: list[str]) -> None:
    documents = load_documents(settings.docs_dir)
    print(f"Loaded {len(documents)} documents from {settings.docs_dir}")
    for strategy in strategies:
        chunks = build_chunks(documents, strategy, settings.chunk_size, settings.chunk_overlap)
        avg, smallest, largest = summarise_chunks(chunks)
        print(
            f"{strategy:<11} chunks={len(chunks):<4} avg_chars={avg:7.1f} "
            f"min={smallest:<4} max={largest}"
        )


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    try:
        settings = resolve_settings(args)
        print(
            f"chunk_size={settings.chunk_size} chunk_overlap={settings.chunk_overlap} "
            f"embedding_model={settings.embedding_model} dim={settings.embedding_dim}"
        )

        if args.dry_run:
            dry_run(settings, args.strategies)
            return 0

        with PgVectorStore(settings.database_url, settings.embedding_dim) as store:
            if args.check_db:
                version = (
                    store.connect()
                    .execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    .fetchone()
                )
                print(f"Connected. pgvector extension version: {version[0] if version else '?'}")
                return 0
            if args.init_db:
                tables = store.init_schema(args.strategies)
                print(f"Initialised tables: {', '.join(tables)}")
                return 0

            documents = load_documents(settings.docs_dir)
            print(f"Loaded {len(documents)} documents from {settings.docs_dir}")
            embedder = build_embedder(settings)
            print(f"Embedding model: {embedder.model_name} ({embedder.dimension} dims)")
            reports = run_ingestion(
                documents,
                embedder,
                store,
                settings.chunk_size,
                settings.chunk_overlap,
                args.strategies,
            )

        print()
        for report in reports:
            print(report.format_row())
        print("\nAll generated and persisted counts match.")
        return 0
    except NotImplementedError as exc:
        return report_todo(exc)
    except (
        ConfigError,
        DocumentLoadError,
        EmbeddingError,
        IngestionError,
        VectorStoreError,
    ) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
