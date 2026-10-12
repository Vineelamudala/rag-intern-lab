"""Ask a question against one chunking strategy's index.

Examples (PowerShell, from the repository root):

    python scripts/ask.py "How many days of annual leave can I carry forward?" --retrieval-only
    python scripts/ask.py "Which days must hybrid employees be in the office?" --location Pune
    python scripts/ask.py "What is the address of the new office?" --doc-type circular --show-context

--retrieval-only works from M7 onwards (no gateway needed). Full answers need
M8-M10 and the LiteLLM gateway settings in .env.

Exit codes: 0 success, 1 error, 2 a required milestone is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab import STRATEGIES  # noqa: E402
from rag_lab.answering import LLMError, answer_question, build_provider, format_context  # noqa: E402
from rag_lab.config import ConfigError, load_settings  # noqa: E402
from rag_lab.embeddings import EmbeddingError, build_embedder  # noqa: E402
from rag_lab.retrieval import RetrievalError, Retriever  # noqa: E402
from rag_lab.vector_store import PgVectorStore, SearchFilters, SearchResult, VectorStoreError  # noqa: E402

from _common import configure_console, report_error, report_todo  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question from the indexed documents.")
    parser.add_argument("question", help="The question to answer (quote it).")
    parser.add_argument("--strategy", choices=STRATEGIES, default="structured")
    parser.add_argument("--top-k", type=int, help="Override TOP_K.")
    parser.add_argument("--min-similarity", type=float, help="Override MIN_SIMILARITY.")
    parser.add_argument("--doc-type", help="Filter by document type, e.g. policy, circular, faq.")
    parser.add_argument("--department", help="Filter by owning department, e.g. Finance.")
    parser.add_argument(
        "--location",
        help="Filter by office location. Company-wide ('All Locations') documents are included.",
    )
    parser.add_argument(
        "--exact-location",
        action="store_true",
        help="With --location, exclude company-wide documents.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only show retrieved chunks; do not call the LLM gateway.",
    )
    parser.add_argument("--show-context", action="store_true", help="Print the prompt context.")
    return parser.parse_args(argv)


def print_results(results: list[SearchResult]) -> None:
    if not results:
        print("  (no chunks above the similarity threshold)")
        return
    for i, r in enumerate(results, start=1):
        print(f"  [S{i}] {r.score:.3f}  {r.source} | {r.section_heading}")


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    try:
        settings = load_settings().with_overrides(
            top_k=args.top_k, min_similarity=args.min_similarity
        )
        filters = SearchFilters(
            doc_type=args.doc_type,
            department=args.department,
            location=args.location,
            include_all_locations=not args.exact_location,
        )
        # Fail fast on missing gateway settings before loading the model.
        provider = None if args.retrieval_only else build_provider(settings)
        embedder = build_embedder(settings)

        with PgVectorStore(settings.database_url, settings.embedding_dim) as store:
            retriever = Retriever(
                embedder=embedder,
                store=store,
                strategy=args.strategy,
                top_k=settings.top_k,
                min_similarity=settings.min_similarity,
            )
            print(f"Strategy: {args.strategy} | top_k={settings.top_k} | filters={filters}")

            if provider is None:
                results = retriever.retrieve(args.question, filters)
                print("\nRetrieved chunks:")
                print_results(results)
                if args.show_context and results:
                    print("\n" + format_context(results))
                return 0

            result = answer_question(args.question, retriever, provider, filters)

        print(f"\nQuestion: {result.question}\n")
        print(f"Answer:\n{result.answer}\n")
        if result.found:
            print("Sources:")
            for ref in result.sources:
                print(f"  - {ref.format()}")
        else:
            print(f"(not answered: {result.reason})")
        print("\nRetrieved chunks:")
        print_results(result.retrieved)
        if args.show_context and result.retrieved:
            print("\n" + format_context(result.retrieved))
        return 0
    except NotImplementedError as exc:
        return report_todo(exc)
    except (ConfigError, EmbeddingError, LLMError, RetrievalError, VectorStoreError) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
