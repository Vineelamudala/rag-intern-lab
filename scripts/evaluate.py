"""Run the 15 evaluation queries against one or more chunking strategies (M11).

Examples (PowerShell, from the repository root):

    python scripts/evaluate.py --retrieval-only                    # no gateway needed
    python scripts/evaluate.py --strategies fixed structured
    python scripts/evaluate.py --output eval_results.md

Expected facts in data/rag_test_queries.md are only used by YOU to judge the
answers afterwards; they are never sent to retrieval or the LLM.

Exit codes: 0 success, 1 error, 2 a required milestone is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab import STRATEGIES  # noqa: E402
from rag_lab.answering import LLMError, build_provider  # noqa: E402
from rag_lab.config import ConfigError, load_settings  # noqa: E402
from rag_lab.embeddings import EmbeddingError, build_embedder  # noqa: E402
from rag_lab.evaluation import (  # noqa: E402
    EvalRecord,
    EvaluationError,
    evaluate_strategy,
    load_eval_queries,
    render_markdown,
    summarise,
)
from rag_lab.retrieval import RetrievalError, Retriever  # noqa: E402
from rag_lab.vector_store import PgVectorStore, VectorStoreError  # noqa: E402

from _common import configure_console, report_error, report_todo  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers per strategy.")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--top-k", type=int, help="Override TOP_K.")
    parser.add_argument("--min-similarity", type=float, help="Override MIN_SIMILARITY.")
    parser.add_argument("--retrieval-only", action="store_true", help="Do not call the gateway.")
    parser.add_argument("--output", type=Path, help="Also write the Markdown report to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    try:
        settings = load_settings().with_overrides(
            top_k=args.top_k, min_similarity=args.min_similarity
        )
        queries = load_eval_queries(settings.queries_file)
        print(f"Loaded {len(queries)} queries from {settings.queries_file}")
        provider = None if args.retrieval_only else build_provider(settings)
        embedder = build_embedder(settings)

        records: list[EvalRecord] = []
        with PgVectorStore(settings.database_url, settings.embedding_dim) as store:
            for strategy in args.strategies:
                retriever = Retriever(
                    embedder, store, strategy, settings.top_k, settings.min_similarity
                )
                print(f"Evaluating {strategy} ...")
                records.extend(evaluate_strategy(queries, retriever, provider))

        report = render_markdown(records, summarise(records))
        print("\n" + report)
        if args.output:
            args.output.write_text(report, encoding="utf-8")
            print(f"Report written to {args.output}")
        return 0
    except NotImplementedError as exc:
        return report_todo(exc)
    except (
        ConfigError,
        EmbeddingError,
        EvaluationError,
        LLMError,
        RetrievalError,
        VectorStoreError,
    ) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
