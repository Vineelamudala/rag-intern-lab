# Submission Checklist

Copy this checklist into your pull request description and tick each item.

## Environment (M0)

- [ ] Python 3.12 venv created; `pip install -r requirements.txt` succeeded
- [ ] `.env` created from `.env.example`; `git status` does **not** list `.env`
- [ ] `docker compose ps` shows `db` as healthy; `python scripts/ingest.py --check-db` succeeds
- [ ] `python -m pytest -m m0` passes

## Task 1: Ingestion and chunking (M1–M6)

- [ ] M1: documents load with validated metadata and stable IDs (`pytest -m m1` passes)
- [ ] M2–M4: fixed, recursive and structured chunkers pass `pytest -m m2`, `-m m3` and `-m m4`
- [ ] Dry-run statistics recorded for at least three chunk size/overlap configurations
- [ ] One concrete example of a poor fixed-size split documented
- [ ] M5: local MiniLM embeddings (`pytest -m m5` passes, including model tests)
- [ ] M6: `pytest -m m6` and `pytest -m "m6 and integration"` pass
- [ ] `python scripts/ingest.py` output pasted, showing `[OK]` for all three tables
- [ ] Re-ingestion shown to be idempotent (same counts; `count(*) = count(DISTINCT chunk_id)`)
- [ ] Replacement strategy described and justified

## Task 2: Retrieval and question answering (M7–M11)

- [ ] M7: `pytest -m m7` and `pytest -m "m7 and integration"` pass
- [ ] M8: prompt iterations and the reasons for each change documented
- [ ] M9: `pytest -m m9` passes; `pytest -m gateway` passes with my gateway settings
- [ ] M10: `pytest -m m10` passes; one answered example (with sources) and one refused example included
- [ ] M11: `pytest -m m11` passes; `scripts/evaluate.py` reports attached
- [ ] All 15 queries × 3 strategies recorded in `data/rag_test_queries.md`; summary table completed
- [ ] Q13–Q15 refused with no invented details; hallucination attempts documented
- [ ] `TOP_K` / `MIN_SIMILARITY` trade-off documented, with a recommended configuration
- [ ] No expected answers hard-coded in `src/`, `scripts/` or prompts

## Quality

- [ ] `python -m pytest --strict-todos` passes (no TODOs left)
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] Provided function signatures unchanged; new code has type hints and docstrings
- [ ] All SQL values are parameterised; identifiers use `psycopg.sql.Identifier`
- [ ] No credentials, gateway keys, `.env`, model weights or database dumps committed

## Pull request

- [ ] Branch named `intern/<your-name>-rag-lab`
- [ ] PR description includes findings, this checklist, and the ingest and test output
- [ ] Mentor requested as reviewer
