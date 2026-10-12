# RAG Intern Training Lab (Starter)

In this lab you **build** a complete Retrieval-Augmented Generation (RAG)
pipeline yourself: load documents, chunk them three ways, embed them locally,
store them in PostgreSQL + pgvector, retrieve with metadata filters, and
generate grounded, cited answers through the organisation's LiteLLM gateway.
Answers must refuse rather than invent facts.

This repository is a **starter**. The infrastructure works; the RAG pipeline
does not exist yet. Most core functions raise `NotImplementedError` until you
implement them, milestone by milestone.

> All documents in `data/rag_docs/` describe a **fictional** company,
> *Lumen Ridge Technologies*. They are training data, not real policy.

---

## 1. Objectives and learning outcomes

By the end of the lab you will be able to:

1. Parse documents and metadata robustly, with useful error messages.
2. Implement and compare **fixed-size**, **recursive** and **structure-aware**
   chunking, and explain their trade-offs with measurements.
3. Generate **local embeddings** with `sentence-transformers/all-MiniLM-L6-v2`.
4. Persist vectors in **pgvector** with **deterministic, idempotent** ingestion
   and verified counts.
5. Write **parameterised** similarity-search SQL with **metadata filters**.
6. Engineer a **grounded prompt**, call an LLM through **LiteLLM**, and validate
   citations.
7. Handle **unanswerable** questions without inventing an answer.
8. **Evaluate** a RAG system across configurations with a fixed query set.

## 2. Architecture

```
INGESTION  (scripts/ingest.py)          no LLM involved
  data/rag_docs/*.md ─► load + metadata ─► chunk (fixed | recursive | structured)
                     ─► local embeddings (MiniLM, 384-d) ─► pgvector tables

QUESTION ANSWERING  (scripts/ask.py)
  question ─► local embedding ─► similarity search + filters (one strategy table)
           ─► grounded prompt ─► LiteLLM gateway ─► validated answer + citations

EVALUATION  (scripts/evaluate.py)
  data/rag_test_queries.md (15 queries, never ingested) ─► per-strategy results
```

Constraints:
- Embeddings are **local only** (Sentence Transformers). No embedding API.
- Answer generation goes **only** through the organisation-provided **LiteLLM
  gateway**. Its settings come from `.env`.
- Storage is **PostgreSQL + pgvector**, with three tables: `rag_chunks_fixed`,
  `rag_chunks_recursive` and `rag_chunks_structured`.

## 3. Repository structure

```
rag-intern-lab/
├── README.md                    ← you are here
├── requirements.txt  pyproject.toml  .env.example  .gitignore  compose.yaml
├── db/init/                     ← enables the vector extension on first start
├── data/
│   ├── rag_docs/                ← the ONLY folder that is ingested (6 documents)
│   └── rag_test_queries.md      ← 15 evaluation queries (NEVER ingested)
├── src/rag_lab/
│   ├── __init__.py              PROVIDED  strategies, table names, todo() helper
│   ├── config.py                PROVIDED  settings from .env, validation
│   ├── document_loader.py       M1        loading + metadata
│   ├── chunkers.py              M2–M4     three chunking strategies
│   ├── embeddings.py            M5        local embeddings
│   ├── vector_store.py          M6–M7     connection/schema PROVIDED; storage + search TODO
│   ├── ingestion.py             M6        pipeline + persistence validation
│   ├── retrieval.py             M7        retrieval with threshold
│   ├── answering.py             M8–M10    prompt, LiteLLM, citations, refusals
│   └── evaluation.py            M11       query-file parser PROVIDED; scoring TODO
├── scripts/                     PROVIDED  CLI entry points (ingest, ask, evaluate)
├── tests/                       PROVIDED  acceptance tests per milestone + test doubles
└── docs/
    ├── milestones.md            ← step-by-step plan: start here after setup
    ├── lab_guide.md             ← concepts behind each stage
    └── submission_checklist.md
```

**PROVIDED vs TODO.** Every module docstring says what is provided. Every
unfinished function raises
`NotImplementedError("[Mx] <name> is not implemented yet ...")` and has a
docstring that specifies its behaviour. Do not change the provided function
signatures; the tests depend on them.

## 4. Prerequisites

- Windows 10/11 with **PowerShell**
- **Python 3.12** (`py -3.12 --version`)
- **Docker Desktop**, running
- **Git**
- About 2 GB of free disk space (PyTorch CPU, the 90 MB embedding model, and the
  Docker image)
- **LiteLLM gateway access** from your mentor: base URL, personal key and model
  alias. You only need it from milestone M9 onwards.

## 5. Setup (Windows PowerShell)

Run all commands from the repository root.

### 5.1 Virtual environment and dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
python --version                       # Python 3.12.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5.2 Environment configuration

```powershell
Copy-Item .env.example .env
notepad .env
```

- Set `POSTGRES_PASSWORD` to any local password.
- Leave `LITELLM_API_BASE`, `LITELLM_API_KEY` and `LITELLM_MODEL` empty until
  your mentor gives you real values. **Never guess them, commit them, or paste
  them into code, tests or PRs.** `.env` is git-ignored.

### 5.3 Database: start, initialise, stop

```powershell
docker compose up -d                   # start PostgreSQL 17 + pgvector
docker compose ps                      # wait for "healthy"
python scripts/ingest.py --check-db    # connectivity check
python scripts/ingest.py --init-db     # create the three empty tables + indexes

docker compose logs db                 # troubleshooting
docker compose down                    # stop, keep data
docker compose down -v                 # stop and DELETE all data
```

### 5.4 Verify the starter (milestone M0)

```powershell
python -m pytest -m m0                 # should pass completely
python -m pytest                       # passes M0; every unfinished milestone shows as SKIPPED "TODO [Mx] ..."
```

## 6. Working through the lab

Follow **[docs/milestones.md](docs/milestones.md)**. Each milestone lists the
functions to implement, the tests to run, the expected output, and debugging
tips. Read **[docs/lab_guide.md](docs/lab_guide.md)** for the concepts.

### Task 1: Ingestion and chunking (M1–M6)

1. Load all six documents with validated metadata and stable IDs (M1).
2. Implement fixed-size (M2), recursive (M3) and structure-aware (M4) chunking.
   Chunk size and overlap come from `.env` or the CLI flags.
3. Embed with the local MiniLM model, using the same model for every strategy
   (M5).
4. Store chunks, vectors and metadata in the three pgvector tables. Verify that
   generated and persisted counts match, and make re-ingestion deterministic,
   with no duplicates (M6). Document your replacement strategy.
5. Report chunk counts and average sizes for at least three chunk size/overlap
   configurations.

```powershell
python scripts/ingest.py --dry-run                        # M1–M4: no model, no DB
python scripts/ingest.py                                  # M6: all strategies
python scripts/ingest.py --strategies structured --chunk-size 600 --chunk-overlap 80
```

### Task 2: Retrieval and question answering (M7–M11)

1. Similarity search with metadata filters (M7).
2. Build the context and prompt (M8), then call the LiteLLM gateway (M9).
3. Return source filenames and section headings, and refuse when the answer is
   not in the documents (M10).
4. Run the 15 evaluation queries on all three strategies and record the results
   in `data/rag_test_queries.md` (M11).

```powershell
python scripts/ask.py "How many days of annual leave can I carry forward?" --retrieval-only
python scripts/ask.py "Which days must hybrid employees be in the office?" --location Pune
python scripts/ask.py "What is the address of the new office?" --doc-type circular --show-context
python scripts/ask.py "What is the vesting schedule for employee stock options?"   # must be refused
python scripts/evaluate.py --retrieval-only --output eval_retrieval.md
```

Expected answers live **only** in `data/rag_test_queries.md`, for you to judge
results. Never hard-code them, put them in prompts, or ingest that file.

## 7. Tests

| Command | What runs | Needs |
| --- | --- | --- |
| `python -m pytest` | All unit tests | nothing |
| `python -m pytest -m m4` | One milestone (unit + model tests) | as listed |
| `python -m pytest -m integration` | Your SQL against real pgvector (`test_*` tables) | Docker |
| `python -m pytest -m "m7 and integration"` | Integration tests for one milestone | Docker |
| `python -m pytest -m model` | Tests that load the real embedding model | internet once |
| `python -m pytest -m gateway` | One real LiteLLM call | `LITELLM_*` in `.env` |
| `python -m pytest --strict-todos` | Treat TODO skips as failures | – |
| `ruff check .` / `ruff format --check .` | Lint / formatting | – |

Reading results: **SKIPPED "TODO [Mx]"** means not implemented yet.
**FAILED** means implemented, but the behaviour differs from the specification.
Integration tests are skipped automatically if the database is unreachable.
Unit tests use simple test doubles (`tests/fakes.py`): a word-hashing embedder,
a recording store and a fixed-response LLM. They make each component testable
in isolation; they are not implementations of the real components.

## 8. Definition of Done

- [ ] `python -m pytest --strict-todos` passes, meaning no TODOs remain.
- [ ] `python -m pytest -m integration` and `python -m pytest -m model` pass,
      and `python -m pytest -m gateway` passes with your gateway settings.
- [ ] `ruff check .` and `ruff format --check .` are clean.
- [ ] `python scripts/ingest.py` shows `[OK]` for all three strategies, and a
      second run gives identical counts and chunk IDs.
- [ ] All 15 queries are evaluated on all three strategies, and every results
      table and the summary in `data/rag_test_queries.md` are filled in.
- [ ] Q13–Q15 are refused with no invented details. Every answer shown has
      sources (filename + section).
- [ ] Type hints on all new code. SQL values are parameterised. No secrets,
      `.env`, model weights or database dumps are committed.
- [ ] `docs/submission_checklist.md` is completed in the PR description.

## 9. Submission and pull request

```powershell
git switch -c intern/<your-name>-rag-lab
git add src tests data/rag_test_queries.md docs
git status                              # .env must NOT be listed
git commit -m "RAG lab: <milestones completed>"
git push -u origin intern/<your-name>-rag-lab
```

Open a PR against the lab branch your mentor names. Include:
1. your findings and recommended configuration (strategy, chunk size/overlap,
   top-k, threshold);
2. your ingestion replacement strategy and why;
3. the completed checklist from `docs/submission_checklist.md`;
4. output of `python scripts/ingest.py` and `python -m pytest --strict-todos`;
5. one answered `ask.py` example with sources and one correctly refused example.

Small PRs per milestone are welcome, and easier to review.

## 10. Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `Could not connect to PostgreSQL` | Run `docker compose up -d`, wait for `healthy`, and check `POSTGRES_PORT`/`DATABASE_URL` in `.env` |
| Compose: `Set POSTGRES_PASSWORD in .env` | Run `Copy-Item .env.example .env` and set a password |
| Port 5433 already in use | Change `POSTGRES_PORT` in `.env`, then `docker compose up -d` again |
| `Table rag_chunks_* does not exist` | Run `python scripts/ingest.py --init-db`, or finish M6 and run `python scripts/ingest.py` |
| `NOT IMPLEMENTED YET: [Mx] ...` (exit code 2) | The command needs milestone Mx. See `docs/milestones.md` |
| Test `SKIPPED ... TODO [Mx]` | Expected until Mx is implemented |
| `ModuleNotFoundError: rag_lab` in a Python shell | Run from the repo root, and add `src` to `sys.path` (the scripts and tests do this already) |
| `LiteLLM gateway is not configured` | Get the real values from your mentor and put them in `.env` |
| Gateway 401/403/404 | Wrong key, or wrong model alias or prefix. See M9 tips |
| Model download fails | Network or proxy problem. Retry on another network or ask your mentor |
| `EMBEDDING_DIM=... but model produces ...` | Fix `EMBEDDING_DIM`, then re-ingest every strategy |
| Every question returns "could not find" | Check with `--retrieval-only --min-similarity 0`. Look for the wrong table, a filter that is too strict, or a threshold that is too high |
