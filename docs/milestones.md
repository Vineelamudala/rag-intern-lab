# Milestones

Build the pipeline one component at a time. Each milestone lists the files to
edit, the tests that define "done", what you should see when it works, and
common problems.

**How tests report progress.** A test that reaches an unfinished function
(`NotImplementedError`) is reported as **SKIPPED** with a reason like
`TODO [M3] chunk_recursive is not implemented yet`. A **FAILED** test means your
implementation runs but behaves differently from the specification. Read the
assertion, then fix your code, not the test. To see TODOs as failures instead,
add `--strict-todos`.

```powershell
python -m pytest -m m3          # one milestone (includes its model tests, if any)
python -m pytest                # everything except integration/model/gateway tests
python -m pytest -m "m6 and integration"   # integration tests for one milestone
python -m pytest -m m3 -x -vv   # stop at first failure, show full diffs
```

| # | Component | Files | Needs |
| --- | --- | --- | --- |
| [M0](#m0) | Environment and infrastructure check | none (provided) | Docker for one test |
| [M1](#m1) | Document loading and metadata | `document_loader.py` | – |
| [M2](#m2) | Fixed-size chunking and heading helpers | `chunkers.py` | M1 for corpus tests |
| [M3](#m3) | Recursive character chunking | `chunkers.py` | M2 helpers |
| [M4](#m4) | Structure-aware Markdown chunking | `chunkers.py` | M2 helpers |
| [M5](#m5) | Local embeddings | `embeddings.py` | Internet once (model download) |
| [M6](#m6) | Ingestion, storage, persistence validation | `ingestion.py`, `vector_store.py` | M1–M5, Docker |
| [M7](#m7) | Similarity retrieval and metadata filters | `vector_store.py`, `retrieval.py` | M6, Docker |
| [M8](#m8) | Context construction and prompt engineering | `answering.py` | – |
| [M9](#m9) | LiteLLM gateway integration | `answering.py` | Gateway settings for the live test |
| [M10](#m10) | Citations and unanswerable questions | `answering.py` | M8 |
| [M11](#m11) | Evaluation across all three strategies | `evaluation.py`, `data/rag_test_queries.md` | M1–M10 |

---

## M0

**Goal:** prove your machine is ready before writing any code.

1. Follow the README setup (venv, `pip install -r requirements.txt`, `.env`).
2. `docker compose up -d`, then `docker compose ps` should show `healthy`.
3. `python scripts/ingest.py --check-db` should print `Connected. pgvector extension version: ...`.
4. `python scripts/ingest.py --init-db` should print `Initialised tables: rag_chunks_fixed, ...`.
5. `python -m pytest -m m0` should pass completely. With Docker running,
   `python -m pytest -m "m0 and integration"` should pass too.

**If it fails:** see README → Troubleshooting. Do not continue until M0 is green;
every later milestone depends on it.

---

## M1

**Goal:** turn each Markdown file into a `Document` with validated metadata.
**Implement:** `make_doc_id`, `parse_front_matter`, `load_document`, `load_documents`.
**Tests:** `python -m pytest -m m1`

**Expected outcome:** all M1 tests pass. In a Python shell:

```powershell
python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from rag_lab.document_loader import load_documents; [print(d.source, d.doc_type, d.location, d.effective_date) for d in load_documents(Path('data/rag_docs'))]"
```

This prints six lines, sorted by file name.

**Debugging tips**
- The HR circular's title contains a colon and is quoted. Make sure you split
  on the first colon only.
- If IDs differ between runs, you are probably using Python's `hash()`, which is
  randomised per process.
- Error messages must name the file. Future you will thank present you.

---

## M2

**Goal:** the naive baseline, written without a library, plus helpers that every
strategy reuses.
**Implement:** `validate_chunk_params`, `format_section_heading`,
`heading_path_at`, `chunk_fixed`.
**Tests:** `python -m pytest -m m2`. This includes the corpus-wide metadata tests
in `test_metadata.py`, which need M1.

**Expected outcome:** `python scripts/ingest.py --dry-run --strategies fixed`
prints a chunk count and average size for `fixed`. Try several
`--chunk-size`/`--chunk-overlap` values and record the numbers.

**Debugging tips**
- An off-by-one in the window step shows up in `test_fixed_windows_overlap_exactly`.
- Do not `.strip()` fixed windows. The concatenation test will tell you if you do.
- Print `doc.text[start:start+80]` for a few windows to see where they cut.

---

## M3

**Goal:** use LangChain's `RecursiveCharacterTextSplitter`.
**Implement:** `chunk_recursive`.
**Tests:** `python -m pytest -m m3`

**Expected outcome:** `--dry-run --strategies recursive` works. Chunks mostly end
at paragraph or line boundaries.

**Debugging tips:** you need each piece's start offset to find its section. Read
the splitter's documentation for an option that records it.

---

## M4

**Goal:** split by Markdown headings so each chunk belongs to exactly one section.
**Implement:** `chunk_structured`.
**Tests:** `python -m pytest -m m4`

**Expected outcome:** `--dry-run` works for all three strategies. Structured
produces the most, and the smallest, chunks. Explain why in your report.

**Debugging tips**
- The header splitter can strip or keep heading lines. The tests expect them kept.
- Print `section.metadata` for a few sections to see what the splitter gives you.
- Header lines may come back with trailing whitespace.

---

## M5

**Goal:** local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
**Implement:** `SentenceTransformerEmbedder` and `validate_vectors`.
**Tests:** `python -m pytest -m m5`. The first run downloads about 90 MB, so
expect the model tests to take a minute.

**Expected outcome:** all M5 tests pass, including `test_semantic_similarity`.

**Debugging tips**
- "Vectors not normalised": look for an option of `encode` rather than
  normalising by hand.
- A model download that fails behind a proxy is an environment issue. Ask your
  mentor.
- The embedding model never uses the LiteLLM gateway.

---

## M6

**Goal:** persist chunks and vectors, verify the counts, and make re-ingestion
idempotent.
**Implement:** `summarise_chunks`, `build_chunks`, `ingest_strategy` (in
`ingestion.py`), and `replace_chunks`, `count`, `chunk_ids`, `embedding_models`
(in `vector_store.py`).
**Tests:** `python -m pytest -m m6` (unit, with fakes), then
`python -m pytest -m "m6 and integration"` (your SQL against the real database,
using `test_*` tables).

**Expected outcome:**

```powershell
python scripts/ingest.py
```

This prints one row per strategy ending in `[OK]`, followed by
`All generated and persisted counts match.` Run it twice: the counts must not
change. Check directly as well:

```powershell
docker exec -it rag-intern-lab-db psql -U rag -d rag_lab -c "SELECT count(*), count(DISTINCT chunk_id) FROM rag_chunks_structured"
```

**Debugging tips**
- `can't adapt type 'list'` on the embedding column: use `to_pgvector(...)`.
- `expected 384 dimensions, not N`: the embedder and `EMBEDDING_DIM` disagree.
- Counts grow on every run: your replacement strategy leaves old rows behind.
- A crash mid-insert left a half-filled table: wrap the work in `conn.transaction()`.
- Write down your replacement strategy and why you chose it. Reviewers will ask.

---

## M7

**Goal:** cosine-similarity search with metadata filters.
**Implement:** `build_search_query`, `PgVectorStore.search`, `Retriever.retrieve`.
**Tests:** `python -m pytest -m m7`, then `python -m pytest -m "m7 and integration"`.

**Expected outcome:**

```powershell
python scripts/ask.py "What is the address of the new office?" --doc-type circular --retrieval-only
```

This lists only `hr_circular.md` chunks, with scores between -1 and 1.

**Debugging tips**
- Scores look inverted (best result has the lowest score): `<=>` is a
  *distance*.
- To check your SQL, print `query.as_string(store.connect())` and run it in
  `psql`.
- A filter that returns nothing: check case sensitivity and the
  `All Locations` rule.

---

## M8

**Goal:** turn retrieved chunks into a prompt that keeps the model grounded.
**Implement:** `format_context`, `build_prompt`.
**Tests:** `python -m pytest -m m8`

**Expected outcome:** `--retrieval-only --show-context` prints numbered `[S1]`,
`[S2]` blocks. Keep notes on every prompt change you try (M11 uses them).

---

## M9

**Goal:** call the organisation's LiteLLM gateway.
**Implement:** `LiteLLMProvider.generate`.
**Tests:** `python -m pytest -m m9`. These mock `litellm`, so no network or key is
needed. With your gateway settings in `.env`, run `python -m pytest -m gateway`
for one real call.

**Expected outcome:** the live smoke test passes.

**Debugging tips**
- `ConfigError ... LITELLM_* missing`: fill in `.env` with the values from your
  mentor. Never invent them.
- 401/403: wrong or expired key. 404 or "model not found": check the model alias
  and whether the SDK needs a provider prefix for a proxy (see the LiteLLM docs).
- Never print, log or commit the key, including in screenshots.

---

## M10

**Goal:** citations from metadata, and refusing instead of inventing.
**Implement:** `extract_citations`, `is_not_found`, `sources_for`, `answer_question`.
**Tests:** `python -m pytest -m m10`

**Expected outcome:**

```powershell
python scripts/ask.py "How many days of annual leave can I carry forward?"
python scripts/ask.py "What is the vesting schedule for employee stock options?"
```

The first prints an answer with sources (`leave_policy.md | ...`). The second
prints `I could not find the answer in the available documents.`

---

## M11

**Goal:** compare the three strategies with evidence.
**Implement:** `is_retrieval_hit`, `evaluate_strategy`, `summarise`, `render_markdown`.
**Tests:** `python -m pytest -m m11`

**Expected outcome:**

```powershell
python scripts/evaluate.py --retrieval-only --output eval_retrieval.md
python scripts/evaluate.py --output eval_full.md
```

Then fill in every results table and the summary in `data/rag_test_queries.md`.
Answer correctness still needs your judgement against the expected facts.
