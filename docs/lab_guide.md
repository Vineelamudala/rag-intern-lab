# RAG Lab Guide

This guide explains the *ideas* behind each stage of the pipeline you will
build. It tells you what each stage is for and what to watch out for, not how to
code it. The exact requirements for each function are in its docstring, and the
order of work is in [milestones.md](milestones.md).

---

## 1. The big picture

A language model only knows what was in its training data. It knows nothing
about *Lumen Ridge's* leave policy. **Retrieval-Augmented Generation** fixes
this by finding the relevant passages first and then asking the model to answer
**only from those passages**.

```
          INGESTION (offline, scripts/ingest.py)
 documents ─► chunks ─► embeddings ─► pgvector tables
                                           │
          QUESTION ANSWERING (online, scripts/ask.py)
 question ─► embedding ─► similarity search (+ filters) ─► prompt ─► LLM ─► answer + citations
```

The two halves are deliberately separate. You can re-index without touching the
answering code, and you can evaluate retrieval without any LLM at all.

---

## 2. Chunking

**Why chunk at all?** Embedding a whole policy produces one vector that blurs
together sick leave, maternity leave and encashment. A question about one rule
then matches the whole document weakly. Smaller chunks give sharper vectors and
let the prompt carry only the relevant text. The trade-off: chunks that are too
small lose context ("10 days" of *what*?).

The lab compares three strategies:

| Strategy | Idea | Typical weakness |
| --- | --- | --- |
| **Fixed-size** | Cut every *N* characters, with overlap | Splits sentences, tables and rules mid-way |
| **Recursive** | Prefer paragraph breaks, then lines, then words | Ignores the document's logical sections |
| **Structure-aware** | Split on Markdown headings first, then by size | Small sections can be too short to stand alone |

**Overlap** repeats the end of one chunk at the start of the next, so a fact
near a boundary appears whole in at least one chunk. It costs storage and can
return near-duplicate results.

Questions to investigate:

- Which strategy produces the most chunks, and why?
- Find a rule that one strategy keeps intact and another splits apart.
- What happens to a Markdown table under each strategy?

---

## 3. Metadata

Each chunk stores more than text: its **source file**, **document ID**,
**title**, **document type**, **effective date**, **section heading**,
**department** and **location**. Metadata serves three purposes:

1. **Filtering.** Narrow the search *before* ranking. For example, use only
   circulars, or only rules that apply in Pune. Filters are exact and
   explainable; similarity is neither.
2. **Citations.** The answer can say where each fact came from. The citation is
   built from stored metadata, never from text the model wrote.
3. **Reasoning about conflicts.** When two passages disagree, the effective date
   tells you which one is current. See the Hyderabad anchor-day rule in the
   circular versus the remote work policy.

Think about it: the `location` filter also returns documents tagged
`All Locations`. Why is that the right default, and when would you want
`--exact-location`?

**Stable IDs.** Document IDs are derived from the file name, and chunk IDs from
`(document, strategy, position)`. This makes re-ingestion deterministic. Ask
yourself what would break if IDs were random UUIDs.

---

## 4. Embeddings

An embedding model maps text to a vector (384 numbers for MiniLM) so that
texts with similar *meaning* end up close together. "Carry forward unused leave"
lands near "How many leave days roll over?" even though the two share few
words.

Rules that matter in practice:

- **Use one model for everything.** Document and query vectors must come from
  the same model, or the distances are meaningless. That is why each stored
  row records its model name, and why your retriever should refuse to mix
  models.
- **The dimension is part of the schema.** A pgvector column is declared as
  `vector(384)`. Changing the model means changing the column and re-ingesting.
- **Normalise the vectors.** With unit-length vectors, cosine similarity and
  the dot product agree, and scores are comparable across queries.

The unit tests use a fake *hashing* embedder (`tests/fakes.py`). It is
deterministic and needs no download, but it only counts shared words. Once your
MiniLM embedder works, compare the two on a few questions to see what
"semantic" search adds.

---

## 5. Vector search with pgvector

pgvector adds a `vector` column type and distance operators to PostgreSQL. This
lab uses **cosine distance** (`<=>`); similarity = `1 − distance`.

- **Exact vs. approximate search.** Without an index, PostgreSQL compares the
  query against every row. That is exact, and fine for a hundred chunks. An
  **HNSW** index makes search fast at scale by trading a little recall for
  speed. The provided schema helper creates one. Read the pgvector docs on its
  parameters.
- **Filters plus vectors.** Metadata filters go in the `WHERE` clause. With
  approximate indexes, very selective filters can return fewer than `top_k`
  rows. That is something to keep in mind for larger corpora.
- **Parameterised SQL.** Filter values and vectors must always be passed as
  query parameters. Table names come from a fixed whitelist and are quoted with
  `psycopg.sql.Identifier`. One M7 unit test feeds an SQL-injection payload as a
  filter value. Make sure your query survives it.

---

## 6. Grounding

*Grounded* means every claim in the answer is supported by retrieved text.
Your pipeline should have several layers of defence:

1. **Relevance threshold.** If nothing is similar enough, do not call the LLM
   at all.
2. **Constrained prompt.** The prompt contains only numbered context blocks and
   the question, and tells the model to reply with a fixed sentinel when the
   context is insufficient.
3. **Output validation.** An answer without valid `[S#]` citations is rejected,
   even if it *sounds* right. This catches the model answering from its own
   general knowledge.

No layer is perfect. A model can cite `[S1]` and still misstate what S1 says.
That is why evaluation (section 8) checks answer *content*, not just the
presence of citations.

---

## 7. Citations

The user sees the **source filename and section heading** for every cited
block. Good citations let a reader verify an answer in seconds and make
retrieval mistakes visible. The sources list should contain only the blocks the
model actually cited, not everything retrieved. Why does that matter?

---

## 8. Evaluation

You cannot improve what you do not measure. `data/rag_test_queries.md` contains
15 queries in five categories:

| Category | What it tests |
| --- | --- |
| Single document | Basic retrieval and answer accuracy |
| Multiple passages | Whether `top_k` and chunking surface every needed piece |
| Metadata filters | Whether filters are applied, and whether they help |
| Dates and locations | Whether the right *version* or *office* rule is used |
| Unanswerable | Whether the system refuses instead of inventing |

Measure **retrieval** and **generation** separately:

- *Retrieval hit:* is the expected source and section in the top-k? You can
  check this with `--retrieval-only`, without an LLM.
- *Answer correctness:* are the key facts right, and is everything cited?
- *Refusal correctness:* for Q13–Q15, did the system say it could not find the
  answer?

A strategy that wins on retrieval can still lose on answers, for example when
its chunks are too small to contain the full rule. Report both numbers.

---

## 9. The LiteLLM gateway

LiteLLM gives many model providers one OpenAI-style interface. The organisation
runs a LiteLLM **gateway** (a proxy). Your code sends chat-completion requests to
it with a personal key, and the gateway handles provider credentials, model
routing, logging and budgets. For this lab:

- The gateway is used **only for answer generation**. Embeddings stay local.
- The URL, key and model alias come from `.env` (`LITELLM_*`). If they are
  missing, `Settings.require_gateway()` tells you exactly what is missing.
- Treat the key like a password: never print it, log it, commit it or put it
  in a screenshot.
- The unit tests mock the `litellm` module, so you can finish M9 before you
  have access. The `gateway` test makes one real call.

---

## 10. Going further (optional)

- *Contextual chunk headers:* prefix each sub-chunk with its section heading
  and measure the effect.
- *Re-ranking:* re-score the top-k with a cross-encoder before prompting.
- *Hybrid search:* combine PostgreSQL full-text search with vector similarity.
