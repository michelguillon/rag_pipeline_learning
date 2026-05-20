# RAG Pipeline — Document Q&A with Mistral + ChromaDB

A retrieval-augmented question-answering system, built from scratch: it ingests
a CV, embeds it into a vector store, retrieves the relevant chunks for a
question, and has Mistral generate an answer grounded **only** in those chunks.

Built as an AI learning project. The point was not just working code —
it was to understand and document *every* architectural decision. The reasoning
lives in [docs/SPEC.md](docs/SPEC.md) (Phase 1 spec),
[docs/SPEC_PHASE2.md](docs/SPEC_PHASE2.md) (Phase 2 spec), and
[docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md) (what each phase taught).

---

## Architecture

```
                        analyse.py ── Mistral ──► chunking recommendation
Document (.docx) ───►        │                          │
                       human approves              config.json
                             │                          │
                       review_chunks.py ──► preview chunks (no embedding)
                             │                          │
                       human confirms                   │
                             │                          │
                        ingest.py ── mistral-embed ──► ChromaDB (4 collections)
                                                        │
Question ──► query.py ── embed ──► ChromaDB ── top-k ──► Mistral ──► Answer
```

Two **human-in-the-loop gates** by design: a human approves the chunking config
(`analyse`) and reviews the actual chunks (`review_chunks`) before any embedding
spend. The pipeline is a set of small single-purpose scripts over shared
modules:

| Module | Role |
|--------|------|
| `models/paragraph.py` | common `Paragraph` dataclass — every loader's output |
| `loaders/docx_loader.py` | `.docx` → `list[Paragraph]` |
| `loaders/pdf_loader.py` | `.pdf` → `list[Paragraph]` (pdfplumber) |
| `loaders/__init__.py` | `load(path)` dispatcher — picks loader by extension |
| `chunker.py` | config-driven decode (signal grammar, ordered rules) → chunks |
| `mistral_helpers.py` | Mistral client + `call_with_retry()` |

---

## Setup

Requires Docker Desktop. The container provides Python 3.13 and all
dependencies — nothing is installed on the host.

```powershell
# 1. API key
Copy-Item .env.example .env        # then paste your MISTRAL_API_KEY into .env

# 2. Build the image
docker compose build

# 3. Drop a CV at data/sample_cv.docx
```

`.env` is git-ignored; the project directory is mounted into the container, so
`config.json`, `chroma_db/` and `outputs/` persist between runs.

---

## Running the pipeline

```powershell
# 1. Analyse the document, get a chunking recommendation, write config.json
docker compose run pipeline python analyse.py data/sample_cv.docx

# 2. Preview the proposed chunks before spending anything on embeddings
docker compose run pipeline python review_chunks.py data/sample_cv.docx

# 3. Embed the chunks and store them in the 4 ChromaDB collections
docker compose run pipeline python ingest.py data/sample_cv.docx

# 4. Ask a question
docker compose run pipeline python query.py "What was his most recent role?" `
  --collection cv1_bullet_cosine --top-k 3 --format labelled

# 5. Run the full experiment matrix (resumable)
docker compose run pipeline python stress_test.py

# 6. Phase 2 cross-document validation (3 CVs, fixed question set)
docker compose run pipeline python validate.py
```

`analyse.py --trace` writes the full prompt + Mistral response to `outputs/`.
`analyse.py --profile-only` prints the fingerprint profile without calling
Mistral — useful for hand-authoring rules. `stress_test.py` checkpoints after
every cell — re-run it to resume. `validate.py --report-only` re-renders
`outputs/phase2_validation/comparison.md` from `results.json` without
re-spending API.

---

## Architectural decisions

Full reasoning is in [docs/SPEC.md](docs/SPEC.md) and [docs/SPEC_PHASE2.md](docs/SPEC_PHASE2.md). The load-bearing ones:

- **Semantic chunking, not fixed-size.** Each chunk is a structural unit (a
  role, or a bullet) — no chunk-size or overlap parameter. Two strategies are
  built and compared: **A** (one chunk per role) and **A2** (one chunk per
  bullet + a context prefix).
- **Profile the document, don't assume it.** Real Word files encode hierarchy
  inconsistently — `analyse.py` enumerates formatting fingerprints and flags
  inconsistency rather than hardcoding "Heading 3 = company".
- **Config-driven decode (Phase 2).** `chunker.py` carries no document-specific
  logic; it reads an ordered `fingerprint_rules` list from per-CV `config.json`
  and executes it. Four signals (`has_numPr`, `is_bold`, `rendered_size==n`,
  `style==name`) and a no-`eval()` parser. Order replaces compound conditions
  (decision list, not boolean expression tree).
- **Common `Paragraph` model + format dispatcher (Phase 2).** All loaders emit
  the same dataclass; `chunker.py` knows nothing about `.docx` vs `.pdf`.
  Adding a new format is one new file in `loaders/`.
- **ChromaDB, persistent**, 4 collections (chunk strategy × cosine/L2 metric).
- **`mistral-embed`** for documents and queries — the one un-swappable choice,
  because vectors from different embedding models live in incompatible spaces;
  **`mistral-small`/`large`** for generation, swappable freely; temperature 0.1.
- **Anti-hallucination**: the model is instructed to answer only from context
  and to emit an exact fallback phrase otherwise — making hallucination
  measurable, not just observable.

---

## Stress test findings

112 cells: 4 collections × top_k × flat/labelled context × 7 questions, on
`mistral-small`. (The small-vs-large model comparison was skipped to stay
within free-tier limits.)

- **Hallucination refusal: 16/16 cells, 100%.** Every out-of-scope question
  returned the exact fallback phrase, in every configuration — the strongest
  and most robust result.
- **top_k drives token cost, roughly linearly** (~250→~480 tokens for role
  chunks, k=1→k=3). Retrieved context is paid for on every call.
- **Labelled context costs ~10–25% more tokens than flat**, with no visible
  answer-quality gain on this corpus.
- **Strategy A vs A2 produced near-equivalent answers.** The expected
  advantage of whole-role chunks (A) for synthesis questions did not clearly
  materialise — with enough top_k, per-bullet chunks (A2) reconstruct the
  context.

**Meta-finding:** on a small (~36-chunk), clean, single-document corpus the
pipeline is forgiving — strategy/metric/format choices move token cost far more
than answer quality. The *discipline* (grounding, human review, checkpointing)
mattered more than the *tuning*. Tuning earns its keep at scale.

**Generalisation test:** running the pipeline on two additional CVs confirmed
the boundary. The fingerprint profiler described both correctly and diagnosed
its own downstream failures. The chunker, hardcoded to the first CV's
conventions, collapsed silently — producing whole-CV mega-chunks with no error.
The profiler generalised. The chunker didn't, until Phase 2 closed the loop ↓

---

## Cross-document validation (Phase 2)

The Phase 1 cross-CV failure was the explicit motivation for Phase 2. The
chunker now reads its decode rules from `config_cv*.json` (an ordered list of
`fingerprint_rules` under a 4-signal no-`eval()` grammar) rather than carrying
them in code. Same chunker, three structurally-different CVs — only the
config changes:

| CV | Structure | Phase 1 result | Phase 2 result (A / A2) |
|----|-----------|----------------|--------------------------|
| `sample_cv.docx` | Heading 1 + table, mixed-size headings | ✅ 11 / 25 | ✅ **11 / 25** unchanged |
| CV #2 (private) | No heading styles at all — sizes + bold only | ❌ 3 mega-chunks | ✅ **6 / 31** |
| CV #3 (private) | Table-based, `Title` paragraph style as company signal | ❌ 4 mega-chunks | ✅ **10 / 36** |

**Decode rules diff (the only thing that changes between runs):**

```
cv1                                    cv2 (no heading styles)         cv3 (Title-style)
  has_numPr        -> bullet            has_numPr       -> bullet       has_numPr        -> bullet
  style==Heading 1 -> section_header    rendered_size==12 -> section    style==Title     -> company
  rendered_size==14 -> company          rendered_size==20 -> section    rendered_size==22 -> section
  rendered_size==18 -> company          is_bold         -> job_title    rendered_size==16 -> section
  style==Heading 3 -> job_title                                         is_bold          -> company
```

A fixed 7-question set (the spec's 5 factual/synthesis/hallucination questions
+ a date-range and a multi-company synthesis question) runs against each CV
under identical settings (A2, cosine, top-3, `mistral-small`, labelled context).
**All three CVs correctly refused the hallucination question** (the "current
salary" probe). Answerable-question refusals are reported separately as
*retrieval gaps* — a useful diagnostic, distinct from hallucination.

**Bonus — PDF parity.** A PDF rendering of the same `sample_cv.docx` content
loaded through `loaders/pdf_loader.py` (pdfplumber) produced **11 / 29 chunks**
(vs 11 / 28 for the `.docx`) with identical bullet recovery (21 / 21). The
chunker code was unchanged; only the loader differs. That is the "format-
agnostic pipeline" claim demonstrated end-to-end.

Full per-question results live in `outputs/phase2_validation/comparison.md`
(git-ignored — contains real CV text). Findings, decision-list reasoning, and
the Mistral compound-signal anecdote are in
[docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md).

---

## Reflections

The most useful finding wasn't in the stress test — it was earlier,
when a naive parser silently read 13 of 43 paragraphs and produced
wrong-but-plausible chunks with no error. Real Word documents encode
hierarchy inconsistently: a CV's visual structure and its underlying
markup are two different things, and a parser that assumes otherwise
fails quietly. I rebuilt the analysis step as a fingerprint profiler
that discovers structure rather than assuming it — and when I tested
on two other CVs, the profiler correctly described both and diagnosed
its own downstream failures. The chunker, still hardcoded at that
point, collapsed. Phase 2 closed that loop: the decode rules now travel
in `config.json` as an ordered list of single-signal rules under a tiny
no-`eval()` grammar, derived per-document by `analyse.py` + a human.
Three structurally-different CVs validate cleanly through the same
chunker — only `fingerprint_rules` changes.

The 112-cell stress test confirmed something less obvious: on a small,
clean corpus, the tuning knobs — chunk strategy, distance metric,
context format — move token cost far more than answer quality. The
discipline mattered more than the tuning: grounding the model strictly
to retrieved context, inserting human review before embedding spend,
checkpointing a batch job that would otherwise lose 91 completed calls
on failure 92. These aren't glamorous decisions. They're the ones that
determine whether a RAG system works on a customer's actual documents,
not just the clean sample it was demoed on.

The Phase 2 work added one more lesson worth keeping: when you give an
LLM a small grammar and tell it to honour the grammar, it may quietly
refuse — Mistral persistently produced compound `&&` signals across
three prompt iterations despite explicit "FORBIDDEN" language. The
validation gate caught every attempt before any bad config landed on
disk. The "Mistral proposes, human approves" loop is load-bearing, not
ceremonial.

---

## Production upgrade path

| Area | This project | Production |
|------|-------------|------------|
| Chunking | ✅ Config-driven decode (Phase 2), derived per-document by `analyse.py` + human approval | — |
| Format support | ✅ `.docx` + `.pdf` via a common `Paragraph` model and a `load(path)` dispatcher (Phase 2) | Pluggable chunking-**strategy** registry per document class (CV vs RFP vs report) — the loader story is done; the strategy story remains |
| Vector store | ChromaDB (local) | Pinecone / pgvector when corpus > ~1M vectors or multi-tenant |
| API tier | Mistral free tier (rate-limited) | Paid tier; batch embedding |
| Corpus | Single document per ingestion run | Multi-document corpus with metadata-filtered retrieval (by company / date / tenant) — the real scaling lever |
| Retrieval | Semantic only | Hybrid semantic + BM25 keyword |
| Evaluation | Token cost, similarity, hallucination refusal, retrieval-gap rate | + Automated answer-quality scoring |
