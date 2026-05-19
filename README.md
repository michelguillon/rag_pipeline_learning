# RAG Pipeline — Document Q&A with Mistral + ChromaDB

A retrieval-augmented question-answering system, built from scratch: it ingests
a CV, embeds it into a vector store, retrieves the relevant chunks for a
question, and has Mistral generate an answer grounded **only** in those chunks.

Built as a Week-1 AI learning project. The point was not just working code —
it was to understand and document *every* architectural decision. The reasoning
lives in [rag_pipeline_spec.md](rag_pipeline_spec.md) (the spec) and
[LEARNING_NOTES.md](LEARNING_NOTES.md) (what each phase taught).

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
| `docx_parser.py` | `.docx` → paragraph records (style, size, list, dates) |
| `chunker.py` | decode rule → chunks, for strategy A and A2 |
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
  --collection cv_role_cosine --top-k 3 --format labelled

# 5. Run the full experiment matrix (resumable)
docker compose run pipeline python stress_test.py
```

`analyse.py --trace` writes the full prompt + Mistral response to `outputs/`.
`stress_test.py` checkpoints after every cell — re-run it to resume.

---

## Architectural decisions

Full reasoning is in [rag_pipeline_spec.md](rag_pipeline_spec.md). The load-bearing ones:

- **Semantic chunking, not fixed-size.** Each chunk is a structural unit (a
  role, or a bullet) — no chunk-size or overlap parameter. Two strategies are
  built and compared: **A** (one chunk per role) and **A2** (one chunk per
  bullet + a context prefix).
- **Profile the document, don't assume it.** Real Word files encode hierarchy
  inconsistently — `analyse.py` enumerates formatting fingerprints and flags
  inconsistency rather than hardcoding "Heading 3 = company".
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
The profiler generalises. The chunker doesn't, yet.

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
its own downstream failures. The chunker, which was still hardcoded,
collapsed. That gap is the next build.

The 112-cell stress test confirmed something less obvious: on a small,
clean corpus, the tuning knobs — chunk strategy, distance metric,
context format — move token cost far more than answer quality. The
discipline mattered more than the tuning: grounding the model strictly
to retrieved context, inserting human review before embedding spend,
checkpointing a batch job that would otherwise lose 91 completed calls
on failure 92. These aren't glamorous decisions. They're the ones that
determine whether a RAG system works on a customer's actual documents,
not just the clean sample it was demoed on.

The tuning earns its keep at scale and with noisier retrieval — which
is exactly where this is going next.

---

## Production upgrade path

| Area | This project | Production |
|------|-------------|------------|
| Chunking | Hardcoded decode rules | Config-driven, derived per-document by `analyse.py` |
| Vector store | ChromaDB (local) | Pinecone / pgvector when corpus > ~1M vectors or multi-tenant |
| API tier | Mistral free tier (rate-limited) | Paid tier; batch embedding |
| Document types | Single CV (`.docx`) | Format-agnostic loader + pluggable chunking-strategy registry per document class |
| Metadata | Attached, lightly used | Metadata-filtered retrieval (by company / date / tenant) — the real scaling lever |
| Retrieval | Semantic only | Hybrid semantic + BM25 keyword |
| Evaluation | Token cost, similarity, hallucination refusal | + Automated answer-quality scoring |
