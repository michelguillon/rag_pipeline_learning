# CLAUDE.md — RAG Pipeline

## What this is

A document Q&A system (RAG) built with **Mistral API + ChromaDB**, ingesting a CV,
retrieving relevant chunks on query, and generating grounded answers.

**This is a learning project.** The goal is not just working code — it is for the
user to *understand* every architectural decision. When implementing:
- Explain the reasoning behind choices, not just the choice.
- Surface tradeoffs and alternatives that were rejected and why.
- After a concept is worked through, record the learning in [LEARNING_NOTES.md](LEARNING_NOTES.md).

## Sources of truth

| File | Role |
|------|------|
| [rag_pipeline_spec.md](rag_pipeline_spec.md) | **The spec.** Every architectural decision + ordered implementation steps. Defer to this. |
| [LEARNING_NOTES.md](LEARNING_NOTES.md) | Running record of concepts learned. Keep it updated as we progress. |

If code and spec disagree, the spec wins — or flag the conflict before proceeding.

## Repo state

**All 6 phases complete** — the pipeline is built, run end-to-end, and the
README is written. The 5 pipeline scripts sit over 3 shared modules:
`docx_parser.py` (.docx → records), `chunker.py` (records → chunks),
`mistral_helpers.py` (client + retry). The old single-file prototype was
deleted; [mistral_basics.py](mistral_basics.py) is kept as the Hours 1–2
learning file.

Post-Week-1 work the user wants (not started): generalise `analyse.py` beyond
the CV into a multi-document-type tool — see `LEARNING_NOTES.md` "Phase 3 — How
far the pipeline generalises".

## Target architecture (per spec)

```
analyse.py        → inspect docx structure, ask Mistral to recommend chunking → config.json
review_chunks.py  → preview proposed chunks, human approves before embedding
ingest.py         → chunk + embed (mistral-embed) + store in 4 ChromaDB collections
query.py          → embed query + retrieve top-k + generate answer (Mistral)
stress_test.py    → run the full experiment matrix
```

Human-in-the-loop by design: `analyse` and `review_chunks` print and wait for `y/n` approval.

`review_chunks.py` is the spec's "Step 2 — Chunk inspector". The spec names it
`inspect.py`; that name is unusable — see the gotcha below.

## Conventions & gotchas

- **Mistral SDK v2.x** — `from mistralai.client import Mistral` (NOT `from mistralai import Mistral`).
  `requirements.txt` pins `mistralai>=2.0.0,<3.0.0`.
- **Wrap every Mistral API call** in the `call_with_retry()` pattern (see [mistral_basics.py](mistral_basics.py)).
  Retryable: 429 + 5xx. Not retryable: 400/401/404.
- **Embedding model is fixed** (`mistral-embed`, 1024-dim, L2-normalised). Changing it
  means re-indexing the whole corpus. Same model for documents and queries — always.
- **ChromaDB**: `PersistentClient(path="./chroma_db")`. Distance metric is set at
  collection creation and is **immutable** after.
- **No venv** — dependencies run in Docker (or installed globally for the basics file).
- **API key** lives in a git-ignored `.env`; never commit it. Docker Compose reads it automatically.
- Platform is **Windows / PowerShell**. Docker Desktop is installed.
- **Never name a script after a Python stdlib module.** Python puts a script's
  own directory first on `sys.path`, so a local `inspect.py` / `json.py` /
  `types.py` shadows the real module for *every* script in the project — and
  chromadb imports stdlib `inspect` on its first line. The spec's `inspect.py`
  was renamed to `review_chunks.py` for exactly this reason.

## Code style — decision comments are mandatory

This is a learning project, so the code itself must teach. Every script must carry:
- **`ARCHITECTURAL DECISION:`** comment blocks at each non-obvious choice — what was
  chosen, the alternatives rejected, and why.
- **`KEY INSIGHT:`** comments summarising what the code demonstrates / what was learned.

This matches the style of [mistral_basics.py](mistral_basics.py) (the old
`rag_pipeline.py` prototype, now deleted, also used it). When a script is rewritten
to the spec's architecture, it must keep — or exceed — that level of annotation.
Concepts worked through also get a short entry in [LEARNING_NOTES.md](LEARNING_NOTES.md).

## Running (once Docker scaffold exists)

```powershell
docker compose build
docker compose run pipeline python analyse.py data/cv.docx
docker compose run pipeline bash      # interactive shell
```

## Project phases — all complete

- ✅ **Phase 1 — Environment setup**: Docker, requirements, scaffold.
- ✅ **Phase 2 — `analyse.py`**: docx profiler + Mistral chunking recommendation.
- ✅ **Phase 3 — `chunker.py` + `review_chunks.py`** (+ `docx_parser.py`).
- ✅ **Phase 4 — `ingest.py`**: embed + store in 4 ChromaDB collections.
- ✅ **Phase 5 — `query.py`**: retrieval-augmented Q&A.
- ✅ **Phase 6 — `stress_test.py`**: 112-cell experiment matrix + findings.
- ✅ **README** written against real findings.

Open items: the README's "before making public" checklist (swap real CV for a
sample, audit `outputs/` for real CV text); the small-vs-large model comparison
was skipped on the free tier.

## A note on hierarchical CLAUDE.md

The project is currently a **flat layout** — all scripts at the repo root, with
`data/`, `outputs/`, `chroma_db/` holding data only (no code). A nested CLAUDE.md
adds value only when a subdirectory has enough of its own context to be worth
loading separately. That is not the case yet. Revisit if the project grows
(e.g. a `tests/` suite or a multi-document module gets its own conventions).
