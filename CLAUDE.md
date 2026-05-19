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
| [SPEC_PHASE2.md](SPEC_PHASE2.md) | **Active spec — Phase 2.** Config-driven chunker, common paragraph model, PDF loader. Defer to this for current work. |
| [rag_pipeline_spec.md](rag_pipeline_spec.md) | Phase 1 spec (Week 1, complete) — architecture decisions + history. |
| [LEARNING_NOTES.md](LEARNING_NOTES.md) | Running record of concepts learned. Keep it updated as we progress. |

If code and spec disagree, the spec wins — or flag the conflict before proceeding.

## Repo state

**Phase 1 (Week 1) is complete and the repo is public**
(github.com/michelguillon/rag_pipeline_learning). The pipeline is built, run
end-to-end, README written. 5 pipeline scripts over 3 shared modules
(`docx_parser.py`, `chunker.py`, `mistral_helpers.py`); `mistral_basics.py`
kept as the Hours 1–2 learning file.

**Phase 2 is the active work — spec: [SPEC_PHASE2.md](SPEC_PHASE2.md).** It
closes the gap a 3-CV cross-test exposed: `analyse.py`'s profiler generalises,
but `chunker.py`'s decode is hardcoded to cv.docx's structure and collapses on
other CVs. Phase 2, in priority order: (1) **config-driven chunker** — decode
rules flow from `config.json`; (2) cross-document validation; (3) PDF loader
(stretch). Phase 2 also **restructures the repo** (`docs/`, `loaders/`,
`models/`) — so the file map below is the current Phase-1 layout and changes
early in Phase 2.

Data note: only `data/sample_cv.docx` (fake) is committed/public; the real CV
and friends' CVs stay git-ignored. `config.json` / `chroma_db/` are regenerable
per-document.

## Current file layout (Phase 1 — Phase 2 reorganises this)

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

## Running

```powershell
docker compose build
docker compose run pipeline python analyse.py data/cv.docx
docker compose run pipeline bash      # interactive shell
```

## Project status

**Phase 1 (Week 1) — complete.** Six implementation steps, all done:
environment/Docker · `analyse.py` · `chunker.py` + `review_chunks.py` ·
`ingest.py` · `query.py` · `stress_test.py` (112-cell matrix) · README.
(These were labelled "Phase 1–6" in earlier notes — they are *steps* of
Phase 1, not to be confused with project Phase 2 below.)

**Phase 2 — not started.** See [SPEC_PHASE2.md](SPEC_PHASE2.md): config-driven
chunker, a common `Paragraph` model + `loaders/`, cross-document validation,
PDF loader (stretch).

Carried-over note: the small-vs-large model stress comparison was skipped on
the free tier.

## A note on hierarchical CLAUDE.md

Phase 1 was a flat layout. Phase 2 adds `docs/`, `loaders/`, `models/` — all
small, single-purpose directories. A nested CLAUDE.md earns its place only when
a subdirectory carries enough of its own context to load separately; that is
still not the case. Revisit if any of those directories grows its own conventions.
