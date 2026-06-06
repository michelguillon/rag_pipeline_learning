# CLAUDE.md — RAG Pipeline

## What this is

A document Q&A system (RAG) built with **Mistral API + ChromaDB**, ingesting a CV,
retrieving relevant chunks on query, and generating grounded answers.

**This is a learning project.** The goal is not just working code — it is for the
user to *understand* every architectural decision. When implementing:
- Explain the reasoning behind choices, not just the choice.
- Surface tradeoffs and alternatives that were rejected and why.
- After a concept is worked through, record the learning in [docs/LEARNING_NOTES_RAG_PIPELINE.md](docs/LEARNING_NOTES_RAG_PIPELINE.md).

## Sources of truth

| File | Role |
|------|------|
| [docs/SPEC_RAG_PIPELINE.md](docs/SPEC_RAG_PIPELINE.md) | **The spec.** Single rolling document — Part 1 = Phase 1 (Week 1), Part 2 = Phase 2. New phases append new parts. |
| [docs/LEARNING_NOTES_RAG_PIPELINE.md](docs/LEARNING_NOTES_RAG_PIPELINE.md) | **The learning notes.** Single rolling document — append new findings here. |

If code and spec disagree, the spec wins — or flag the conflict before proceeding.

## Repo state

**Phases 1 and 2 are both complete and merged to `main`. The repo is public**
([github.com/michelguillon/rag_pipeline_learning](https://github.com/michelguillon/rag_pipeline_learning)).

- **Phase 1** built the end-to-end pipeline on a single CV: analyse → review → ingest → query, plus the 112-cell stress test.
- **Phase 2** closed the gap the cross-CV test exposed: the chunker is now config-driven, the loader stack has a common `Paragraph` model, three structurally-different CVs validate cleanly, and a `pdfplumber` PDF loader plugs in without any change to the chunker.

No phase is actively under construction. When the next phase starts, it gets appended as a new Part of [docs/SPEC_RAG_PIPELINE.md](docs/SPEC_RAG_PIPELINE.md).

Data note: only `data/sample_cv.docx` (fake) is committed/public; the real CV
and friends' CVs stay git-ignored. `data/*.pdf` is git-ignored too (drop a PDF
in to test the loader). `config_cv*.json` / `chroma_db/` are regenerable
per-document; `outputs/phase2_validation/` is git-ignored because it carries
real CV text.

## File layout

```
analyse.py           profile a doc, ask Mistral for fingerprint_rules (config-driven decode)
chunker.py           rule-engine decode + unit grouping + 2 strategies (A, A2)
review_chunks.py     preview chunks before any embedding spend
ingest.py            embed (mistral-embed) + store in per-CV ChromaDB collections
query.py             embed query + retrieve top-k + generate grounded answer
stress_test.py       Phase 1's 112-cell experiment matrix
validate.py          Phase 2 cross-document validation (3 CVs, fixed Q set)
loaders/
  docx_loader.py     .docx -> list[Paragraph]
  pdf_loader.py      .pdf  -> list[Paragraph]   (pdfplumber, words → lines → paragraphs)
  __init__.py        load() dispatcher (picks loader by file extension)
models/
  paragraph.py       the common Paragraph dataclass
config_cv1.json      per-CV configs — fingerprint_rules + collections + prefix template
config_cv2.json
config_cv3.json
docs/                spec + learning notes
```

Human-in-the-loop by design: `analyse.py` and `review_chunks.py` print and wait for `y/n` approval. `validate.py` runs without interaction — its configs are pre-approved.

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
- **Signal grammar is a decision LIST, not an expression tree.** `chunker.parse_signal` accepts four single signals (`has_numPr`, `is_bold`, `rendered_size==n`, `style==name`) and explicitly rejects `&&`/`||`. Use ordered single-signal rules; let order disambiguate overlaps. See [docs/LEARNING_NOTES_RAG_PIPELINE.md](docs/LEARNING_NOTES_RAG_PIPELINE.md) "Phase 2 — Decision lists vs expression trees".
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
Concepts worked through also get a short entry in [docs/LEARNING_NOTES_RAG_PIPELINE.md](docs/LEARNING_NOTES_RAG_PIPELINE.md).

## Running

```powershell
docker compose build
docker compose run pipeline python analyse.py data/sample_cv.docx     # profile + recommend
docker compose run pipeline python analyse.py data/sample_cv.docx --profile-only   # no API call
docker compose run pipeline python review_chunks.py data/sample_cv.docx
docker compose run pipeline python ingest.py data/sample_cv.docx
docker compose run pipeline python validate.py                         # Phase 2 cross-doc run
docker compose run pipeline bash                                       # interactive shell
```

## Project status

**Phase 1 (Week 1) — complete.** Pipeline + 112-cell stress test + README. The
small-vs-large model stress comparison was skipped on the free tier.

**Phase 2 — complete.** Config-driven chunker (signal grammar, no eval), common
`Paragraph` model + `loaders/`, cross-document validation on 3 structurally-
different CVs, PDF loader via pdfplumber. The Phase 1 cross-CV failure (mega-
chunks on cv2/cv3) is gone — only `fingerprint_rules` changes between CVs.

**No active phase.** The next phase will be written up as a new Part in
[docs/SPEC_RAG_PIPELINE.md](docs/SPEC_RAG_PIPELINE.md). The current branch is
`feature/rfi-pipeline` — name is historical; consider renaming/closing if no
further work lands on it.

## A note on hierarchical CLAUDE.md

`docs/`, `loaders/`, `models/` are small, single-purpose directories. A nested
CLAUDE.md earns its place only when a subdirectory carries enough of its own
context to load separately; that is still not the case. Revisit if any of
those directories grows its own conventions.
