# RAG Pipeline — Phase 2 Spec

**AI Learning Track | Partner Solution Architect Preparation**

Continuation of Phase 1 (see `rag_pipeline_spec.md` for full context and architecture decisions).
This document is the single source of truth for Phase 2 implementation.

Work in two modes:

- **Architecture conversation** — decisions and reasoning (captured here)
- **Claude Code** — implementation only, guided by the steps below

---

## Problem statement

Phase 1 left one documented gap: the profiler generalises, the chunker doesn't.

`analyse.py` correctly described two additional CVs and diagnosed its own
downstream failures. `chunker.py` ignored the config it had discovered —
its decode rules were hardcoded to the first CV's conventions. On the other
two CVs it produced whole-document mega-chunks with no error. Silent wrong
output is the worst failure mode in a data pipeline.

Phase 2 closes that gap and extends the pipeline to a second document format.

---

## What we're building

Three deliverables, in strict priority order:

**1. Config-driven chunker** — `chunker.py` reads decode rules from
`config.json` rather than hardcoding them. The profiler was already doing
the hard work. This closes the loop.

**2. Cross-document validation** — run the full pipeline on three
structurally different CVs and produce a comparison table. This is the
portfolio artifact, not the code.

**3. PDF loader (stretch)** — `pdf_parser.py` maps a PDF to the same
paragraph model that `chunker.py` already expects. Most enterprise
documents are PDFs. Even a basic implementation is a meaningful signal.

---

## Why this order

Config-driven chunker first because it's the thing Phase 1 promised and
didn't deliver. "The chunker doesn't, yet" is in the README — shipping
Phase 2 without fixing it would mean the portfolio claim is still false.

Cross-document validation second because it's the proof. Code without
evidence is just code. The comparison table turns the fix into a
demonstrable result that a recruiter can read in 10 seconds and a
technical interviewer can probe.

PDF loader third because it's an extension, not a fix. It broadens the
story only after the core story is true.

---

## Architecture decisions

### Decision 1 — Config-driven decode rules

**Phase 1 state:**
`chunker.py` contained hardcoded rules:

```python
# Hardcoded — only works for cv.docx
if rendered_size == 14:
    role = "company"
elif style_name == "Heading 3":
    role = "job_title"
```

**Phase 2 state:**
`chunker.py` reads the fingerprint rules from `config.json` at runtime:

```python
# Config-driven — works for any document analyse.py has profiled
rules = config["fingerprint_rules"]  # ordered list, checked in sequence
for rule in rules:
    if matches(paragraph, rule["signal"]):
        return rule["role"]
```

**Why the rules must be ordered:**
The Phase 1 finding still applies — Microsoft is both `Heading 3` AND
14pt. Checking size before style classifies it correctly as company.
The order is set by `analyse.py` + human approval and stored in
`config.json`. `chunker.py` trusts that order and does not re-reason it.

**What changes in `config.json`:**

```json
{
  "fingerprint_rules": [
    { "signal": "has_numPr", "role": "bullet" },
    { "signal": "rendered_size==14", "role": "company" },
    { "signal": "style==Heading 3", "role": "job_title" },
    { "signal": "style==Heading 1", "role": "section_header" }
  ]
}
```

The signal syntax must be parseable by `chunker.py`. Define a small,
explicit signal grammar — no eval(), no dynamic execution:

```python
SIGNAL_PARSERS = {
    "has_numPr":           lambda p: has_num_pr(p),
    "rendered_size=={n}":  lambda p, n: rendered_size(p) == float(n),
    "style=={name}":       lambda p, name: p.style.name == name,
    "is_bold":             lambda p: all(r.bold for r in p.runs if r.text),
}
```

**Why no eval():**
eval() executes arbitrary Python. A config file is user-editable.
Even in a personal project, the habit of not eval()-ing config values
is worth building — it's the right answer in every production context.

---

### Decision 2 — Common paragraph model

**The problem PDF loading introduces:**
`chunker.py` currently receives `python-docx` paragraph objects directly.
PDF paragraphs come from a different library with a different object shape.
If `chunker.py` knows about both, it becomes a format-specific parser —
which defeats the purpose of separating concerns.

**The solution: a common paragraph model.**
Every format-specific loader (`.docx`, `.pdf`) converts its output to
a shared intermediate representation before anything else touches it.
`chunker.py` only ever sees this model — it has no knowledge of source format.

```python
# Common paragraph model — a plain dataclass
@dataclass
class Paragraph:
    text: str
    style_name: str        # "Heading 1", "Normal", etc. (or None)
    rendered_size: float   # pt, after overrides (or None)
    is_bold: bool
    has_num_pr: bool       # True if list item
    in_table: bool
    source_format: str     # "docx" | "pdf" — for diagnostics only
```

**Where each component lives:**

```
loaders/
  docx_loader.py    → .docx → list[Paragraph]
  pdf_loader.py     → .pdf  → list[Paragraph]  (stretch)
models/
  paragraph.py      → Paragraph dataclass
chunker.py          → list[Paragraph] + config → list[Chunk]
analyse.py          → list[Paragraph] → fingerprint map → Mistral → config.json
```

**Why this matters for the RFI project:**
The RFI documents will be a mix of formats. A common paragraph model
means adding a new format is adding one loader — nothing else changes.
This is the pluggable architecture the Phase 1 spec described as
out-of-scope. Phase 2 puts the foundation in place without building
the full registry.

---

### Decision 3 — PDF parsing approach

_(Stretch goal — only if config-driven chunker and validation are complete)_

PDF is structurally different from `.docx`. There is no paragraph style
system, no `numPr` element, no inherited formatting. Structure is inferred
from visual properties: font size, font weight, position on page, whitespace.

**Library choice: `pdfplumber`**

- Extracts text with font metadata (size, bold, font name) per character
- Handles multi-column layouts better than `pypdf`
- Returns bounding box coordinates — useful if positional heuristics
  are needed for structure detection
- Install: `pdfplumber>=0.10.0`

**What the PDF loader does:**

1. Extract text blocks with font metadata via `pdfplumber`
2. Group characters into line objects by vertical position
3. Infer paragraph boundaries from line spacing and font changes
4. Map to `Paragraph` dataclass:
   - `style_name`: None (PDFs have no styles — set explicitly)
   - `rendered_size`: from font metadata
   - `is_bold`: from font name (contains "Bold") or weight
   - `has_num_pr`: detect by line prefix ("•", "-", numbers)
   - `in_table`: basic table detection by alignment

**Known limitation:**
Scanned PDFs (image-based) have no extractable text — `pdfplumber` will
return empty strings. Handling scanned PDFs requires OCR (out of scope).
`pdf_loader.py` should detect this case and raise a clear error rather
than returning empty chunks silently.

**Why pdfplumber over alternatives:**

- `pypdf`: text extraction only, no font metadata
- `pdfminer`: lower-level, verbose, more complex to use correctly
- `pymupdf`: fast and feature-rich but GPL licensed — relevant if this
  ever goes into a commercial context

---

## Repository structure changes

```
rag-pipeline/
├── docs/                        ← NEW — all documentation in one place
│   ├── SPEC.md                  ← Phase 1 spec (existing, moved)
│   ├── SPEC_PHASE2.md           ← this document
│   └── LEARNING_NOTES.md        ← master learning notes (existing, moved)
├── loaders/                     ← NEW — format-specific parsers
│   ├── __init__.py
│   ├── docx_loader.py           ← refactored from docx_parser.py
│   └── pdf_loader.py            ← NEW (stretch)
├── models/                      ← NEW — shared data models
│   ├── __init__.py
│   └── paragraph.py             ← Paragraph dataclass
├── analyse.py                   ← updated: uses loader, common model
├── chunker.py                   ← updated: config-driven decode
├── review_chunks.py             ← minor update: show config rules used
├── ingest.py                    ← minor update: accepts format flag
├── query.py                     ← unchanged
├── stress_test.py               ← unchanged
├── outputs/
│   └── phase2_validation/       ← NEW — cross-document comparison results
└── data/
    ├── sample_cv.docx           ← existing fake CV
    ├── cv2.docx                 ← second test CV (git-ignored if real)
    └── cv3.docx                 ← third test CV (git-ignored if real)
```

**Documentation separation rationale:**

- `docs/` contains all reasoning and learning artifacts
- Root contains only runnable code and config
- A reader of the repo can go straight to `docs/` for the thinking
  or straight to the scripts for the implementation
- README stays at root — it's the entry point, not a deep-dive doc

**.gitignore additions:**

```
# Real test CVs — git-ignored
data/cv2.docx
data/cv3.docx
data/*.pdf

# Phase 2 validation outputs may contain real CV text
outputs/phase2_validation/
```

---

## Implementation steps for Claude Code

Work through these in order. Do not move to the next step until the
current one produces correct output verified against known ground truth.

---

### Step 1 — Common paragraph model

**What it does:**
Creates the `Paragraph` dataclass that all loaders output and all
downstream components consume.

**Claude Code prompt:**

> Create `models/paragraph.py` with a `Paragraph` dataclass containing
> fields: text (str), style_name (str | None), rendered_size (float | None),
> is_bold (bool), has_num_pr (bool), in_table (bool), source_format (str).
> Add a `__repr__` that shows style_name, rendered_size, and the first
> 60 chars of text — useful for debugging. Add `models/__init__.py`
> exporting Paragraph. No logic beyond the dataclass.

---

### Step 2 — Refactor docx_loader

**What it does:**
Moves the existing `.docx` parsing logic into `loaders/docx_loader.py`
and updates it to output `list[Paragraph]` using the common model.
No behaviour change — same parsing logic, different output type.

**Claude Code prompt:**

> Refactor `docx_parser.py` into `loaders/docx_loader.py`. The public
> interface is a single function: `load_docx(path: str) -> list[Paragraph]`.
> Internally it does exactly what docx_parser.py does today — walk the
> table, deduplicate gridSpan cells, compute rendered font size via the
> style inheritance chain, detect numPr — but outputs Paragraph objects
> instead of raw dicts. Add `loaders/__init__.py`. Update `analyse.py`,
> `chunker.py`, and `review_chunks.py` to import from the new location.
> Run the full pipeline on sample_cv.docx and confirm chunk count matches
> Phase 1 output before proceeding.

---

### Step 3 — Config-driven chunker

**What it does:**
Replaces hardcoded decode rules in `chunker.py` with a rule engine that
reads `fingerprint_rules` from `config.json`.

**Signal grammar** (implement exactly this — no eval()):

```python
# Supported signal patterns:
"has_numPr"              # boolean field check
"rendered_size=={n}"     # float comparison
"style=={name}"          # string equality
"is_bold"                # boolean field check
```

**Claude Code prompt:**

> Update `chunker.py` to read fingerprint_rules from config.json and
> apply them in order using the signal grammar defined in the spec.
> Implement a `parse_signal(signal_str, paragraph) -> bool` function
> using explicit string matching — no eval(), no exec(). The decode
> loop applies rules in config order and returns the first match.
> If no rule matches, classify as "body_text".
>
> Verification: run on sample_cv.docx (Phase 1 CV) and confirm output
> is identical to Phase 1. Then update config.json for cv2.docx,
> run on cv2.docx, confirm chunks are sensible. Document the config
> changes needed per CV.

---

### Step 4 — Cross-document validation

**What it does:**
Runs the full pipeline (analyse → review → ingest → query) on all three
CVs and records results in a structured comparison. Produces the
portfolio artifact.

**Test questions (same across all three CVs — use generic phrasing):**

```python
questions = [
    "What was his most recent role?",
    "What companies has he worked for?",
    "What are his core skills?",
    "Has he managed teams?",              # tests synthesis
    "What is his current salary?",        # hallucination test
]
```

**Comparison table to produce** (`outputs/phase2_validation/comparison.md`):

```markdown
| CV             | Structure           | Phase 1 chunks      | Phase 2 chunks | Notes              |
| -------------- | ------------------- | ------------------- | -------------- | ------------------ |
| sample_cv.docx | Table, mixed styles | 11 role / 25 bullet | 11 / 25        | Baseline unchanged |
| cv2.docx       | ...                 | 3 mega-chunks       | ...            | ...                |
| cv3.docx       | ...                 | 4 mega-chunks       | ...            | ...                |
```

Also record per CV: hallucination refusal rate, top retrieval similarity
for answerable questions, any chunks that were too large or too small.

**Claude Code prompt:**

> Build a `validate.py` script that runs all three CVs through the
> full pipeline using their respective config.json files (stored as
> config_cv1.json, config_cv2.json, config_cv3.json) and records
> chunk counts, query results, similarity scores, and hallucination
> refusal rate for each. Output a markdown comparison table to
> `outputs/phase2_validation/comparison.md` and a JSON file with
> full results. The script should run without interaction — configs
> are pre-approved, validation is automated.

---

### Step 5 — PDF loader (stretch)

**Only start this if Steps 1–4 are complete and verified.**

**What it does:**
Implements `loaders/pdf_loader.py` that maps a PDF to `list[Paragraph]`
using `pdfplumber`. The rest of the pipeline — `analyse.py`, `chunker.py`,
`ingest.py` — is unchanged.

**Claude Code prompt:**

> Add `pdfplumber>=0.10.0` to requirements.txt. Implement
> `loaders/pdf_loader.py` with a single public function:
> `load_pdf(path: str) -> list[Paragraph]`. Use pdfplumber to extract
> text blocks with font metadata. Group into paragraphs by vertical
> position (lines within 2pt of each other = same paragraph). Map to
> Paragraph dataclass: rendered_size from font size, is_bold from font
> name containing "Bold", has_num_pr from line prefix (•, -, digit+dot).
> Detect scanned PDFs (all text empty) and raise ValueError with a clear
> message. Test on one PDF CV. Confirm analyse.py runs on it without
> modification.

---

## README updates (after all steps complete)

Add a **Cross-document validation** section after Stress Test Findings:

```markdown
## Cross-document validation (Phase 2)

| CV             | Structure                   | Phase 1 result          | Phase 2 result    |
| -------------- | --------------------------- | ----------------------- | ----------------- |
| sample_cv.docx | Table, mixed styles         | ✅ 11 clean role chunks | ✅ unchanged      |
| CV #2          | Table, Title style          | ❌ 3 mega-chunks        | ✅ N clean chunks |
| CV #3          | Plain paragraphs, bold+size | ❌ 4 mega-chunks        | ✅ N clean chunks |

Config-driven chunking: the profiler discovers structure, the human approves
the decode rules, the chunker executes them. The same code works on all three
documents — only config.json changes.
```

Update **Production upgrade path** table — config-driven chunker row moves
from "Production" to "This project" ✅.

Update **Reflections** last line from:

> "The tuning earns its keep at scale — which is exactly where this is going next."

To something that reflects Phase 2 is done.

---

## Learning notes to capture (add to LEARNING_NOTES.md after build)

These are the questions Phase 2 will answer that are worth documenting:

- Does the signal grammar approach (explicit string matching vs eval) hold
  up across real configs? Any signals that didn't fit the grammar?
- What did pdfplumber's font metadata actually look like on a real CV PDF?
  Did bold detection by font name work reliably?
- How different were the config.json files across the three CVs?
  What does that tell us about document class variability?
- Did the common paragraph model create any friction, or did it simplify
  things as expected?
- What would the pluggable strategy registry actually look like now that
  the config-driven decode is built? Is the shape clearer?

---

## Definition of done

Phase 2 is complete when:

- [ ] `chunker.py` is config-driven — no hardcoded decode rules
- [ ] Pipeline runs correctly on all three CVs with per-CV config files
- [ ] `outputs/phase2_validation/comparison.md` exists with full results
- [ ] README updated with comparison table and corrected upgrade path
- [ ] `docs/` directory created, spec and learning notes moved there
- [ ] PDF loader implemented and tested on one PDF (stretch)
- [ ] Learning notes updated with Phase 2 findings
- [ ] All changes committed with clear commit messages per step
