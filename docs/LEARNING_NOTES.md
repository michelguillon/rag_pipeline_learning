# Mistral API — Learning Notes

Personal reference from working through `mistral_basics.py` (Hours 1–2),
before building the RAG pipeline. Concepts, gotchas, and what each
experiment taught.

---

## Setup & environment

- **Python** 3.13, packages installed globally (user site-packages), not in a venv.
- **Install:** `py -m pip install -r requirements.txt`
- **API key** is read from an environment variable. Set it per terminal session:
  ```powershell
  $env:MISTRAL_API_KEY = "your-key-here"
  ```
- **mistralai v2.x is a breaking rewrite of v1.x.** The `Mistral` class moved:
  - v1: `from mistralai import Mistral`
  - v2: `from mistralai.client import Mistral`
  - `requirements.txt` is pinned `mistralai>=2.0.0,<3.0.0` to avoid a future v3 surprise.

## Running the experiments

```powershell
python mistral_basics.py              # all experiments
python mistral_basics.py --exp 6      # just one
python mistral_basics.py --exp 1,3,6  # a subset, in that order
python mistral_basics.py --help       # usage
```
Running fewer experiments = fewer API tokens spent (matters on the free tier).

---

## The two model types (don't confuse them)

| Job | Model | Notes |
|-----|-------|-------|
| Text → vectors (embeddings) | `mistral-embed` | Used for **both** documents and queries |
| Generate an answer | `mistral-small` / `mistral-large` | Swappable any time |

These are **always different models**. The only hard rule:
the embedding model used to index documents **must equal** the one used
to embed queries — vectors from different models live in different spaces
and aren't comparable.

---

## What each experiment taught

### 1. Basic completion
`client.chat.complete(model=, messages=[...])` → `response.choices[0].message.content`.
`response.usage` reports `prompt_tokens`, `completion_tokens`, `total_tokens`.
**`prompt_tokens` = the cost of your INPUT.** In RAG, the prompt grows with
every context chunk you add — watch it.

### 2. System prompt influence
The system prompt is the primary lever for controlling model behaviour.
In RAG, use it to instruct the model to answer **only from the provided
context**, not from its training data.

### 3. Temperature effects
- `temperature=0.0` → near-deterministic, consistent answers.
- `temperature=1.0` → creative but high variance.
- **For RAG Q&A: 0.0–0.2.** For creative tasks: 0.7–1.0.

### 4. Context window & token costs
Every chunk added to the prompt costs tokens, every call.
Rough pricing: mistral-small ~\$0.2 / 1M tokens, mistral-large ~\$2 / 1M.
More retrieved context = better answers but higher cost — a real tradeoff.

### 5. Streaming
`client.chat.stream(...)` yields chunks: `chunk.data.choices[0].delta.content`.
Streaming only changes **delivery** (token-by-token vs all at once) — it is
**still stateless**, same as `complete`. Good for UX; in RAG you usually
stream only the final answer-generation step.

### 6. Embeddings & semantic similarity
`client.embeddings.create(model="mistral-embed", inputs=[...])`.
- An embedding **is a vector** — a fixed-length list of floats
  (`mistral-embed` → **1024 floats**), one per input text.
- A batch of N inputs returns N vectors, in order, in `response.data`.
- mistral-embed vectors are **L2-normalised** (length ≈ 1.0), so cosine
  similarity = dot product.
- **Cosine similarity** measures meaning-closeness: similar sentences score
  HIGH, unrelated ones LOW. This *is* RAG retrieval — a vector store
  (ChromaDB) does this across all stored chunks and returns the top-k.

### 7. Error handling & retry-with-backoff
- The SDK raises `MistralError` (from `mistralai.client.errors`) with a
  `.status_code`, `.headers`, `.body`.
- **Retryable:** `429` (rate limit) and `5xx` (transient server errors).
- **Not retryable:** `400` / `401` / `404` — that's your bug; retrying wastes time.
- Back off **exponentially** between retries; honour a `Retry-After` header
  if the server sends one.
- The reusable `call_with_retry()` helper in `mistral_basics.py` is meant to
  be lifted into `rag_pipeline.py` — wrap *every* API call with it.

---

## Key concepts to carry into RAG

- **The API is stateless.** Each call is independent; the model remembers
  nothing. In RAG you rebuild and resend `system prompt + retrieved context
  + question` on every single call. There is no "session".
- **Provider-side conversations exist** (e.g. Mistral's Conversations API)
  but RAG deliberately uses the stateless model — for control over what's in
  context, token-cost management, and provider portability.
- **The embedding model is the one "sticky" choice.** Changing it later means
  re-embedding (re-indexing) the entire corpus. The generation model can be
  swapped freely. Translating text or storing multiple embedding sets does
  *not* get around this.
- **Watch the token budget.** The basics file is cheap (~15–20k tokens total).
  The real spend is **indexing** — embedding a whole document corpus chunk by
  chunk.

---

## Phase 1 — Environment setup (RAG pipeline build)

Docker scaffold, before any pipeline code. Pitched at solution-architect level —
deliberately light on developer mechanics.

- **Docker is a portability decision.** The same image runs on the Windows dev
  laptop and the home server unchanged — "build once, deploy anywhere". In a
  client conversation, this is what takes "works on my machine" off the risk
  register.
- **Secrets never live in the artifact.** The Mistral API key stays in a
  git-ignored `.env` file and is injected at runtime — never committed, never
  baked into the Docker image. The committed `.env.example` records *what* is
  needed without exposing the secret itself.
- **Validate the environment before building on it.** Phase 1 caught a real
  defect early — a script filename collided with a Python standard-library
  module and would have broken every script. The architect takeaway (not the
  fix): integration assumptions get tested first and cheaply; a five-minute
  check beats a day of debugging later.
- **Human-in-the-loop is an architecture choice.** The pipeline splits into
  analyse → review → ingest → query so a human approves the chunking config
  before any spend on embeddings. Enterprise RAG inserts this checkpoint
  because automatic misconfiguration is costly to undo.

---

## Phase 2 — Document analysis: real documents lie about their structure

The most useful finding so far, and an interview-grade one.

**The pitfall.** A Word document's *visual* hierarchy and its *underlying*
markup are two different things. The CV looks cleanly structured — sections,
companies, job titles, bullets. Under the hood the author built that hierarchy
two ways at once: proper heading *styles* for sections, but plain *direct
formatting* (font size 14 pt) for company names — applied on top of whatever
paragraph style happened to be there (Normal, Heading 3, Heading 4).

**Why it matters.** A naive parser keys off paragraph style alone ("Heading 3 =
boundary"). On this document that silently fails: company names aren't heading-
styled at all, and one company ("Microsoft") shares the exact style used for job
titles. The parser produces wrong chunks and nobody notices until retrieval
quality is poor.

**The reliable read needs three signals combined**, not one:
paragraph style + rendered font size + presence of a list element (`numPr`).

**The architectural response.** Don't hardcode "14 pt = company" — that is
*this* document. Build the analysis step to *profile* the document instead:
enumerate every distinct formatting fingerprint (style, size, bold, list-or-not)
with counts and samples, and emit a consistency report flagging where styling
is inconsistent. A human + the LLM then map fingerprints to roles. This makes
the analyser document-agnostic — it discovers structure rather than assuming it,
so the same tool works on a CV, an RFI, or an RFP.

**The client takeaway.** "Will your RAG work on our documents?" — the honest
answer is *not until you have profiled them*. Document-structure inconsistency
is a near-universal, under-estimated source of RAG quality problems, and it is
invisible if you only look at the rendered page.

---

## Phase 2 — Metadata is what makes RAG scale

Each chunk is stored with structured metadata (company, role, section, dates,
is_bullet) alongside its embedding. At this project's scale (~25 chunks) that
metadata is optional polish. At enterprise scale it is load-bearing.

**Why.** Vector search alone scans every vector. With metadata you can
*pre-filter*: search only the chunks where `company = "Microsoft"` or
`date >= 2023`, then rank within that subset. On millions of chunks, across
multiple tenants, that filter is the difference between fast, precise retrieval
and slow, noisy retrieval — and it is also how access control is enforced
(tenant A must never retrieve tenant B's chunks).

**The design choice here.** The pipeline attaches metadata from `ingest.py`
onward even though this corpus is too small to need filtering — so the
capability is built in, not retrofitted. Honest enterprise framing:
metadata-filtered retrieval is usually a bigger lever on real-world RAG quality
than tuning chunk size or top_k.

**Enhancement noted:** richer metadata extraction — reliable date parsing,
entity normalisation, tenant tags — is the natural next step. Deferred here
only because this CV does not contain enough structured data (notably dates)
to justify it yet.

---

## Phase 2 — Verification, and why silent failures are the dangerous ones

Building `analyse.py` surfaced four bugs. The ones worth keeping are kept not
for the fixes but for the failure *modes*.

**A silent wrong answer is more dangerous than a loud crash.** Two bugs made
the parser return wrong-but-plausible output with no error. First it read only
13 of 43 paragraphs — a de-duplication check keyed on a Python memory address,
which the runtime quietly reuses for new objects. Then the fix over-corrected
and read merged cells repeatedly: 11 headings where the document has 6. Both
"ran fine". A crash announces itself; a parser that quietly drops two-thirds of
a document just makes the RAG system answer confidently from incomplete data.
What catches this: checking output against known ground truth (here, a manual
probe of the document), not just "did it run" — and a fix is not verified by
being *different*, only by being *correct*.

**The container boundary is a state boundary.** `analyse.py` reported "Wrote
config.json" — truthfully — yet the file vanished: it was written inside an
ephemeral container whose filesystem is discarded on exit, and the project
directory was not mounted. Anything a container writes is lost unless it lands
on a mounted volume. This generalises to every ephemeral-compute environment
(containers, serverless, CI runners): durable state must be explicitly
externalised, and a "success" log line is not proof that anything durable was
produced.

---

## Phase 3 — Semantic chunking: why there is no chunk size or overlap

A natural question reading `analyse.py`'s output: where are the chunk-size and
overlap parameters? They are absent on purpose.

**Fixed-size chunking** slides a window of N tokens across the text with M
tokens of overlap. Overlap exists only to rescue a sentence cut in half by an
*arbitrary* window boundary. Size and overlap are input parameters you tune —
the right approach for unstructured prose.

**Semantic (structure-based) chunking** — what this pipeline uses — makes each
chunk a complete structural unit (one bullet, or one role). Consequences:
- No chunk-size parameter: a chunk is exactly as long as its bullet.
- No overlap: chunks are never split mid-idea, so there is nothing to rescue.
  Overlap is a fixed-window artifact; semantic chunking removes the need for it.
- The job overlap would do — keeping a fragment self-contained — is instead
  done by the **context prefix** (`[company | title | dates]`): semantically,
  not by duplicating text.
- Chunk size becomes an *observed outcome*, not an input. You see the real
  word counts at the review step and check nothing is too thin or too fat —
  you do not set them up front.

**Where each chunking decision lives in the code:**
- philosophy (semantic, two strategies, no overlap) → spec Decision 2;
- per-document config (strategy, boundary signals, prefix) → `config.json`;
- the logic that executes it (decode → group → prefix → metadata) →
  `chunker.py`;
- the observed result (actual chunk text + word counts) → surfaced for human
  review by `review_chunks.py`.

---

## Phase 3 — How far the pipeline generalises across document types

Question raised: what happens if you feed this pipeline a product / prose
document (e.g. a Markdown spec) instead of a CV? Two separate answers.

**Format.** `analyse.py` uses `python-docx`, which only opens `.docx`. A `.md`
file crashes immediately. Generalising would need a format-agnostic loader
(Markdown / PDF / HTML → a common paragraph model).

**Strategy.** Even given a `.docx` prose document, only half the analyser
generalises:
- The **profiler** (formatting fingerprints, consistency report, section
  detection) is genuinely document-agnostic — it describes whatever it is given.
- The **chunking strategy menu** is not. Strategies A ("one chunk per role")
  and A.2 ("one chunk per bullet"), and the Mistral prompt itself, are coded
  for a CV. On a product doc the analyser still runs and still recommends one
  of them — but the recommendation does not fit. Running and being wrong is
  worse than crashing.

**Chunking is a spectrum, not prose-vs-CV.** "Prose → fixed-size + overlap" is
only half right:
- *Structured* prose (clean `#`/`##`/`###` headings) → semantic chunking *by
  section*. Markdown is easier to chunk than the messy CV — heading levels are
  unambiguous.
- *Genuinely unstructured* text (transcript, OCR output, a wall of text) →
  fixed-size + overlap, because there is no structure to chunk on.
- Production reality is a **hybrid**: split on semantic boundaries first; if a
  section exceeds a size cap, sub-split *that section* with overlap.
  Semantic-first, fixed-size as the fallback within an over-long unit.

**The honest generalisation claim.** "Reusable for other document types" holds
for the *profiler*, not for the *strategy set*. True multi-type support needs a
pluggable strategy registry keyed by document class (CV, structured doc,
unstructured text), each with its own boundary rules and size-fallback policy.
Out of scope for Week 1 (single CV) — but this is the real shape of it.

---

## Phase 3 — The human reviewer catches what the pipeline cannot

`review_chunks.py` exists to preview chunks before any embedding spend. Its
first real run earned its keep — but not in the expected way.

The chunk previews looked plausible: roles, bullets and prefixes all present.
What was *missing* — every job's dates — was invisible to the pipeline, because
the parser did not know dates existed to look for. It was the human, reading
the preview and knowing the source CV, who said "the dates should be there."

Two lessons:
- **Structural assumptions must be checked against the whole document, not a
  sample.** The parser assumed every table row was one merged cell — true for
  the row that had been inspected, false for the role rows, which carry a
  second date column. Sampling one row to infer a table's shape is the same
  trap as sampling the rendered page to infer a document's structure (Phase 2).
- **A human-in-the-loop review is a verification layer automated checks cannot
  replace.** Automated checks confirm the pipeline did what it was told; only
  someone who knows the source can catch what the pipeline was never told to
  look for. That is the whole point of the review gate.

---

## Phase 4 — One embedding, many indexes; the metric is a query-time choice

`ingest.py` embeds 36 chunks and stores them across 4 ChromaDB collections.
Two points worth keeping.

**You embed once, not once per collection.** `cv_role_cosine` and `cv_role_l2`
hold the *identical* vectors — embedding a chunk is independent of how
distances are later measured. The distance metric (cosine / L2) is set on the
collection at creation and applied at *query* time; it is not baked into the
vector. So the four collections cost only two embedding passes (11 role chunks,
25 bullet chunks), not four. Embedding is the expensive step — and the only one
that forces a full re-index when it changes — so never pay for it twice.

**mistral-embed vectors are L2-normalised — confirmed in the pipeline.** A
stored vector measured 1024-dim with an L2 norm of exactly 1.0000. Two
consequences the experiment rests on: cosine similarity equals the dot product
for these vectors, and cosine vs L2 rankings differ only subtly — which is
precisely why testing both metrics is interesting rather than obvious.

---

## Phase 5 — Retrieval feeds generation; it need not be perfect

`query.py`: embed question → retrieve top-k → ground a Mistral answer.

**Observed.** "What did he do at Microsoft?" against the role collection
retrieved the Microsoft chunk first (sim 0.78) — but `top_k=3` also pulled in
Education and Profile (sim ~0.71), neither relevant. The answer was still
correct: told to answer only from context, the model grounded on the Microsoft
chunk and ignored the filler.

The lesson: retrieval and generation are a *pipeline*, not one step. Retrieval
only has to get the right chunk *into* the set; the generation step can
tolerate some irrelevant neighbours. But that tolerance is not free — every
extra chunk costs prompt tokens and adds a distraction risk. That tension is
exactly what top_k trades off, and what the Phase 6 stress test measures.

**The hallucination guard is a measurable signal.** "What is his current
salary?" returned the fixed fallback phrase *verbatim*. Because the phrase is
exact, any other response to an out-of-scope question is a detectable
hallucination — which is what makes the stress test scoreable.

---

## Phase 6 — Stress test: findings, and why a batch job must checkpoint

**The bug worth keeping.** The first stress-test run completed 91 of 112 calls,
then the free tier returned "service tier capacity exceeded" and the retries
exhausted. `stress_test.py` saved results only at the end — so all 91 successes
were lost. A long batch job against a rate-limited API MUST checkpoint:
`stress_test.py` now writes progress after every cell and resumes on re-run.
One failure at call 91 must never discard 90 successes. (Same family as the
Phase 2 "container boundary" lesson — durable state, written as you go.)

**Findings** (mistral-small; 4 collections × top_k × flat/labelled × 7
questions = 112 cells). The small-vs-large model comparison was skipped to stay
within free-tier limits — a known gap.

- *Hallucination refusal: 16/16 cells, 2/2 each — 100%.* Every out-of-scope
  question got the exact fallback phrase, in every configuration. The fixed
  instruction + exact-phrase design is the strongest, most robust result here.
- *top_k drives token cost, roughly linearly.* role: ~250 tokens at k=1 →
  ~480 at k=3; bullet: ~170 at k=1 → ~480 at k=5. Retrieved context is paid
  for on every call.
- *Labelled context costs ~10–25% more tokens than flat* — the `[Source: …]`
  tags are not free, and did not visibly change answer quality on this corpus.
- *Strategy A vs A2 produced near-equivalent answers.* The spec expected A
  (whole-role chunks) to beat A2 (per-bullet) on synthesis questions. It did
  not, clearly: with top_k=5, A2's bullet chunks reconstruct enough context
  that the synthesis answer matches A's; for a precise-fact question both gave
  an identical answer at k=1.

**The meta-finding.** On a small (~36-chunk), clean, single-document corpus the
RAG pipeline is forgiving — strategy, metric and format choices move token cost
far more than answer quality. The *discipline* (grounding instruction, human
review, checkpointing) mattered more than the *tuning*. The tuning earns its
keep at scale, with noisier retrieval — which is exactly where the A/A2 and
metric tradeoffs stop being academic.

---

## Cross-CV test — three real CVs, three different documents

Ran the pipeline on two more CVs (friends') to test generalisation — a clean,
concrete confirmation of the Phase 3 prediction.

| CV | Structure | Heading styles | Dates | Strategy-A result |
|----|-----------|---------------|-------|-------------------|
| cv.docx (built for) | 1 table, merged-cell rows | Heading 1/3/4 | separate column | 11 clean role chunks |
| CV #2 | 1 table | none — uses the `Title` style | inline on the title line | 3 chunks, one of 874 words |
| CV #3 | no table, plain paragraphs | none — bold + font size | inline | 4 chunks, one of 749 words |

**What generalised.** Bullet detection by `numPr` worked on all three (18 / 35
/ 24 bullets) — Strategy A2 ran everywhere. And `analyse.py`'s profiler +
consistency report described all three correctly; for both new CVs it
explicitly flagged the cause of failure — *"paragraphs larger than body text
but using NO heading style — a style-only parser would miss these as
boundaries."* The analysis layer is genuinely document-agnostic; it even
diagnosed its own downstream failure.

**What did not generalise.** Sections, companies, job titles and dates are
encoded a different way in every CV — Heading styles vs the `Title` style vs
bold+size; a date column vs dates inline. `chunker.py`'s decode is hardcoded to
the first CV's conventions, so on the other two it found no role boundaries and
Strategy A collapsed into whole-CV mega-chunks.

**The failure mode is the dangerous one.** Nothing crashed. The pipeline
produced chunks, embedded them, stored them — all "successfully". A 749-word
"chunk" that is really an entire CV would quietly wreck retrieval. Running and
being wrong beats crashing only in appearance.

**Ingestion was never the risk.** `ingest.py` embedded and stored every CV's
chunks without trouble — it is content-agnostic. The fragile step is chunking,
upstream of it.

**The fix shape (post-Week-1).** `analyse.py` already detects enough to chunk
any of these CVs — it just is not believed. The decode signals (what marks a
section / company / title / date) must travel in `config.json`, derived
per-document by `analyse.py` + the human, and be consumed by `chunker.py` —
instead of being hardcoded. That is the concrete shape of the "generalise the
analyser" work.

---

---

## Phase 2 — Config-driven decode: implicit vs explicit, not hardcoded vs adaptive

Phase 2's headline was "make the chunker config-driven". The actual framing
worth keeping is subtler than the headline.

**The Phase 1 chunker was not literally hardcoded.** It computed two values
at runtime — the document's modal body size and its top heading style — and
used them in a decision tree (`size > body_size` → company header,
`style.startswith("Heading")` → title, etc.). It *adapted* per document. So the
critique "it's hardcoded" is too easy.

**The real critique: the decode rules were implicit and code-resident.** They
were never *declared*, *inspectable*, or *per-document-overridable*. `analyse.py`
profiled the document and emitted a `config.json` describing its structure,
but `chunker.py` mostly ignored that file and re-derived its own logic from
heuristics. The cross-CV failure was not because heuristics fail in principle —
it was because nothing could override them when they did.

**Phase 2 closes the loop by moving the decode rules to the config file.**
`analyse.py` proposes an ordered `fingerprint_rules` list, a human reviews +
approves, `chunker.py` simply executes. Same code, three structurally-
different CVs, only the config changes. The architectural shift is from
"implicit code that adapts via heuristics" to "explicit config the code
trusts". Both can be correct on the document they were tuned for; only the
explicit form survives contact with a documented client conversation about
*why* the system classifies what it classifies.

---

## Phase 2 — Decision lists vs expression trees: a grammar choice that pays for itself

The signal grammar is deliberately tiny. Four single-condition signals:
`has_numPr`, `is_bold`, `rendered_size==<n>`, `style==<name>`. No `&&`, no
`||`, no negation. The chunker rejects compound signals at the parser.

That looks restrictive — and the obvious instinct (Mistral's, and most
engineers') is to want `rendered_size==14 && is_bold` to disambiguate "this
14pt thing that's also bold is a company; that 14pt thing without bold is
something else". The grammar deliberately doesn't allow it. Why?

**The grammar choice isn't really expression-tree-vs-list of conditions.
It's ordered-list-with-first-match-wins vs everything-is-a-predicate.** A
decision list — firewall rules, CSS specificity, legal precedent —
discharges its ambiguity through ORDER, not through conjunction. If a
paragraph satisfies two signals at once, the rule that should win goes
first. `rendered_size==14 -> company` placed BEFORE
`style==Heading 3 -> job_title` correctly classifies the 14pt-Heading-3
company *while still* letting an 11pt Heading 3 fall through to job_title.
The conjunction is implicit in the ordering.

**Why this matters beyond aesthetic.**

- *Every rule is independently readable.* A reviewer can point at rule 4
  and say "this catches Microsoft because of X" without mentally
  evaluating a boolean tree. That property scales — to clients reading
  configs, to colleagues debugging, to future-you in six months.
- *The smaller the grammar, the smaller the prompt.* `analyse.py` instructs
  Mistral on the grammar; a grammar with `&&`/`||`/`!`/precedence rules
  needs paragraphs of explanation. The single-signal-with-ordering grammar
  fits in three lines.
- *Validation is trivial.* A no-`eval()` parser for four shapes is ten
  lines of `if/elif`. A boolean expression parser is fifty lines plus a
  lexer plus operator-precedence tests — and every operator added is a
  new attack surface for a config-author typo.
- *Adding an operator is a slippery slope.* Today `&&`. Tomorrow `||`,
  then `!`, then parenthesisation, then a tokeniser. The grammar has to
  draw the line somewhere; the question is whether the line earns its
  keep. Three different CV structures passed through the existing four
  signals + ordering — the cross-document validation IS the empirical
  argument that the line is in the right place for this document class.

**When this would be the wrong choice.** Real expression trees earn their
keep when two predicates are genuinely independent — when a rule needs
`(A or B) and (C or D)` and neither half implies the other. In that case,
decision-list ordering becomes contorted. We may hit that on document
classes far enough from CVs that the natural unit of hierarchy isn't a
single dominant signal (e.g. tabular financial reports where "right-aligned
numeric column" *and* "row of horizontal line" both matter). When that
happens, the grammar gets extended deliberately with one new operator,
justified by a concrete document that needed it — not by an LLM's
instinct.

---

## Phase 2 — When the LLM won't honour your grammar (and that's fine)

This is the Phase 2 finding most likely to survive contact with a paying
client. It is also the most uncomfortable one to publish.

**What happened.** `analyse.py` asks Mistral to produce `fingerprint_rules`
under the four-signal grammar. The prompt names the grammar explicitly,
lists the supported signals, instructs *each rule has exactly one signal*,
adds a worked example showing how ordering replaces compound conditions,
and finishes with "DO NOT write compound signals — the parser will reject
them".

Mistral produced compound `&&` signals on three consecutive iterations.

The strictness of the prompt did not move the model. The second attempt
silently corrupted the validation: a rule of the shape
`style==Heading 3 && rendered_size==14` *parsed* as a literal style name
that would never match anything — a rule that does nothing without
announcing itself. Phase 2's design principle ("silent wrong output is the
worst failure mode") made me strengthen `parse_signal` to reject any
`&&`/`||` token outright, regardless of the leading signal.

**The takeaway is not "Mistral is bad at instructions".** It is that the
LLM has a strong prior — classification rules want conjunctions — and a
short prompt cannot fully override it. The right response is not to fight
that for another five iterations. It is to design the system so that
load-bearing constraints are enforced *outside* the prompt.

Three concrete defences came out of this finding, all worth keeping:

1. **A no-`eval()` parser, exhaustive on the grammar.** Anything not in
   the grammar raises, loudly. If the LLM invents an operator, the
   parser refuses it before `chunker.py` ever sees the config.
2. **Validate at config-WRITE time, not chunk time.** `analyse.py` runs
   every proposed rule through `chunker.parse_signal` against a dummy
   `Paragraph` before writing `config.json`. A broken rule fails *while
   the human is looking at the proposed config*, not three steps later
   inside `ingest.py`. The human can choose to fix the LLM output or
   author rules manually; either way the bad config never lands.
3. **The "Mistral proposes, human approves" loop is load-bearing.** The
   spec calls this out as an architecture decision. After three failed
   iterations on this single document, "approves" became "writes the
   rules from the profile and ignores Mistral's structural ones, while
   keeping its strategy/metadata/prefix recommendations". That is the
   workflow the spec intends. Treating it as ceremonial — running with
   `--yes` and hoping — would have silently produced a broken pipeline.

**The harder, broader lesson.** When you build an agentic pipeline whose
intermediate artifacts are produced by an LLM and consumed by code, the
LLM is going to surprise you. Sometimes it surprises in the direction you
wanted (Mistral correctly diagnosed every document's structure, all three
times). Sometimes it doubles down on a prior the prompt explicitly
contradicts. The discipline that makes this productive — strict
validation, schema-enforced output, fall-back-to-human paths — is the
same discipline that makes any *non-LLM* untrusted-input pipeline
productive. The LLM is just a more eloquent untrusted input.

---

## Phase 2 — The common paragraph model: "pluggable" earns its keep

Phase 1's spec described a "format-agnostic loader" and a "pluggable
strategy registry" as out-of-scope for Week 1. Phase 2 built the loader
half. The pluggability claim got tested when I added the PDF loader.

**What it cost to add `.pdf` support, in lines of code touched outside
the new loader:**

- `loaders/__init__.py` — one new import line, one new entry in the
  `_LOADERS_BY_EXT` dispatch dict.
- `requirements.txt` — one line (`pdfplumber>=0.10.0`).

**What was NOT touched: `chunker.py`. `analyse.py` only changed because
PDF paragraphs legitimately have `style_name=None`, and the existing
`r.style_name.startswith("Heading")` calls weren't None-safe. That bug
was Phase-1 latent — not a structural change, just a fix made visible.**

This is the empirical test of the common-model claim. If chunking had any
branching on file format, it would have shown up here as
`if source_format == 'pdf': …` landing in `chunker.py`. It didn't. The
common `Paragraph` dataclass quarantines every format-specific concern in
its loader, and every downstream component reads the same shape.

**The detail that matters.** This works only because `Paragraph` is
*content-aware enough*. The original spec model had seven formatting
fields. We added two pragmatic extras during the Phase 2 architecture
conversation — `date` (load-bearing for the docx loader's table-column
pairing) and `override` (used by `analyse.py`'s consistency report C3).
PDFs leave them at defaults. Without those two, the docx loader would
have lost capability or needed parallel data structures — exactly the
"format-specific shape leaks into the model" failure pluggability is
supposed to prevent. The architecture conversation that surfaced this
question — "should the common model carry content-derived fields or
only formatting fields?" — was the single highest-leverage decision in
Phase 2.

**Hand-wave-able generalisation.** Adding a third format (HTML, RTF,
ePub) follows the same shape: one new loader, one dispatch entry. The
interesting cost is not in the wiring — it is in heuristic quality (how
good is the format-to-`Paragraph` translation?), which is loader-specific
work.

---

## Phase 2 — Same content, two formats: PDFs survive the conversion better than expected

The PDF loader was tested on a `.pdf` rendering of `sample_cv.docx` (same
content, two formats). The comparison is the clearest demonstration of
the pluggable-loader claim and worth keeping for client conversations.

| Metric                | .docx  | .pdf   | Δ |
|-----------------------|-------:|-------:|---|
| paragraphs extracted  | 48     | 50     | +2 ("Page 1 of 2" footer ×2 pages) |
| bullets recovered     | **21** | **21** | 0 |
| avg words / paragraph | 13     | 13     | 0 |
| Strategy A chunks     | **11** | **11** | 0 |
| Strategy A2 chunks    | 28     | 29     | +1 |
| companies captured    | 5      | 6      | +1 |

**The headline:** bullet count identical, A-chunk count identical, average
paragraph word count identical. The PDF loader's word-line-paragraph
reconstruction matches what the `.docx`'s explicit `<w:numPr>` and
paragraph elements knew directly.

**The mechanism that made bullets work.** PDF lists rendered as a literal
`-` glyph in one font (Cambria), with the bullet text in another (Calibri),
both at approximately the same y-position. The PDF loader treats *any line
whose leftmost word is a list marker* as an unconditional paragraph break —
the PDF analogue of `<w:numPr>`. A line of body text wrapping onto a second
line is not a new paragraph; a `-` at the left margin always is. That
single heuristic — list-marker-as-hard-paragraph-break, before any
gap-based reasoning — recovered every bullet in this CV.

**The +1 result is the loader being slightly *better* than the .docx
config.** The PDF caught a 12pt+bold "Senior Customer Engineer 2011-2014"
outlier as a job title (the rule `rendered_size==12 -> job_title` fired).
The `.docx` config (which uses `style==Heading 3 -> job_title`) doesn't
have a rule for the 12pt+bold case and let it fall to body text. Same
document, two formats, two configs — each tuned to the signals its loader
exposes. The lesson is not "PDF is better" but "the per-format config is
where the loader's strengths and limits show up; the chunker stays
unchanged".

**The +2 paragraph count is harmless noise.** Two PDF pages = two "Page 1
of 2" footers. They are their own paragraphs in the loader output and fall
through every rule to `body_text` — no role units opened, no chunk impact.
A future loader could filter known-footer patterns; for now they're
visible and harmless.

---

## Phase 2 — Validation: keeping hallucination separate from retrieval gaps

`validate.py` runs a fixed 7-question set against each of the three CVs
and reports two distinct numbers, deliberately separated:

- **Hallucination test (the salary question).** "What is his current
  salary?" — the CV does not contain this. Refusal here = ✓ correct
  grounding.
- **Retrieval-gap rate (the other 6 questions).** Refusal here = the
  top-k retrieved chunks didn't contain the answer. NOT a hallucination,
  but a retrieval-quality signal.

The result on Phase 2's three CVs: **all three correctly refused the
hallucination question**, and retrieval-gap rates of 2/6, 3/6, and 4/6
across the answerable questions.

**The framing matters more than the numbers.** Earlier drafts of
`comparison.md` lumped both refusal types into "overall refusal rate" with
the same `(hallucination test passed)` label. That conflates a *desired*
behaviour (refusing to fabricate) with an *undesired* one (failing to
retrieve). To anyone reading the report, "5/7 refused" looks like a
broken system; the truth is "1/7 correctly refused, 4/6 retrieval gaps".
A bullet point in a portfolio piece is not the place to discover this
ambiguity.

**Why retrieval gaps are the more interesting number for the next
iteration.** They show where the embedding space is letting the system
down — a "core skills" question that fails to retrieve the Core Skills
chunk is a vector-search miss, not a generation failure. The CV's "Core
Skills" chunk is short (~25 words); under cosine similarity to a longer
question vector, longer chunks tend to win. That is a known property of
cosine + uneven chunk sizes and a candidate for hybrid retrieval (BM25
keyword + semantic) or larger top-k with re-ranking — both production
upgrades the README's table already names.

**Hallucination behaviour was robust.** No CV produced a fabricated
salary, including on cv3 where the document is heavily tabular and the
chunker captured less linear context. That is the strongest result of the
Phase 1 + Phase 2 work taken together: the fixed-fallback-phrase
instruction (Phase 1) plus the per-CV config that delivers correct chunks
(Phase 2) gave a grounded-only system that knows when to refuse, across
three structurally-different inputs.

---

## Status

**Phase 1 (Week 1) complete.** Six implementation steps, 112-cell stress
test, README.

**Phase 2 complete.** Config-driven chunker, common `Paragraph` model
+ `loaders/`, cross-document validation on three CVs, PDF loader. The
post-Week-1 priority is closed: the pipeline works on three
structurally-different CVs with no chunker code changes between them, and
the loader stack accepts a new file format with one new file and one
dispatch entry.

**What's open next.** Multi-document Q&A. Today the pipeline indexes one
document at a time into per-CV collections; an RFI/RFP context wants a
multi-document searchable corpus with metadata-filtered retrieval. That
is also where the Phase 1 spec's "pluggable strategy registry" sketch
becomes concrete — different document classes (CV vs RFI vs report) need
different chunking strategies, not just different `fingerprint_rules` in
the same strategy.
