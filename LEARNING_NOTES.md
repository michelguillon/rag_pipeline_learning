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

## Next step

Project build complete. Remaining: the README — written last, against these
real findings (spec README structure).
