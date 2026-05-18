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

## Next step

`rag_pipeline.py` — build the full retrieval-augmented generation system:
chunk documents → embed chunks → store in ChromaDB → embed query → retrieve
top-k → feed as context to `chat.complete`.
