"""
mistral_basics.py — Hours 1–2: Mistral API Fundamentals
=========================================================
Run this before building the RAG pipeline.
Each section is a standalone experiment — run them in order,
read the output, and understand what you're seeing.

Run: python mistral_basics.py
"""

import argparse
import os
import time

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import MistralError

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

DIVIDER = "\n" + "─" * 60 + "\n"


# ──────────────────────────────────────────────
# REUSABLE HELPER: retry with exponential backoff
# ──────────────────────────────────────────────
# This is the one piece of mistral_basics.py meant to be lifted
# straight into rag_pipeline.py — wrap every API call with it.

# Which HTTP statuses are worth retrying. 429 = rate limit (free tier!);
# 5xx = transient server errors. Everything else (400 bad request,
# 401 bad key, 404) is a bug on our side — retrying would not help.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def call_with_retry(func, *args, max_retries=5, base_delay=1.0, **kwargs):
    """Call a Mistral SDK method, retrying transient failures with backoff.

    Returns whatever `func` returns. Re-raises the error if it is not
    retryable, or if `max_retries` is exhausted.

    Backoff doubles each attempt (base_delay, 2x, 4x, ...). If the server
    sends a `Retry-After` header, we honour that instead — it knows best.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except MistralError as exc:
            attempt += 1
            # Not retryable, or we have tried enough — give up and raise.
            if exc.status_code not in RETRYABLE_STATUS or attempt > max_retries:
                raise

            retry_after = exc.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)            # server told us how long
            else:
                delay = base_delay * (2 ** (attempt - 1))  # exponential backoff

            print(f"  ⚠️  HTTP {exc.status_code} on attempt "
                  f"{attempt}/{max_retries} — retrying in {delay:.1f}s")
            time.sleep(delay)


# ──────────────────────────────────────────────
# EXPERIMENT 1: Basic completion
# ──────────────────────────────────────────────

def experiment_1_basic():
    print(DIVIDER + "EXPERIMENT 1: Basic completion")

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": "What is a vector database? Answer in 2 sentences."}
        ]
    )

    print(f"Answer: {response.choices[0].message.content}")
    print(f"\nToken usage:")
    print(f"  Prompt tokens:     {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total:             {response.usage.total_tokens}")

    # KEY INSIGHT: prompt_tokens is the cost of your INPUT.
    # In RAG, your prompt = system prompt + context chunks + question.
    # As chunks grow, prompt_tokens grows — watch this closely.


# ──────────────────────────────────────────────
# EXPERIMENT 2: System prompt influence
# ──────────────────────────────────────────────

def experiment_2_system_prompt():
    print(DIVIDER + "EXPERIMENT 2: System prompt influence")

    question = "What is retrieval-augmented generation?"

    for system_prompt in [
        "You are a helpful assistant.",
        "You are a terse assistant. Answer in one sentence only.",
        "You are an expert explaining to a non-technical CEO. Avoid jargon.",
        "You are sceptical. Always mention limitations and caveats.",
    ]:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )
        print(f"\nSystem: '{system_prompt[:50]}...'")
        print(f"Answer: {response.choices[0].message.content}...")

    # KEY INSIGHT: The system prompt is your primary lever for controlling
    # model behaviour. In RAG, your system prompt should instruct the model
    # to ONLY use provided context — not its training data.


# ──────────────────────────────────────────────
# EXPERIMENT 3: Temperature effects
# ──────────────────────────────────────────────

def experiment_3_temperature():
    print(DIVIDER + "EXPERIMENT 3: Temperature effects")

    prompt = "List 3 potential risks of deploying a RAG system in production."

    for temperature in [0.0, 0.5, 1.0]:
        print(f"\nTemperature: {temperature}")
        # Run twice at each temperature to see variance
        for run in range(2):
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            print(f"  Run {run+1}: {response.choices[0].message.content[:150]}...")

    # KEY INSIGHT:
    # temperature=0.0 → near-deterministic, same answer each run
    # temperature=1.0 → high variance, creative but inconsistent
    # For RAG Q&A: use 0.0–0.2. For creative tasks: 0.7–1.0.


# ──────────────────────────────────────────────
# EXPERIMENT 4: Context window limits in practice
# ──────────────────────────────────────────────

def experiment_4_context_window():
    print(DIVIDER + "EXPERIMENT 4: Context window and token costs")

    # Simulate what happens as you add more RAG context
    base_question = "Summarise the key points."

    # Simulate increasingly large context (like adding more RAG chunks)
    for num_fake_chunks in [1, 3, 5, 10]:
        fake_chunk = "This is a sample document chunk containing important information about the topic. " * 30
        context = "\n\n".join([f"Chunk {i}: {fake_chunk}" for i in range(num_fake_chunks)])

        prompt = f"Context:\n{context}\n\nQuestion: {base_question}"

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )

        print(f"\n{num_fake_chunks} chunks → {response.usage.prompt_tokens} prompt tokens "
              f"(${response.usage.prompt_tokens * 0.000002:.4f} approx cost)")

    # KEY INSIGHT:
    # Every chunk you add costs tokens. In production:
    # - mistral-small: ~$0.2/1M tokens
    # - mistral-large: ~$2/1M tokens
    # With 1000 queries/day and 3 chunks of 512 tokens each:
    # ~1536 tokens/query × 1000 = 1.5M tokens/day on prompts alone.


# ──────────────────────────────────────────────
# EXPERIMENT 5: Streaming
# ──────────────────────────────────────────────

def experiment_5_streaming():
    print(DIVIDER + "EXPERIMENT 5: Streaming responses")
    print("(Watch the tokens arrive in real-time)\n")

    stream = client.chat.stream(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": "Explain the difference between semantic search and keyword search in 4 sentences."}
        ]
    )

    # Stream prints each token as it arrives
    for chunk in stream:
        delta = chunk.data.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)

    print("\n")
    # KEY INSIGHT: Streaming is critical for UX in production — users see
    # the response building rather than waiting for the full completion.
    # In RAG pipelines, you often stream the final generation step only.


# ──────────────────────────────────────────────
# EXPERIMENT 6: Embeddings
# ──────────────────────────────────────────────

def experiment_6_embeddings():
    print(DIVIDER + "EXPERIMENT 6: Embeddings and semantic similarity")

    texts = [
        "The cat sat on the mat.",
        "A feline rested on a rug.",         # semantically similar to [0]
        "Quantum computing uses qubits.",    # semantically different
    ]

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts,
    )

    embeddings = [item.embedding for item in response.data]
    dim = len(embeddings[0])
    print(f"Embedding dimensions: {dim}")

    # ── What does an embedding actually look like? ──
    # Each embedding is a plain Python list of floats — a vector.
    # The model turned a sentence into `dim` numbers.
    first = embeddings[0]
    print(f"\nThe embedding for '{texts[0]}' is a {type(first).__name__} "
          f"of {len(first)} floats.")
    print(f"  First 8 values : {[round(x, 5) for x in first[:8]]}")
    print(f"  Last 3 values  : {[round(x, 5) for x in first[-3:]]}")
    print(f"  Min / Max      : {min(first):.5f} / {max(first):.5f}")
    # mistral-embed returns L2-normalised vectors (length 1.0), which is why
    # cosine similarity below is just the dot product.
    magnitude = sum(x**2 for x in first) ** 0.5
    print(f"  Vector length  : {magnitude:.5f}  (≈1.0 → already normalised)")

    # KEY INSIGHT: A RAG vector store (ChromaDB) holds one of these lists
    # per chunk. "Searching" means finding the stored vectors closest to
    # your question's vector — no keywords involved.

    # Cosine similarity between two vectors
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = sum(x**2 for x in a) ** 0.5
        mag_b = sum(x**2 for x in b) ** 0.5
        return dot / (mag_a * mag_b)

    sim_01 = cosine_sim(embeddings[0], embeddings[1])
    sim_02 = cosine_sim(embeddings[0], embeddings[2])
    sim_12 = cosine_sim(embeddings[1], embeddings[2])

    print(f"\nSimilarity scores (cosine):")
    print(f"  'cat sat' vs 'feline rested'  : {sim_01:.4f}  ← should be HIGH")
    print(f"  'cat sat' vs 'quantum computing': {sim_02:.4f}  ← should be LOW")
    print(f"  'feline rested' vs 'quantum'    : {sim_12:.4f}  ← should be LOW")

    # KEY INSIGHT: This is how RAG retrieval works.
    # ChromaDB does this cosine similarity calculation across ALL stored
    # chunks in milliseconds, returning the top-k most similar to your query.


# ──────────────────────────────────────────────
# EXPERIMENT 7: Error handling and retry-with-backoff
# ──────────────────────────────────────────────

def experiment_7_error_handling():
    print(DIVIDER + "EXPERIMENT 7: Error handling and retry-with-backoff")

    # ── Part A: simulate a flaky API — costs ZERO tokens ──
    # We hand call_with_retry() a fake function that fails with HTTP 429
    # (rate limit) twice, then succeeds. No real request leaves your machine,
    # so this part does not touch your token quota at all.
    print("\nPart A: simulated rate-limit — fails twice, then succeeds")
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            # Build a fake HTTP 429 response and raise the real SDK error type.
            fake_response = httpx.Response(
                429, headers={"content-type": "application/json"}
            )
            raise MistralError(
                "Simulated 429 Too Many Requests", fake_response, body="slow down"
            )
        return "the call finally went through"

    result = call_with_retry(flaky_call, base_delay=0.5)
    print(f"  ✅ Succeeded on attempt {attempts['count']}: {result}")

    # ── Part B: a real call routed through the same wrapper ──
    # Tiny prompt + max_tokens=5 → a few tokens only. Shows that the wrapper
    # is transparent: on success it just returns the normal response object.
    print("\nPart B: a real API call through call_with_retry() (tiny prompt)")
    response = call_with_retry(
        client.chat.complete,
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
    )
    print(f"  Model replied: {response.choices[0].message.content!r}")

    # KEY INSIGHT:
    # - 429 (rate limit) and 5xx are TRANSIENT — back off and retry.
    # - 400/401/404 are YOUR bug — retrying just wastes time; let them raise.
    # - Honour a `Retry-After` header when the server sends one.
    # In rag_pipeline.py you will embed many chunks in a loop — wrap every
    # client.embeddings.create / client.chat.complete call in call_with_retry().


# ──────────────────────────────────────────────
# RUN ALL (or a selected subset)
# ──────────────────────────────────────────────

# Registry: experiment number → (function, one-line description).
# Keep this in sync if you add experiments.
EXPERIMENTS = {
    1: (experiment_1_basic,          "Basic completion"),
    2: (experiment_2_system_prompt,  "System prompt influence"),
    3: (experiment_3_temperature,    "Temperature effects"),
    4: (experiment_4_context_window, "Context window and token costs"),
    5: (experiment_5_streaming,      "Streaming responses"),
    6: (experiment_6_embeddings,     "Embeddings and semantic similarity"),
    7: (experiment_7_error_handling, "Error handling and retry-with-backoff"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mistral API learning experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python mistral_basics.py                # run all\n"
               "  python mistral_basics.py --exp 6        # only experiment 6\n"
               "  python mistral_basics.py --exp 1,3,6    # a subset, in that order\n"
               "\nTip: running fewer experiments uses fewer API tokens.",
    )
    # Several spellings accepted: -e 6 / --exp 6 / --exp=6 / -exp=6
    parser.add_argument(
        "-e", "-exp", "--exp",
        dest="exp",
        default="all",
        metavar="LIST",
        help="'all' (default), a single number (6), "
             "or a comma-separated list (1,3,6).",
    )
    return parser.parse_args()


def resolve_experiments(spec):
    """Turn the --exp string into an ordered list of valid experiment numbers."""
    if spec.strip().lower() == "all":
        return sorted(EXPERIMENTS)

    chosen = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) not in EXPERIMENTS:
            valid = ", ".join(map(str, sorted(EXPERIMENTS)))
            raise SystemExit(f"❌ Unknown experiment: {part!r}. Valid values: {valid}")
        num = int(part)
        if num not in chosen:          # de-duplicate, keep first occurrence
            chosen.append(num)
    if not chosen:
        raise SystemExit("❌ No experiments selected. Try --exp 6 or --exp all.")
    return chosen


if __name__ == "__main__":
    args = parse_args()
    to_run = resolve_experiments(args.exp)

    print("🚀 Mistral API Basics — Hours 1-2")
    print(f"Running {len(to_run)} of {len(EXPERIMENTS)} experiment(s): "
          f"{', '.join(map(str, to_run))}\n")

    for i, num in enumerate(to_run):
        func, _ = EXPERIMENTS[num]
        func()
        if i < len(to_run) - 1:        # brief pause between experiments only
            time.sleep(1)

    print(DIVIDER)
    print("✅ Done. You now understand the building blocks.")
    print("Next: open rag_pipeline.py and build the full system.")
