"""
stress_test.py — run the full experiment matrix  [Phase 6 / spec Step 5]
========================================================================
Runs every experiment combination over a fixed question set and records the
results, so the chunk-strategy / metric / top_k / model / format tradeoffs
can be compared empirically.

    CLI:
      python stress_test.py                       # run (resumes if interrupted)
      python stress_test.py --dry-run             # print the matrix, no API calls
      python stress_test.py --fresh               # ignore prior progress, restart
      python stress_test.py --models mistral-small-latest   # subset
      python stress_test.py --collections cv_role_cosine    # subset

ARCHITECTURAL DECISION: checkpoint after every cell, and resume.
This is a long batch job against a rate-limited free tier. Progress is written
to a working file after each cell; on re-run, completed cells are skipped. One
229-on-call-91 failure must not discard 90 successes. See LEARNING_NOTES Phase 6.

Output: outputs/stress_test_{timestamp}.json (on completion) + a comparison table.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import chromadb

import query
from mistral_helpers import get_client, call_with_retry

COLLECTIONS = ["cv_role_cosine", "cv_role_l2",
               "cv_bullet_cosine", "cv_bullet_l2"]
MODELS = ["mistral-small-latest", "mistral-large-latest"]
FORMATS = ["flat", "labelled"]

# Fixed question set — covers every failure mode (spec Step 5).
QUESTIONS = [
    {"q": "What was his role at Utiq?", "in_doc": True},
    {"q": "What languages does he speak?", "in_doc": True},
    {"q": "What did he achieve at Microsoft?", "in_doc": True},
    {"q": "Summarise his experience in solutions consulting.", "in_doc": True},
    {"q": "What revenue growth did he deliver at Appnexus?", "in_doc": True},
    {"q": "What is his current salary?", "in_doc": False},
    {"q": "Has he worked in healthcare?", "in_doc": False},
]

CALL_PAUSE = 1.5    # seconds between calls — gentle on the free-tier rate limit
RETRIES = 6         # per-call retries; base_delay 2s -> up to ~64s backoff
PROGRESS = Path("outputs/stress_test_progress.json")


def top_k_values(collection_name):
    """top_k calibrated to collection size (spec Decision 6): role ~11 chunks,
    bullet ~25. Min exposes single-best-match; max stays near the 40% ceiling."""
    return [1, 3] if collection_name.startswith("cv_role") else [1, 5]


def run_cell(mistral, collection, q_vector, question, top_k, model, fmt):
    """Run one matrix cell: retrieve + generate for a single question."""
    retrieved, _ = query.retrieve(collection, q_vector, top_k)
    prompt = query.build_prompt(question, retrieved, fmt)
    gen = call_with_retry(
        mistral.chat.complete,
        model=model,
        messages=[
            {"role": "system", "content": query.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_retries=RETRIES, base_delay=2.0,
    )
    answer = gen.choices[0].message.content
    sims = [r["similarity"] for r in retrieved if r["similarity"] is not None]
    return {
        "answer": answer,
        "refused": answer.strip() == query.FALLBACK,
        "top_sim": retrieved[0]["similarity"] if retrieved else None,
        "avg_sim": sum(sims) / len(sims) if sims else None,
        "prompt_tokens": gen.usage.prompt_tokens,
        "completion_tokens": gen.usage.completion_tokens,
    }


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"q_vectors": {}, "results": []}


def save_progress(state):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def print_table(results, cells):
    """Comparison table — one row per matrix cell that has results."""
    print(f"\n{'─' * 78}")
    print(f"{'collection':18s} {'k':>2} {'model':14s} {'format':9s} "
          f"{'avg_sim':>8} {'avg_tok':>8} {'refused':>8}")
    print("─" * 78)
    for collection, top_k, model, fmt in cells:
        cell = [r for r in results
                if r["collection"] == collection and r["top_k"] == top_k
                and r["model"] == model and r["format"] == fmt]
        if not cell:
            continue
        sims = [r["top_sim"] for r in cell if r["top_sim"] is not None]
        toks = [r["prompt_tokens"] + r["completion_tokens"] for r in cell]
        halluc = [r for r in cell if not r["in_doc"]]
        refused_ok = sum(1 for r in halluc if r["refused"])
        avg_sim = f"{sum(sims) / len(sims):.3f}" if sims else "-"
        print(f"{collection:18s} {top_k:>2} {model.split('-')[1]:14s} "
              f"{fmt:9s} {avg_sim:>8} {sum(toks) // len(toks):>8} "
              f"{refused_ok}/{len(halluc):>7}")
    print("─" * 78)


def main():
    parser = argparse.ArgumentParser(description="Run the RAG experiment matrix.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the matrix and query count, make no calls")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved progress and restart from scratch")
    parser.add_argument("--collections", default=",".join(COLLECTIONS),
                        help="comma-separated subset of collections to run")
    parser.add_argument("--models", default=",".join(MODELS),
                        help="comma-separated subset of models to run")
    args = parser.parse_args()

    collections = [c for c in args.collections.split(",") if c]
    models = [m for m in args.models.split(",") if m]
    cells = [(c, k, m, f)
             for c in collections
             for k in top_k_values(c)
             for m in models
             for f in FORMATS]
    total = len(cells) * len(QUESTIONS)
    print(f"Matrix: {len(collections)} collections x top_k x {len(models)} "
          f"models x {len(FORMATS)} formats x {len(QUESTIONS)} questions "
          f"= {total} calls")
    if args.dry_run:
        print("(dry run — no API calls made)")
        return

    if args.fresh and PROGRESS.exists():
        PROGRESS.unlink()
    state = load_progress()
    done = {(r["collection"], r["top_k"], r["model"], r["format"], r["question"])
            for r in state["results"]}
    if done:
        print(f"Resuming — {len(done)}/{total} cells already complete.")

    mistral = get_client()
    chroma = chromadb.PersistentClient(path=query.CHROMA_PATH)

    # Embed each question ONCE — the vector is independent of collection /
    # model / format, and is cached in the progress file across resumes.
    for item in QUESTIONS:
        if item["q"] not in state["q_vectors"]:
            resp = call_with_retry(mistral.embeddings.create,
                                   model=query.EMBED_MODEL, inputs=[item["q"]],
                                   max_retries=RETRIES, base_delay=2.0)
            state["q_vectors"][item["q"]] = resp.data[0].embedding
            save_progress(state)

    coll_objs = {c: chroma.get_collection(c) for c in collections}

    stopped = False
    for collection, top_k, model, fmt in cells:
        if stopped:
            break
        for item in QUESTIONS:
            key = (collection, top_k, model, fmt, item["q"])
            if key in done:
                continue
            try:
                outcome = run_cell(mistral, coll_objs[collection],
                                   state["q_vectors"][item["q"]], item["q"],
                                   top_k, model, fmt)
            except Exception as exc:  # retries exhausted — checkpoint and stop
                print(f"\nStopped at {len(state['results'])}/{total}: {exc}")
                print("Progress saved — re-run `python stress_test.py` to resume.")
                stopped = True
                break
            state["results"].append({
                "collection": collection, "top_k": top_k, "model": model,
                "format": fmt, "question": item["q"], "in_doc": item["in_doc"],
                **outcome,
            })
            done.add(key)
            save_progress(state)
            time.sleep(CALL_PAUSE)
        if not stopped:
            print(f"  [{len(done):>3}/{total}] {collection} k={top_k} "
                  f"{model.split('-')[1]} {fmt}")

    results = state["results"]
    print_table(results, cells)
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"]
                       for r in results)
    print(f"{len(results)}/{total} cells | {total_tokens} tokens")

    if not stopped and len(results) == total:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final = Path("outputs") / f"stress_test_{ts}.json"
        final.write_text(json.dumps({"questions": QUESTIONS, "results": results},
                                    indent=2), encoding="utf-8")
        PROGRESS.unlink(missing_ok=True)
        print(f"Complete — saved → {final}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
