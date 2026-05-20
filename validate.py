"""
validate.py — Phase 2 cross-document validation  [Phase 2 / spec Step 4]
========================================================================
Runs the full pipeline on three structurally-different CVs using their
hand-authored per-CV configs (config_cv1/2/3.json) and emits the portfolio
artifact: chunk counts, retrieval similarities, hallucination refusal, all
side-by-side.

Pipeline per CV (no interaction — configs are pre-approved):
  1. load config -> chunker.all_chunks            (Strategy A and A2)
  2. embed + store into per-CV ChromaDB collections (ingest helpers)
  3. ask a fixed set of test questions against the recommended strategy's
     cosine collection; record top-1 similarity, top-k previews, answer,
     refusal (the exact FALLBACK phrase from query.py)
  4. aggregate one result row per CV

ARCHITECTURAL DECISION: hold the query side CONSTANT, vary only the
document side. The point of this comparison is not "strategy A vs A2" —
that was the Phase 1 question. Phase 2 asks whether the SAME chunker,
on SAME-shape questions, handles three structurally-different documents.
Same model, same top-k, same prompt format, same questions — only the
fingerprint_rules and the document change. That isolates the variable
the README claims to demonstrate.

Outputs (outputs/phase2_validation/):
  comparison.md  — human-readable, recruiter-scannable comparison table
  results.json   — full per-question results for inspection
"""

import argparse
import json
import statistics
from pathlib import Path

import chromadb

import chunker
import ingest as ingest_mod
from mistral_helpers import get_client, call_with_retry
from query import (
    EMBED_MODEL, FALLBACK, SYSTEM_INSTRUCTION,
    build_prompt, retrieve,
)


OUTPUT_DIR = Path("outputs/phase2_validation")
CHROMA_PATH = "./chroma_db"
GENERATION_MODEL = "mistral-small-latest"   # Phase 1 stress-test default
TOP_K = 3
QUERY_FORMAT = "labelled"                   # show [Source: ...] in the context

# The three test documents, in the order they appear in comparison.md.
# `phase1_chunks` records the Phase 1 cross-CV failure — the bar Phase 2
# clears.
# cv1 is the public fake CV (committed). cv2 and cv3 reference the private
# test CVs the cross-CV failure was found on — those .docx files are
# gitignored, so a fresh clone cannot reproduce the cv2/cv3 rows without
# supplying its own documents. The structural FINDINGS (chunk counts,
# decode rule diffs) live in comparison.md, also gitignored — the public
# evidence of Phase 2 lives in the README.
CVS = [
    {"tag": "cv1", "label": "sample_cv.docx",
     "doc": "data/sample_cv.docx",
     "config": "config_cv1.json",
     "structure": "Heading 1 + table, mixed-size headings",
     "phase1_chunks": "11 role / 25 bullet (worked)"},
    {"tag": "cv2", "label": "private CV #2",
     "doc": "data/Gautham_Dilip_Kripalani-CV2024_v4.docx",
     "config": "config_cv2.json",
     "structure": "No heading styles at all — sizes + bold only",
     "phase1_chunks": "3 mega-chunks (failed)"},
    {"tag": "cv3", "label": "private CV #3",
     "doc": "data/2021_GdeChateauvieux_LCV.docx",
     "config": "config_cv3.json",
     "structure": "Table-based, 'Title' paragraph style as company signal",
     "phase1_chunks": "4 mega-chunks (failed)"},
]

# Spec's 5 baseline questions + two extras (synthesis across companies and a
# date-range question that exercises chunk-level date metadata). Same set
# across every CV — the comparison only means something if the questions are
# identical.
QUESTIONS = [
    {"q": "What was his most recent role?",                       "type": "factual"},
    {"q": "What companies has he worked for?",                    "type": "factual"},
    {"q": "What are his core skills?",                            "type": "factual"},
    {"q": "Has he managed teams?",                                "type": "synthesis"},
    {"q": "What is his current salary?",                          "type": "hallucination"},
    {"q": "How long has he been in his most recent role?",        "type": "dates"},
    {"q": "How has his career progressed across companies?",      "type": "synthesis"},
]


def chunk_size_stats(chunks):
    """word-count min/max/mean/median for a list of chunks (or None if empty)."""
    counts = [c["metadata"]["word_count"] for c in chunks]
    if not counts:
        return None
    return {
        "n":      len(counts),
        "min":    min(counts),
        "max":    max(counts),
        "mean":   round(statistics.mean(counts), 1),
        "median": round(statistics.median(counts), 1),
    }


def ingest_cv(mistral, chroma, config, chunks):
    """Embed + store all chunks for one CV. Returns total embedding tokens.

    Mirrors ingest.main without the CLI: one embedding pass per strategy,
    same vectors copied into both the cosine and the l2 collection. Same
    delete-and-recreate semantics, so a re-run never accumulates stale data.
    """
    total = 0
    for strategy, names in config["collections"].items():
        strat_chunks = chunks.get(strategy, [])
        if not strat_chunks:
            continue
        vectors, tokens = ingest_mod.embed_texts(
            mistral, [c["text"] for c in strat_chunks])
        total += tokens
        for name in names:
            ingest_mod.store(chroma, name, strat_chunks, vectors)
    return total


def ask(mistral, collection, question):
    """Embed query → retrieve top-k → generate answer. Records everything.

    Refusal detection is EXACT-match against query.py's FALLBACK string,
    not a fuzzy "looks like a refusal" heuristic. That is intentional: the
    system prompt instructs the model to use EXACTLY that phrase when the
    answer is not in context. Anything else is by definition either a
    hallucination or a partial answer — both useful signals.
    """
    embed_resp = call_with_retry(
        mistral.embeddings.create, model=EMBED_MODEL, inputs=[question])
    q_vec = embed_resp.data[0].embedding

    retrieved, _ = retrieve(collection, q_vec, TOP_K)
    user_prompt = build_prompt(question, retrieved, QUERY_FORMAT)

    gen = call_with_retry(
        mistral.chat.complete,
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = gen.choices[0].message.content.strip()

    return {
        "question":       question,
        "answer":         answer,
        "refused":        answer == FALLBACK,
        "top_similarity": retrieved[0]["similarity"] if retrieved else None,
        "retrieved": [{
            "metadata":     r["metadata"],
            "similarity":   r["similarity"],
            "distance":     r["distance"],
            "text_preview": (r["text"][:160] + ("…" if len(r["text"]) > 160 else "")),
        } for r in retrieved],
        "usage": {
            "prompt_tokens":     gen.usage.prompt_tokens,
            "completion_tokens": gen.usage.completion_tokens,
            "total_tokens":      gen.usage.total_tokens,
        },
    }


def run_cv(mistral, chroma, cv, skip_ingest=False):
    """End-to-end for one CV. Returns the result row that feeds the report."""
    print(f"\n[{cv['tag']}]  doc={cv['doc']}  config={cv['config']}")
    config = chunker.load_config(cv["config"])
    chunks = chunker.all_chunks(cv["doc"], config)
    print(f"  chunks: A={len(chunks['A'])}  A2={len(chunks['A2'])}")

    embed_tokens = 0
    if not skip_ingest:
        print("  embedding + storing...")
        embed_tokens = ingest_cv(mistral, chroma, config, chunks)
        print(f"  embedded ({embed_tokens} tokens)")
    else:
        print("  (--skip-ingest: querying existing collections)")

    strategy = config["strategy"]
    collection_name = config["collections"][strategy][0]   # cosine variant
    collection = chroma.get_collection(collection_name)

    print(f"  querying {collection_name} (top-{TOP_K}, {GENERATION_MODEL})")
    results = []
    for q in QUESTIONS:
        r = ask(mistral, collection, q["q"])
        r["type"] = q["type"]
        results.append(r)
        sim = "n/a" if r["top_similarity"] is None else f"{r['top_similarity']:.3f}"
        flag = "REFUSED" if r["refused"] else "answered"
        print(f"    [{q['type']:13s}] sim={sim}  {flag}")

    answerable_sims = [r["top_similarity"] for r in results
                       if r["type"] != "hallucination"
                       and r["top_similarity"] is not None]
    hallucination = next(r for r in results if r["type"] == "hallucination")

    return {
        "tag":               cv["tag"],
        "label":             cv["label"],
        "doc":               cv["doc"],
        "config":            cv["config"],
        "structure":         cv["structure"],
        "phase1_chunks":     cv["phase1_chunks"],
        "strategy":          strategy,
        "query_collection":  collection_name,
        "phase2_chunks":     {"A": len(chunks["A"]), "A2": len(chunks["A2"])},
        "chunk_size_stats":  {"A":  chunk_size_stats(chunks["A"]),
                              "A2": chunk_size_stats(chunks["A2"])},
        "fingerprint_rules": config["fingerprint_rules"],
        "consistency_flags": config.get("consistency_flags", []),
        "embed_tokens":      embed_tokens,
        "results":           results,
        "summary": {
            "refused_on_hallucination": hallucination["refused"],
            "refusal_rate_overall":     round(
                sum(r["refused"] for r in results) / len(results), 3),
            "mean_top_sim_answerable":  round(
                statistics.mean(answerable_sims), 3) if answerable_sims else None,
        },
    }


def render_markdown(rows):
    """Render the comparison.md text from the result rows.

    The columns are deliberately recruiter-scannable in 10 seconds (chunk
    counts before vs after, refusal on the hallucination question) and the
    decode-rule diff section is what an architect-level reviewer probes."""
    n = len(QUESTIONS)
    L = []

    L.append("# Phase 2 — Cross-Document Validation\n")
    L.append("Same code, same chunker, three structurally-different CVs. The "
             "only things that change between runs are the **per-CV "
             "`fingerprint_rules`** (the ordered decode rules in "
             "`config_cv{N}.json`) and the per-CV ChromaDB collection names. "
             "This is the artifact that backs the Phase 2 claim — that the "
             "Phase 1 cross-CV failure (mega-chunks on cv2 and cv3) was a "
             "decode-rule problem, not a parser problem.\n")
    L.append(f"_Generation model: `{GENERATION_MODEL}`. "
             f"Embed: `{EMBED_MODEL}`. top-k={TOP_K}, format=`{QUERY_FORMAT}`._\n")

    L.append("## Chunk counts\n")
    L.append("| CV | Structure | Phase 1 | Phase 2 (A / A2) |")
    L.append("| --- | --- | --- | --- |")
    for r in rows:
        p2 = f"{r['phase2_chunks']['A']} / {r['phase2_chunks']['A2']}"
        L.append(f"| **{r['tag']}** — {r['label']} | {r['structure']} | "
                 f"{r['phase1_chunks']} | **{p2}** |")

    L.append(f"\n## Retrieval quality  (A2 + cosine, top-{TOP_K})\n")
    L.append("Two distinct numbers, kept separate on purpose: "
             "**hallucination behaviour** (does the system refuse when the "
             "answer truly is not in the document?) and **retrieval gaps** "
             "(answerable questions where the right chunk didn't make it into "
             "the top-k). A correct system has the first column ✓ and a low "
             "second column.\n")
    L.append("| CV | Mean top-1 sim (answerable) | Hallucination test "
             "(salary) | Retrieval gaps (answerable Qs refused) |")
    L.append("| --- | --- | --- | --- |")
    n_answerable = n - 1   # exactly one hallucination question
    for r in rows:
        s = r["summary"]
        mts = "n/a" if s["mean_top_sim_answerable"] is None \
            else f"{s['mean_top_sim_answerable']:.3f}"
        # Non-hallucination refusals = retrieval gaps. The system refused
        # because no chunk in the top-k actually answered the question.
        gap = sum(1 for res in r["results"]
                  if res["refused"] and res["type"] != "hallucination")
        hall_cell = ("✓ correctly refused" if s["refused_on_hallucination"]
                     else "✗ HALLUCINATED")
        L.append(f"| **{r['tag']}** | {mts} | {hall_cell} | "
                 f"{gap}/{n_answerable} |")

    L.append("\n## Chunk size (words)\n")
    L.append("| CV | A: n / min / median / max | A2: n / min / median / max |")
    L.append("| --- | --- | --- |")
    for r in rows:
        def fmt(s):
            if not s:
                return "n/a"
            return f"{s['n']} / {s['min']} / {s['median']} / {s['max']}"
        L.append(f"| **{r['tag']}** | {fmt(r['chunk_size_stats']['A'])} | "
                 f"{fmt(r['chunk_size_stats']['A2'])} |")

    L.append("\n## Fingerprint rules (the only thing that changes)\n")
    for r in rows:
        L.append(f"### {r['tag']} — `{r['config']}`\n")
        L.append("```")
        for rule in r["fingerprint_rules"]:
            L.append(f"  {rule['signal']:26s} -> {rule['role']}")
        L.append("```\n")

    L.append("## Per-question answers\n")
    for r in rows:
        L.append(f"### {r['tag']}\n")
        for res in r["results"]:
            sim = "n/a" if res["top_similarity"] is None \
                else f"{res['top_similarity']:.3f}"
            # Different label for the two flavours of refusal: the
            # hallucination test PASSING means the system correctly declined
            # to invent a salary; a refusal on a real factual question means
            # the chunk that would have answered it didn't surface in top-k.
            if res["refused"]:
                if res["type"] == "hallucination":
                    tail = "  _(refused — hallucination test PASSED)_"
                else:
                    tail = "  _(refused — retrieval gap, no relevant chunk in top-k)_"
            else:
                tail = ""
            L.append(f"**Q ({res['type']}):** {res['question']}  ")
            L.append(f"**A:** {res['answer']}{tail}  ")
            L.append(f"_top-1 sim: {sim}_")
            L.append("")

    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 cross-document validation.")
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="skip embedding/storing; query existing collections only")
    parser.add_argument(
        "--report-only", action="store_true",
        help="do not call any API; re-render comparison.md from the existing "
             "results.json (useful for tweaking the report without spending)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --report-only: skip everything except the markdown rendering. Lets a
    # report iteration not cost any API. Falls back to the full run if no
    # results.json is present yet.
    results_path = OUTPUT_DIR / "results.json"
    if args.report_only and results_path.exists():
        full = json.loads(results_path.read_text(encoding="utf-8"))
        rows = full["cvs"]
        (OUTPUT_DIR / "comparison.md").write_text(
            render_markdown(rows), encoding="utf-8")
        print(f"Re-rendered → {OUTPUT_DIR / 'comparison.md'}")
        return

    mistral = get_client()
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    rows = [run_cv(mistral, chroma, cv, skip_ingest=args.skip_ingest)
            for cv in CVS]

    full = {
        "model_generation": GENERATION_MODEL,
        "model_embed":      EMBED_MODEL,
        "top_k":            TOP_K,
        "query_format":     QUERY_FORMAT,
        "questions":        QUESTIONS,
        "cvs":              rows,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(full, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "comparison.md").write_text(
        render_markdown(rows), encoding="utf-8")

    print(f"\nDone.")
    print(f"  → {OUTPUT_DIR / 'comparison.md'}")
    print(f"  → {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
