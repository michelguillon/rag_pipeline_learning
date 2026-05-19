"""
query.py — retrieval-augmented question answering  [Phase 5 / spec Step 4]
==========================================================================
Embeds a question, retrieves the top-k chunks from a chosen ChromaDB
collection, assembles a grounded prompt, and asks Mistral to answer.

    CLI:
      python query.py "What did he do at Microsoft?" \\
        --collection cv_role_cosine --top-k 3 \\
        --model mistral-small-latest --format labelled --save

ARCHITECTURAL DECISION: the query is embedded with the SAME model
(mistral-embed) that embedded the documents. Retrieval compares the query
vector to the stored vectors; vectors from different models live in different
spaces and are not comparable (LEARNING_NOTES.md, Decision 7).
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import chromadb

from mistral_helpers import get_client, call_with_retry

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "mistral-embed"
OUTPUT_DIR = "outputs"

# The fallback phrase is EXACT and fixed (spec Decision 9): any out-of-scope
# answer that is not this exact string is a hallucination signal — which is
# what makes the Phase 6 stress test measurable.
FALLBACK = "I cannot find this in the provided document."

# Instruction is fixed — NOT an experiment variable (spec Decision 9).
SYSTEM_INSTRUCTION = (
    "You are a precise assistant answering questions about a CV.\n"
    "Answer using ONLY the information in the context provided below.\n"
    "If the answer is not present in the context, respond with exactly:\n"
    f'"{FALLBACK}"\n'
    "Do not use your general knowledge. Do not infer or extrapolate."
)


def source_label(meta):
    """A human-readable source label for a chunk, from its metadata."""
    bits = [meta.get("company") or meta.get("section_name"),
            meta.get("job_title"), meta.get("dates")]
    return " | ".join(b for b in bits if b)


def build_prompt(question, retrieved, context_format):
    """Assemble the user-message content: context block + question.

    Two context formats are tested (spec Decision 9):
      flat     — chunk texts only
      labelled — each chunk tagged with its [Source: ...]; lets the model
                 attribute answers and makes a wrong citation visible.
    """
    blocks = []
    for r in retrieved:
        if context_format == "labelled":
            blocks.append(f"[Source: {source_label(r['metadata'])}]\n{r['text']}")
        else:
            blocks.append(r["text"])
    context = "\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def retrieve(collection, query_vector, top_k):
    """Top-k chunks for a query vector. Returns a list of dicts with scores.

    ChromaDB reports a distance; for a cosine collection we also surface the
    intuitive similarity (1 - distance). top_k is capped at the collection
    size — you cannot retrieve more chunks than exist.
    """
    metric = (collection.metadata or {}).get("hnsw:space", "cosine")
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for doc, meta, dist in zip(result["documents"][0],
                               result["metadatas"][0],
                               result["distances"][0]):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "similarity": (1 - dist) if metric == "cosine" else None,
        })
    return retrieved, metric


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question against an ingested collection.")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--collection", default="cv_role_cosine",
                        help="ChromaDB collection to query")
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of chunks to retrieve")
    parser.add_argument("--model", default="mistral-small-latest",
                        help="generation model (mistral-small/large-latest)")
    parser.add_argument("--format", choices=["flat", "labelled"],
                        default="flat", help="context format in the prompt")
    parser.add_argument("--show-prompt", action="store_true",
                        help="also print the full assembled prompt")
    parser.add_argument("--save", action="store_true",
                        help="save the full result as JSON in outputs/")
    args = parser.parse_args()

    mistral = get_client()
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = chroma.get_collection(args.collection)
    except Exception:
        sys.exit(f"Collection '{args.collection}' not found — run ingest.py first.")

    # 1. embed the question (same model as the documents)
    embed_resp = call_with_retry(
        mistral.embeddings.create, model=EMBED_MODEL, inputs=[args.question])
    query_vector = embed_resp.data[0].embedding

    # 2. retrieve
    retrieved, metric = retrieve(collection, query_vector, args.top_k)

    # 3. assemble the prompt
    user_prompt = build_prompt(args.question, retrieved, args.format)

    # 4. generate. temperature 0.1 — near-deterministic, factual Q&A (Decision 8)
    gen = call_with_retry(
        mistral.chat.complete,
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = gen.choices[0].message.content
    usage = gen.usage

    # 5. report
    bar = "─" * 64
    print(f"\n{bar}")
    print(f"QUERY: {args.question}")
    print(f"COLLECTION: {args.collection} | top_k={args.top_k} | "
          f"model={args.model} | format={args.format}")
    print(f"{bar}\nRETRIEVED CHUNKS:")
    for i, r in enumerate(retrieved, 1):
        score = (f"sim={r['similarity']:.3f}" if r["similarity"] is not None
                 else f"dist={r['distance']:.3f}")
        label = source_label(r["metadata"]) or "(section)"
        print(f"  [{i}] {label}  {score}")
        print(f"      {r['text'][:100]}...")

    if args.show_prompt:
        print(f"\n{bar}\nFULL PROMPT\n{bar}\n{user_prompt}")

    print(f"\n{bar}")
    print(f"TOKENS: {usage.prompt_tokens} prompt + "
          f"{usage.completion_tokens} completion = {usage.total_tokens}")
    print(f"{bar}\nANSWER:\n{answer}")

    # 6. optionally save the full result
    if args.save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(OUTPUT_DIR) / f"{ts}_{args.collection}_k{args.top_k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "query": args.question,
            "collection": args.collection,
            "metric": metric,
            "top_k": args.top_k,
            "model": args.model,
            "format": args.format,
            "retrieved": retrieved,
            "prompt": user_prompt,
            "answer": answer,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }, indent=2), encoding="utf-8")
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
