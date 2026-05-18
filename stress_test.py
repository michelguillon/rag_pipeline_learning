"""
stress_test.py — Hours 8–10: Deliberately Break the RAG Pipeline
=================================================================
This script runs systematic experiments and records findings.
This is where the real learning is — and the interview gold.

Run: python stress_test.py your_document.txt
"""

import json
import time
from pathlib import Path
from typing import Any
import sys

from rag_pipeline import (
    ingest_document,
    query_pipeline,
    chunk_text,
    create_collection,
    chroma_client,
)


# ──────────────────────────────────────────────
# TEST QUESTIONS — adapt these for your document
# ──────────────────────────────────────────────
# Use 3 types of questions:
#   1. Clearly answerable from the document
#   2. Partially answerable (answer spans multiple sections)
#   3. NOT in the document at all (hallucination test)

TEST_QUESTIONS = [
    # Replace these with questions specific to YOUR document.
    # The more you know the document, the more useful the test.
    {
        "question": "What is the main topic of this document?",
        "type": "answerable",
        "expected_in_doc": True,
    },
    {
        "question": "Summarise the key points or conclusions.",
        "type": "multi-section",
        "expected_in_doc": True,
    },
    {
        "question": "What are the specific dates mentioned in the document?",
        "type": "specific",
        "expected_in_doc": True,
    },
    {
        "question": "What is the population of Mars?",  # total red herring
        "type": "hallucination_test",
        "expected_in_doc": False,
    },
]


def run_experiment(
    doc_path: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    questions: list[dict],
    label: str,
) -> dict:
    """
    Run a complete ingest + query cycle with given parameters.
    Returns structured results for comparison.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"  chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}")
    print(f"{'='*60}")

    # Fresh collection for each experiment
    collection_name = f"exp_{chunk_size}_{overlap}_{top_k}".replace("-", "_")

    # Delete collection if exists from a previous run
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    # Ingest
    t0 = time.time()
    collection = ingest_document(
        doc_path,
        collection_name=collection_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    ingest_time = time.time() - t0

    # Query each question
    results = []
    for q in questions:
        t0 = time.time()
        result = query_pipeline(
            collection,
            q["question"],
            top_k=top_k,
            verbose=False,
        )
        query_time = time.time() - t0

        # Extract similarity scores for analysis
        similarities = [c["similarity"] for c in result["retrieved_chunks"]]

        results.append({
            "question": q["question"],
            "type": q["type"],
            "expected_in_doc": q["expected_in_doc"],
            "answer": result["answer"],
            "top_similarity": max(similarities) if similarities else 0,
            "min_similarity": min(similarities) if similarities else 0,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "total_tokens": result["usage"]["total_tokens"],
            "query_time_s": round(query_time, 2),
        })

        # Quick print for visibility
        answered = "✅" if "cannot find" not in result["answer"].lower() else "❌ (refused)"
        hallu_risk = "⚠️ HALLUCINATION RISK" if (
            not q["expected_in_doc"] and "cannot find" not in result["answer"].lower()
        ) else ""
        print(f"\n  Q: {q['question'][:60]}...")
        print(f"  A: {result['answer'][:120]}...")
        print(f"  {answered} | top_sim={max(similarities):.3f} | tokens={result['usage']['total_tokens']} {hallu_risk}")

    return {
        "label": label,
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "num_chunks_stored": collection.count(),
        },
        "ingest_time_s": round(ingest_time, 2),
        "results": results,
    }


def analyse_results(all_experiments: list[dict]) -> None:
    """
    Print a comparative analysis across experiments.
    This is the basis for your README observations.
    """
    print("\n\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)

    for exp in all_experiments:
        cfg = exp["config"]
        results = exp["results"]

        # Average similarity across answerable questions
        answerable = [r for r in results if r["expected_in_doc"]]
        hallucination_tests = [r for r in results if not r["expected_in_doc"]]

        avg_sim = (
            sum(r["top_similarity"] for r in answerable) / len(answerable)
            if answerable else 0
        )
        avg_tokens = (
            sum(r["total_tokens"] for r in results) / len(results)
            if results else 0
        )

        # Count hallucinations (answered something when it should have refused)
        hallucinations = sum(
            1 for r in hallucination_tests
            if "cannot find" not in r["answer"].lower()
        )

        print(f"\n📋 {exp['label']}")
        print(f"   Chunks stored: {cfg['num_chunks_stored']}")
        print(f"   Avg retrieval similarity: {avg_sim:.3f}")
        print(f"   Avg tokens/query: {avg_tokens:.0f}")
        print(f"   Ingest time: {exp['ingest_time_s']}s")
        print(f"   Hallucinations: {hallucinations}/{len(hallucination_tests)} out-of-scope questions answered")

    print("\n\n💡 KEY OBSERVATIONS TO DOCUMENT IN README:")
    print("""
   Chunk size effects:
   - Smaller chunks (256): more precise retrieval but may miss context
   - Larger chunks (1024): richer context but may dilute relevance signal

   top_k effects:
   - top_k=1: fast, cheap, but misses multi-part answers
   - top_k=10: comprehensive but burns tokens and risks noise

   Hallucination behaviour:
   - Watch whether the model refuses correctly when answer isn't in doc
   - A well-prompted RAG system should say "I cannot find this"
   - If it confidently answers from training data: your prompt needs work

   Long document degradation:
   - Deep in a long doc, chunks are harder to retrieve if terminology
     changes or the query doesn't match the exact language used
   """)


def main():
    if len(sys.argv) < 2:
        print("Usage: python stress_test.py <path_to_document.txt>")
        print("Example: python stress_test.py my_cv.txt")
        sys.exit(1)

    doc_path = sys.argv[1]

    all_experiments = []

    # ── EXPERIMENT SET 1: Chunk size comparison ──
    for chunk_size in [256, 512, 1024]:
        exp = run_experiment(
            doc_path=doc_path,
            chunk_size=chunk_size,
            overlap=50,
            top_k=3,
            questions=TEST_QUESTIONS,
            label=f"Chunk size {chunk_size} (overlap=50, top_k=3)",
        )
        all_experiments.append(exp)

    # ── EXPERIMENT SET 2: top_k comparison ──
    for top_k in [1, 5, 10]:
        exp = run_experiment(
            doc_path=doc_path,
            chunk_size=512,
            overlap=50,
            top_k=top_k,
            questions=TEST_QUESTIONS,
            label=f"top_k={top_k} (chunk=512, overlap=50)",
        )
        all_experiments.append(exp)

    # ── EXPERIMENT SET 3: Overlap comparison ──
    for overlap in [0, 50, 100]:
        exp = run_experiment(
            doc_path=doc_path,
            chunk_size=512,
            overlap=overlap,
            top_k=3,
            questions=TEST_QUESTIONS,
            label=f"Overlap={overlap} (chunk=512, top_k=3)",
        )
        all_experiments.append(exp)

    # ── Analysis ──
    analyse_results(all_experiments)

    # Save results to JSON for your README
    output_path = Path("stress_test_results.json")
    output_path.write_text(json.dumps(all_experiments, indent=2))
    print(f"\n📁 Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
