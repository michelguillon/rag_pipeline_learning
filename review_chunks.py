"""
review_chunks.py — preview proposed chunks before any embedding  [Phase 3 / spec Step 2]
========================================================================================
Reads config.json, builds the chunks via chunker.py, and prints every one for
human review. No embedding, no storage, no API calls, no spend.

    CLI:
      python review_chunks.py data/cv.docx                  # preview A and A2
      python review_chunks.py data/cv.docx --strategy A      # preview one strategy

WHY THIS STEP EXISTS: embedding costs tokens and writes to ChromaDB. Inspecting
the chunks first means bad boundaries are caught before any spend — the
equivalent of a data-validation gate before a batch job. Chunk *size* is an
observed outcome reviewed here, never an input (see docs/LEARNING_NOTES.md, Phase 3).

review_chunks.py and ingest.py both build chunks through chunker.py, so what is
previewed here is exactly what gets ingested.
"""

import argparse
import sys

import chunker

BAR = "─" * 68


def print_chunk(chunk):
    """Print one chunk: metadata header, then full text."""
    m = chunk["metadata"]
    kind = "bullet" if m["is_bullet"] else "role/section"
    print(BAR)
    print(f"CHUNK {m['chunk_index']:03d} | id={chunk['id']} | "
          f"strategy={m['strategy']} | {kind}")
    if m["section_name"]:
        print(f"  section: {m['section_name']}")
    if m["company"]:
        print(f"  company: {m['company']}")
    if m["job_title"]:
        print(f"  title:   {m['job_title']}")
    if m["dates"]:
        print(f"  dates:   {m['dates']}")
    print(f"  words:   {m['word_count']}")
    print(BAR)
    print(chunk["text"])
    print()


def print_summary(chunks, label):
    """Print the total / avg / min / max word count for a strategy."""
    counts = [c["metadata"]["word_count"] for c in chunks]
    if not counts:
        print(f"{label}: 0 chunks")
        return
    print(f"{label}: {len(chunks)} chunks | "
          f"avg {sum(counts) // len(counts)} words | "
          f"min {min(counts)} | max {max(counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview proposed chunks before ingestion.")
    parser.add_argument("document", help="path to the .docx file")
    parser.add_argument("--strategy", choices=["A", "A2", "both"],
                        default="both", help="which strategy to preview")
    args = parser.parse_args()

    try:
        chunks = chunker.all_chunks(args.document)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    strategies = ["A", "A2"] if args.strategy == "both" else [args.strategy]

    summaries = []
    for strat in strategies:
        print(f"\n{'═' * 68}\nSTRATEGY {strat}\n{'═' * 68}")
        for chunk in chunks[strat]:
            print_chunk(chunk)
        label = f"Strategy {strat}"
        print_summary(chunks[strat], label)
        summaries.append((label, chunks[strat]))

    print(f"\n{'═' * 68}\nSUMMARY\n{'═' * 68}")
    for label, strat_chunks in summaries:
        print_summary(strat_chunks, label)

    # review_chunks.py changes nothing — the prompt is an explicit go/no-go
    # gate. ingest.py is a separate command the human runs next.
    try:
        answer = input("\nDo the chunks look right? Proceed to ingestion? (y/n): ")
        if answer.strip().lower() == "y":
            print("→ Next: python ingest.py data/cv.docx")
        else:
            print("Not confirmed — fix config.json (or the document) and re-run.")
    except EOFError:
        print("\n(no interactive input — review the chunks above, then run "
              "ingest.py when ready)")


if __name__ == "__main__":
    main()
