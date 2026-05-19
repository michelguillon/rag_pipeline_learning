"""
ingest.py — embed chunks and store them in ChromaDB  [Phase 4 / spec Step 3]
============================================================================
Reads config.json, chunks the document via chunker.py, embeds every chunk with
mistral-embed, and stores the chunks + vectors + metadata into the 4 ChromaDB
collections:

    strategy A   -> cv_role_cosine,   cv_role_l2
    strategy A2  -> cv_bullet_cosine, cv_bullet_l2

    CLI:  python ingest.py data/cv.docx

ARCHITECTURAL DECISION: embed each strategy's chunks ONCE.
cv_role_cosine and cv_role_l2 hold the SAME vectors — they differ only in the
distance metric applied at query time. So strategy A's texts are embedded once
and the identical vectors are added to both collections (likewise for A2).
Embedding is the costly step; never pay for it twice.
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb

import chunker
from mistral_helpers import get_client, call_with_retry

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "mistral-embed"
BATCH_SIZE = 16      # conservative for free-tier rate limits (spec Decision 3)
BATCH_PAUSE = 0.5    # seconds between batches — be polite to the API


def embed_texts(mistral, texts):
    """Embed texts with mistral-embed, in batches, each call retry-wrapped.

    Returns (vectors, total_tokens). Every API call goes through
    call_with_retry so a transient 429/5xx mid-loop does not lose the run.
    """
    vectors, tokens = [], 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = call_with_retry(
            mistral.embeddings.create, model=EMBED_MODEL, inputs=batch)
        vectors.extend(item.embedding for item in response.data)
        if response.usage:
            tokens += response.usage.total_tokens
        if start + BATCH_SIZE < len(texts):
            time.sleep(BATCH_PAUSE)
    return vectors, tokens


def metric_for(collection_name):
    """Distance metric for a collection, from its name suffix.

    The metric is immutable once a collection is created (spec Decision 5);
    the naming convention (_cosine / _l2) is the single source of truth.
    """
    return "l2" if collection_name.endswith("_l2") else "cosine"


def store(chroma, name, chunks, embeddings):
    """(Re)create a collection and add all chunks to it.

    ARCHITECTURAL DECISION: delete-and-recreate, not skip-if-exists.
    Re-running ingest.py should always yield a clean index matching the
    current chunker output. Skipping an existing collection would silently
    keep a stale index after the chunking logic changes.
    """
    existing = {getattr(c, "name", c) for c in chroma.list_collections()}
    if name in existing:
        chroma.delete_collection(name)
    collection = chroma.create_collection(
        name=name, metadata={"hnsw:space": metric_for(name)})
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks and store them in ChromaDB.")
    parser.add_argument("document", help="path to the .docx file")
    args = parser.parse_args()

    if not Path(args.document).exists():
        sys.exit(f"Document not found: {args.document}")
    try:
        config = chunker.load_config()
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    chunks = chunker.all_chunks(args.document, config)
    mistral = get_client()
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    bar = "─" * 60
    print(f"{bar}\nINGESTION\n{bar}")
    total_tokens = 0

    # config["collections"] maps strategy -> [cosine collection, l2 collection]
    for strategy, names in config["collections"].items():
        strat_chunks = chunks.get(strategy, [])
        if not strat_chunks:
            print(f"\nStrategy {strategy}: no chunks — skipped")
            continue
        print(f"\nStrategy {strategy}: embedding {len(strat_chunks)} chunks "
              f"with {EMBED_MODEL}...")
        vectors, tokens = embed_texts(mistral,
                                      [c["text"] for c in strat_chunks])
        total_tokens += tokens
        dim = len(vectors[0]) if vectors else 0
        print(f"  {len(vectors)} vectors ({dim}-dim), {tokens} embedding tokens")
        for name in names:
            collection = store(chroma, name, strat_chunks, vectors)
            print(f"  → {name:18s} {collection.count():>3} chunks  "
                  f"(metric: {metric_for(name)})")

    print(f"\n{bar}")
    print(f"Done. {total_tokens} embedding tokens total. "
          f"Persistent store: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
