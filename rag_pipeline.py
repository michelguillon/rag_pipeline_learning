"""
RAG Pipeline — Week 1 of AI Learning Track
==========================================
Architecture: Document Q&A using Mistral + ChromaDB

Flow:
  Ingest: Document → Chunks → Embeddings → ChromaDB
  Query:  Question → Embed → ChromaDB similarity search → Top-k chunks
          → Prompt assembly → Mistral completion → Answer

Every architectural decision is annotated. Read the comments.
"""

import os
import re
import time
from pathlib import Path
from typing import Optional

import chromadb
from mistralai.client import Mistral

# ──────────────────────────────────────────────
# ARCHITECTURAL DECISION 1: Client initialisation
# ──────────────────────────────────────────────
# We initialise both clients once at module level.
# In production you'd inject these via dependency injection or
# load them lazily — but for a learning pipeline, global clients are fine.

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "Set MISTRAL_API_KEY environment variable before running.\n"
        "  export MISTRAL_API_KEY=your_key_here"
    )

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ChromaDB: ephemeral (in-memory) client for experimentation.
# Switch to chromadb.PersistentClient(path="./chroma_db") to persist between runs.
# ARCHITECTURAL DECISION: In-memory means you re-ingest on every run.
# That's fine for learning — it forces you to re-examine the ingestion step.
chroma_client = chromadb.EphemeralClient()


# ──────────────────────────────────────────────
# STEP 1: CHUNKING
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> list[dict]:
    """
    Split text into overlapping chunks.

    ARCHITECTURAL DECISION: Why overlap?
    Without overlap, a key sentence that falls at a chunk boundary gets
    split in half. Neither chunk contains the complete idea, so retrieval
    degrades. Overlap ensures boundary content appears in at least two chunks.

    ARCHITECTURAL DECISION: Token-approximate chunking via word count.
    True tokenisation requires a tokeniser (e.g. tiktoken). As a practical
    approximation: 1 token ≈ 0.75 words, so 512 tokens ≈ 384 words.
    We chunk by words here. To switch to exact token counts, replace
    word splitting with a tokeniser.

    EXPERIMENT HOOK: Change chunk_size (256 / 512 / 1024) and overlap
    (0 / 50 / 100) and observe retrieval quality changes.

    Returns a list of dicts with keys: 'text', 'chunk_index', 'word_count'
    """
    # Normalise whitespace — multiple newlines, tabs, etc.
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    # Approximate token count from word count
    tokens_per_word = 1.33  # rough average
    words_per_chunk = int(chunk_size / tokens_per_word)
    words_overlap = int(overlap / tokens_per_word)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        chunk_text_str = ' '.join(chunk_words)

        chunks.append({
            'text': chunk_text_str,
            'chunk_index': chunk_index,
            'word_count': len(chunk_words),
            # Approximate token count for logging
            'approx_tokens': int(len(chunk_words) * tokens_per_word),
        })

        chunk_index += 1
        # Move forward by (chunk_size - overlap), not full chunk_size
        # This is what creates the overlap
        start += words_per_chunk - words_overlap

        # Safety: if overlap >= chunk_size we'd loop forever
        if words_per_chunk <= words_overlap:
            raise ValueError("overlap must be smaller than chunk_size")

    return chunks


# ──────────────────────────────────────────────
# STEP 2: EMBEDDING
# ──────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Convert text strings to embedding vectors using Mistral's embedding model.

    ARCHITECTURAL DECISION: Why batch?
    The Mistral API has rate limits (requests/min and tokens/min).
    Batching reduces the number of HTTP round-trips and is more efficient.
    batch_size=32 is a safe default; increase for speed, decrease if you
    hit rate limit errors.

    ARCHITECTURAL DECISION: Why mistral-embed specifically?
    Embedding models are trained differently from completion models —
    they're optimised to place semantically similar texts close together
    in vector space. Never use a completion model to generate embeddings.

    Returns a list of vectors (one per input text).
    Each vector has 1024 dimensions for mistral-embed.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Retry logic for rate limits — simple exponential backoff
        for attempt in range(3):
            try:
                response = mistral_client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"  Embedding attempt {attempt + 1} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)

        # Be polite to the API between batches
        if i + batch_size < len(texts):
            time.sleep(0.2)

    return all_embeddings


# ──────────────────────────────────────────────
# STEP 3: STORAGE — ChromaDB
# ──────────────────────────────────────────────

def create_collection(name: str = "rag_documents") -> chromadb.Collection:
    """
    Create (or retrieve) a ChromaDB collection.

    ARCHITECTURAL DECISION: Why ChromaDB?
    ChromaDB is a vector database — it stores embeddings and supports
    fast approximate nearest-neighbour (ANN) search.

    The full stack of vector DB options in production:
      - ChromaDB: open-source, great for prototyping, local or hosted
      - Pinecone: managed cloud, excellent scale, costs money
      - Weaviate: open-source with rich filtering
      - pgvector: Postgres extension — best if you're already on Postgres
      - Qdrant: high-performance, good Rust-based option

    For a Solution Architect role, know *why* you'd pick each one.
    ChromaDB here because: zero setup, local, perfect for learning.

    ARCHITECTURAL DECISION: get_or_create vs create
    get_or_create is safer — won't error if collection already exists.
    """
    collection = chroma_client.get_or_create_collection(
        name=name,
        # ARCHITECTURAL DECISION: distance metric
        # cosine = measures angle between vectors (good for text)
        # l2 = Euclidean distance (common default, slightly different results)
        # ip = inner product (used when vectors are normalised, same as cosine)
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    source_name: str
) -> None:
    """
    Store chunks + embeddings in ChromaDB.

    ARCHITECTURAL DECISION: What metadata to store?
    ChromaDB stores: id, embedding, document (text), metadata (dict).
    The metadata travels with the chunk and is returned on retrieval.
    Store anything you'd want to show the user or filter on:
      - source filename
      - chunk index (for ordering context)
      - approximate token count (for prompt budget management)
    """
    ids = [f"{source_name}__chunk_{c['chunk_index']}" for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [
        {
            "source": source_name,
            "chunk_index": c['chunk_index'],
            "approx_tokens": c['approx_tokens'],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"  Stored {len(chunks)} chunks from '{source_name}' in ChromaDB")


# ──────────────────────────────────────────────
# STEP 4: RETRIEVAL
# ──────────────────────────────────────────────

def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 3
) -> list[dict]:
    """
    Embed the query and find the most similar chunks.

    ARCHITECTURAL DECISION: Why embed the query the same way as the chunks?
    Embeddings only make sense when query and documents live in the same
    vector space — meaning they were embedded by the same model.
    Mixing models (e.g. OpenAI embeddings for docs, Mistral for queries)
    will produce garbage retrieval results.

    ARCHITECTURAL DECISION: top_k
    top_k=3 is a pragmatic default. More chunks = more context = potentially
    better answer, but also more tokens consumed and higher risk of
    irrelevant content confusing the model.

    EXPERIMENT HOOK: Try top_k = 1, 3, 5, 10 and observe answer quality
    and hallucination rate.

    Returns a list of dicts with: text, source, chunk_index, distance
    """
    # Embed the query using the same model as the documents
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # can't request more than stored
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": dist,   # cosine distance: 0 = identical, 2 = opposite
            "similarity": 1 - dist,  # intuitive: 1 = identical, 0 = unrelated
        })

    return chunks


# ──────────────────────────────────────────────
# STEP 5: GENERATION
# ──────────────────────────────────────────────

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Assemble the prompt from retrieved context + user query.

    ARCHITECTURAL DECISION: Prompt structure matters enormously.
    The naive approach: dump all chunks, then ask the question.
    Better: clearly delineate context from question, instruct the model
    to use ONLY the provided context, and tell it what to do if the
    answer isn't there.

    ARCHITECTURAL DECISION: "Answer only from the context" instruction.
    This is the core RAG anti-hallucination mechanism. Without it, the model
    will blend retrieved context with its training data, making it hard to
    know what's grounded vs fabricated. In production you might want some
    blend, but for learning: start strict.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Source: {chunk['source']}, chunk {chunk['chunk_index']}, "
            f"similarity: {chunk['similarity']:.3f}]\n{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions based strictly on provided context.

CONTEXT:
{context_str}

---

QUESTION: {query}

INSTRUCTIONS:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information to answer, say:
  "I cannot find this in the provided documents."
- Do not use your general knowledge or make up facts.
- Be concise and direct.

ANSWER:"""

    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str = "mistral-large-latest",
    temperature: float = 0.1,
) -> dict:
    """
    Generate an answer from context chunks using Mistral.

    ARCHITECTURAL DECISION: Why low temperature (0.1)?
    Temperature controls randomness. For factual Q&A over documents,
    you want the model to be deterministic and stick to the context.
    Higher temperature = more creative/varied = more hallucination risk.
    Use 0.7–0.9 for creative tasks, 0.0–0.2 for factual retrieval.

    ARCHITECTURAL DECISION: Model choice.
    - mistral-large-latest: Best quality, highest cost, slower
    - mistral-small-latest: Fast, cheap, good for simpler queries
    - open-mistral-7b: Cheapest, good for simple extraction tasks
    For a learning pipeline: start with large to see best-case quality,
    then downgrade to understand the quality/cost tradeoff.

    Returns a dict with answer text + metadata for analysis.
    """
    prompt = build_prompt(query, retrieved_chunks)

    # Approximate total prompt tokens for context window awareness
    approx_prompt_tokens = len(prompt.split()) * 1.33

    response = mistral_client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "query": query,
        "model": model,
        "chunks_used": len(retrieved_chunks),
        "approx_prompt_tokens": int(approx_prompt_tokens),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "retrieved_chunks": retrieved_chunks,
    }


# ──────────────────────────────────────────────
# TOP-LEVEL PIPELINE FUNCTIONS
# ──────────────────────────────────────────────

def ingest_document(
    file_path: str,
    collection_name: str = "rag_documents",
    chunk_size: int = 512,
    overlap: int = 50,
) -> chromadb.Collection:
    """
    Full ingestion pipeline: file → chunks → embeddings → ChromaDB.

    This is the offline / batch step. In production this runs when
    new documents are added, not on every query.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    print(f"\n📄 Ingesting: {path.name}")
    text = path.read_text(encoding="utf-8")
    print(f"  Document length: {len(text):,} characters, ~{int(len(text.split()) * 1.33):,} tokens")

    print(f"\n✂️  Chunking (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"  Created {len(chunks)} chunks")

    print(f"\n🔢 Embedding {len(chunks)} chunks via mistral-embed...")
    texts = [c['text'] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    print(f"\n💾 Storing in ChromaDB collection '{collection_name}'...")
    collection = create_collection(collection_name)
    store_chunks(collection, chunks, embeddings, source_name=path.name)

    print(f"\n✅ Ingestion complete. {collection.count()} total chunks in collection.")
    return collection


def query_pipeline(
    collection: chromadb.Collection,
    question: str,
    top_k: int = 3,
    model: str = "mistral-large-latest",
    temperature: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Full query pipeline: question → embed → retrieve → generate → answer.

    This is the online / real-time step. Runs on every user query.
    """
    if verbose:
        print(f"\n🔍 Query: {question}")
        print(f"   Retrieving top {top_k} chunks...")

    chunks = retrieve_chunks(collection, question, top_k=top_k)

    if verbose:
        print(f"\n📦 Retrieved chunks:")
        for c in chunks:
            print(f"   [{c['source']} | chunk {c['chunk_index']} | sim={c['similarity']:.3f}]")
            print(f"   {c['text'][:120]}...")

    if verbose:
        print(f"\n🤖 Generating answer with {model}...")

    result = generate_answer(question, chunks, model=model, temperature=temperature)

    if verbose:
        print(f"\n💬 Answer:")
        print(f"   {result['answer']}")
        print(f"\n📊 Token usage: {result['usage']['total_tokens']} total "
              f"({result['usage']['prompt_tokens']} prompt + "
              f"{result['usage']['completion_tokens']} completion)")

    return result


# ──────────────────────────────────────────────
# ENTRY POINT — basic demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    doc_path = sys.argv[1] if len(sys.argv) > 1 else "sample_document.txt"

    # 1. Ingest
    collection = ingest_document(doc_path, chunk_size=512, overlap=50)

    # 2. Interactive query loop
    print("\n" + "="*60)
    print("RAG Pipeline ready. Type a question (or 'quit' to exit).")
    print("="*60)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        query_pipeline(collection, question, top_k=3)
