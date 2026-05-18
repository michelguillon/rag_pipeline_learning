# RAG Pipeline — Week 1

A document Q&A system built from scratch with Mistral + ChromaDB.
Every architectural decision is annotated in the source code.

## Architecture

```
INGEST (offline/batch):
  Document → chunk_text() → embed_texts() → ChromaDB
              ↑                ↑
          512 tokens        mistral-embed
          50 overlap         (1024 dims)

QUERY (online/real-time):
  Question → embed_texts() → ChromaDB.query() → top-k chunks
                                                       ↓
                                              build_prompt()
                                                       ↓
                                         mistral-large completion
                                                       ↓
                                                    Answer
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Mistral API key
export MISTRAL_API_KEY=your_key_here

# Hours 1-2: Run the basics experiments
python mistral_basics.py

# Hours 3-7: Run the full pipeline
python rag_pipeline.py your_document.txt

# Hours 8-10: Stress test
python stress_test.py your_document.txt
```

## Architectural Decisions

### Chunking
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Chunk size | 512 tokens | Balance between context richness and retrieval precision |
| Overlap | 50 tokens | Prevents key sentences from being split at boundaries |
| Method | Word-approximate | Avoids tokeniser dependency; ±5% accuracy is acceptable |

### Embedding
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Model | `mistral-embed` | Same provider as completion = consistent vector space |
| Dimensions | 1024 | Fixed by model; higher = richer representation |
| Batching | 32 texts/batch | Reduces API round-trips; respects rate limits |

### Storage
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Vector DB | ChromaDB | Zero setup, local, perfect for prototyping |
| Distance metric | Cosine | Angular distance; better for text than L2 |
| Client | EphemeralClient | Re-ingests on each run; forces re-examination of ingestion |

### Retrieval
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| top_k | 3 | Balances context richness vs. token cost vs. noise |
| Query embedding | Same model as chunks | Must share the same vector space |

### Generation
| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Model | `mistral-large-latest` | Best quality for learning; downgrade after understanding tradeoffs |
| Temperature | 0.1 | Near-deterministic; reduces hallucination in factual Q&A |
| Context instruction | "Answer ONLY from context" | Core RAG anti-hallucination mechanism |

---

## Stress Test Findings

*Fill this in after running stress_test.py. These are interview gold.*

### Chunk Size: 256 vs 512 vs 1024

| Metric | 256 tokens | 512 tokens | 1024 tokens |
|--------|-----------|-----------|-------------|
| Chunks created | — | — | — |
| Avg retrieval similarity | — | — | — |
| Hallucination rate | — | — | — |
| Observation | | | |

**My finding:**
> [Write your observation here after running the experiments]

### top_k: 1 vs 5 vs 10

| Metric | top_k=1 | top_k=5 | top_k=10 |
|--------|---------|---------|---------|
| Avg tokens/query | — | — | — |
| Multi-part answer quality | — | — | — |
| Noise introduced | — | — | — |

**My finding:**
> [Write your observation here]

### Overlap: 0 vs 50 vs 100

**My finding:**
> [Write your observation here]

### Hallucination test

- Question used: "What is the population of Mars?"
- Expected: model refuses with "I cannot find this in the provided documents"
- Actual:

**My finding:**
> [Write what happened and what the prompt change did/didn't fix]

### Long document degradation

- Document used:
- Length:
- Where it started to fail:

**My finding:**
> [Write your observation here]

---

## What I Can Now Say in an Interview

> "I built a RAG pipeline from scratch using Mistral's embedding and completion APIs.
> I understand the architectural tradeoffs around chunk size — smaller chunks give more
> precise retrieval but lose surrounding context; larger chunks are richer but dilute the
> relevance signal. I found that top_k=3 is a pragmatic default: top_k=1 misses
> multi-part answers while top_k=10 burns tokens and introduces noise. The most
> important anti-hallucination mechanism is instructing the model to only answer from
> provided context — without this, it blends retrieved content with training data in
> ways that are hard to detect. I also understand when to move from ChromaDB to a
> managed vector store like Pinecone, and why cosine distance outperforms L2 for text."

---

## Production Upgrade Path

When this moves to production, these are the changes to make:

| Component | Prototype | Production |
|-----------|-----------|------------|
| Vector DB | ChromaDB EphemeralClient | ChromaDB PersistentClient or Pinecone |
| Chunking | Word-approximate | Exact token count via tiktoken |
| Embedding | Sequential batches | Async parallel with rate limiting |
| Retrieval | Cosine similarity only | Hybrid: semantic + keyword (BM25) |
| Hallucination | Prompt instruction | + confidence scoring + citation tracking |
| Observability | Print statements | LangSmith / Arize / custom logging |
