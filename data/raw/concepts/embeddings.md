# Embeddings

Embeddings turn text into vectors so similar meaning lies close in vector space.

## Model choices
- **General-purpose**: `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim).
- **Long-context / higher quality**: `all-mpnet-base-v2`, `bge-*`.
- **Multilingual**: `distiluse-base-multilingual-cased-v2`, `bge-m3`.
- Pick based on **domain**, **language**, **latency**.

## Good practices
- **Normalize** vectors (L2) when your index expects it.
- Lowercase + strip control chars; keep punctuation (often helps).
- **Deduplicate** near-identical chunks before indexing.
- Store metadata (doc_id, title, section) alongside vectors.

## Indexing
- Use FAISS (IVF/HNSW) or similar for **Approximate NN**.
- Rebuild or **merge-in** when adding many documents.
- Persist under `INDEX_DIR` so Docker restarts don’t lose it.

## Updating models
- Changing the embedding model ≠ compatible vectors.
- If you switch models, **re-embed** and rebuild the index.

## Limits & pitfalls
- Very short chunks can be noisy; extremely long chunks blur meaning.
- Rare tokens & numbers: consider **hybrid search** with BM25.
- Code/IDs: add **character n-grams** or exact filters.

## Env & config
- `EMBEDDING_MODEL=...`
- Batch size: tune for throughput vs. memory.
- Cache embeddings on disk to avoid recomputation.

## Example: chunk + embed (Python-ish)
```python
def ingest(paths, model="sentence-transformers/all-MiniLM-L6-v2"):
    docs = load(paths)
    chunks = smart_chunk(docs, size=800, overlap=120)
    vecs = embed(chunks, model)
    index = build_faiss(vecs)  # with metadata
    index.save(INDEX_DIR)