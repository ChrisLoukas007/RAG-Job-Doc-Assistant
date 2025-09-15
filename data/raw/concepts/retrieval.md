# Retrieval

Goal: **find the most relevant chunks** for a query so the LLM answers from the right context.

## Pipeline
1. **Chunking**: split docs (e.g., 600–1000 chars, 80–150 overlap).
2. **Embed** chunks into vectors.
3. **First-stage search**: fast **top-k** vector search (FAISS or similar).
4. **Rerank (optional)**: cross-encoder or LLM reorders top candidates.
5. **Pass top-3–5** chunks to the LLM.

## Chunking strategies
- Prefer **semantic boundaries** (headings, paragraphs).
- Use **overlap** so answers crossing boundaries are still captured.
- Keep chunks under the LLM context budget when concatenated.

## Search strategies
- **Pure dense** (vector): default; great semantic recall.
- **Hybrid** (BM25 + vector): improves keyword/rare term matching.
- **MMR (Max Marginal Relevance)**: diversify results to reduce redundancy.

## Reranking (precision boost)
- Cross-encoder models (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) or ColBERT.
- Use reranker when accuracy matters more than latency.

## Metadata filtering
- Tag chunks with fields like `doc_id`, `section`, `date`, `source`.
- Filter by `doc_id` or `source` to scope the search.

## Tuning knobs
- `top_k` (first stage): 10–50 (then rerank).
- `final_k` (to LLM): 3–5.
- Similarity **threshold**: below → treat as **no hit**.

## Evaluation
- **Recall@k**: % of queries where a gold chunk is in top-k.
- **MRR**: ranking quality.
- **Latency**: p50/p95 for search + rerank.

## Troubleshooting
- Missing facts → increase `top_k`, improve chunking, add hybrid, or add docs.
- Off-topic context → increase threshold, apply reranking, add filters.
- Slow → switch off reranker or lower `top_k`.

## Example (pseudocode)
```python
chunks = chunk(doc, size=800, overlap=120)
vecs = embed(chunks, model=EMBEDDING_MODEL)
cands = vector_search(query, vecs, top_k=30)
reranked = cross_encode_rerank(query, cands)  # optional
context = take(reranked, k=4)
answer = llm_answer(query, context)