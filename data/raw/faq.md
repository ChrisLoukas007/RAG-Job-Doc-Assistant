# FAQ

## How can I test the endpoints?
- Open `http://localhost:8000/docs` (Swagger UI) and try the routes.
- Or use curl/Postman (see examples in `endpoints.md`).

## What should I do if the bot can't find an answer?
- The app will reply **"I don't know"** (and may return HTTP 404 if configured).
- Try rephrasing the question, or **ingest more relevant docs**.

## How do I ingest (add) documents?
- Put files under `data/raw/` and run the ingest script (see `retrieval.md`).
- Supported: `.md`, `.txt`, `.pdf` (extendable). Very large PDFs are chunked automatically.

## Why did I get a weird answer?
- The answer is limited to retrieved chunks. If retrieval misses the right chunk, the LLM won’t see it.
- Fixes: better chunking, increase `top_k`, add reranking, add more/better docs (see `retrieval.md`).

## How do I switch models (LLM / embeddings)?
- Set env vars in `.env`:
  - `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (or another)
  - `LLM_PROVIDER=ollama|openai`
  - `OPENAI_API_KEY=…` when using OpenAI
- Restart the server.

## Where are the indexes and logs stored?
- Index: `data/index/`
- Logs: `data/logs/`
- Configure via `INDEX_DIR` and `LOGS_DIR`.

## How do I reset / rebuild the index?
1. Delete `data/index/` (or move it aside).
2. Re-run the ingest command to rebuild.
3. Restart the API.

## What’s the difference between **Ask** and **Stream** in the UI?
- **Ask**: waits for the full answer, then renders once.
- **Stream**: token-by-token server-sent events (SSE) for faster perceived latency.

## How do I speed things up?
- Use smaller `chunk_size` with overlap, enable **Approximate NN** indexing, and keep `top_k` small.
- Turn on streaming; use a lighter LLM for draft + heavier reranker for precision.

## Security / privacy basics?
- The LLM only sees retrieved chunks, not your whole corpus.
- Don’t upload sensitive files to a shared environment.
- Consider disabling `/upload` in production or adding auth.

## Docker quickstart?
- `docker compose up --build`
- Volumes map `./data` so your index/logs persist between runs (see `docker_basics.md`).

## How do I evaluate quality?
- Track **Retrieval Recall@k** and **Answer Faithfulness**.
- Keep a small gold set of Q/A and run the eval script regularly.
