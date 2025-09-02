# RAG-Job-Doc-Assistant
A compact Retrieval-Augmented Generation service: ingest your PDFs/notes → ask questions → get grounded answers with sources.

## Features
- LangChain LCEL pipeline (retriever + prompt + LLM)
- FAISS vector index (local, fast)
- Pluggable LLM (Ollama or OpenAI)
- FastAPI endpoints: `/health`, `/ingest`, `/query`, `/feedback`
- Dockerized, with optional Ollama via docker-compose
- Basic eval script (semantic similarity)

## Quickstart
1. `pip install -r requirements.txt`
2. Put docs into `data/raw/`
3. `python -m app.ingest`
4. `uvicorn app.main:app --reload`
5. `POST /query {"question": "..."}`

## Env
See `.env.example`.

## Swap vector DB
Use FAISS by default. For Qdrant: run a qdrant container and switch vector store code (PR welcome).

## License
MIT
