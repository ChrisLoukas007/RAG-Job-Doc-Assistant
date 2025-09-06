# 🧑‍💻 RAG Helpdesk Assistant

A **Retrieval-Augmented Generation (RAG)** application designed to answer questions grounded in a custom knowledge base.  

This project showcases how to build a modern AI microservice using:

- **LangChain** for chaining retrieval + LLM orchestration
- **FastAPI** for serving APIs
- **Hugging Face embeddings (MiniLM)** + **FAISS** vector database
- **Pluggable LLM providers**: [Ollama](https://ollama.ai) (local models) or OpenAI
- **Docker** for reproducible deployment
- **Evaluation** with semantic similarity metrics

It demonstrates the **end-to-end workflow** of a RAG pipeline: ingesting raw documents → building embeddings → querying with context → serving results through an API.

---

## ✨ Features

- 📂 **Document ingestion**: Process Markdown, TXT, or PDF docs into embeddings.
- 🔎 **Vector retrieval**: Use FAISS for efficient similarity search.
- 🤖 **Grounded answers**: Combine retrieved chunks with an LLM to reduce hallucinations.
- 📑 **Source attribution**: Return answers *with linked sources*.
- 🐳 **Portable deployment**: Run locally or with Docker in one command.
- 🧪 **Evaluation pipeline**: Compare predictions against gold answers using semantic similarity.
