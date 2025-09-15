# What is RAG?
Retrieval-Augmented Generation (RAG) retrieves relevant chunks from your documents and asks an LLM to answer **using only those chunks**, with citations.

# Why use RAG?
It grounds answers in your private knowledge, reduces hallucinations, and provides traceable sources.

# Starter settings
Use chunk size ~600–1000 characters with 80–150 overlap. Retrieve top-k 3–5. If an answer is not in the context, respond “I don’t know.”
