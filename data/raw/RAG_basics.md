RAG fetches relevant text chunks from a private corpus and asks an LLM to answer using only those chunks, with citations.
Start with chunk size ~600–1000 chars and 80–150 overlap. Retrieve top-k=3–5. If answer not in context, say “I don’t know.”
FAISS is a local vector index; Qdrant/Pinecone are hosted alternatives.
