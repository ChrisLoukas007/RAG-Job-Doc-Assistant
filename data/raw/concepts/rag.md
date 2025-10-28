# What is RAG?

Retrieval-Augmented Generation (RAG) retrieves relevant chunks from your documents and asks an LLM to answer **using only those chunks**, with citations.

# Why use RAG?

It grounds answers in your private knowledge, reduces hallucinations, and provides traceable sources.

# Starter settings

Use chunk size ~600–1000 characters with 80–150 overlap. Retrieve top-k 3–5. If an answer is not in the context, respond “I don’t know.”

# Retrieval-Augmented Generation (RAG)

## Overview

RAG combines retrieval-based and generative approaches to create more accurate, grounded responses. Instead of relying solely on an LLM's training data, RAG augments generation with relevant retrieved context.

## Key Components

### 1. Document Processing

- Text extraction from various formats
- Chunking strategies for optimal retrieval
- Metadata extraction and preservation

### 2. Embedding Generation

- Vector representation of text chunks
- Model selection considerations
- Dimensionality and performance tradeoffs

### 3. Vector Storage

- FAISS index organization
- Similarity search optimization
- Index maintenance and updates

### 4. Retrieval Process

- Query processing
- Relevance scoring
- Re-ranking strategies
- Hybrid retrieval approaches

### 5. Context Integration

- Prompt engineering
- Context window management
- Source attribution

## Benefits

- Reduced hallucination
- Up-to-date information
- Verifiable responses
- Domain adaptation
- Cost efficiency

## Common Challenges

- Context length limitations
- Retrieval quality impact
- Prompt engineering complexity
- Response latency
