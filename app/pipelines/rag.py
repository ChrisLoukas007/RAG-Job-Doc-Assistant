# IMPORTS - Getting all the AI tools we need
import os
from typing import Any, AsyncIterator, Dict

from langchain_core.output_parsers import (
    StrOutputParser,  # for parsing AI output as plain text
)

# LangChain imports - AI framework components
from langchain_core.runnables import (  # for building AI pipelines , RunnablePassthrough is used to pass the question through unchanged, RunnableMap is used to create a mapping of inputs to runnables
    RunnableMap,
    RunnablePassthrough,
)

# App-local imports
from ..core.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)
from ..providers.llm_ollama import get_ollama_llm
from ..providers.llm_openai import get_openai_llm
from ..providers.vector_faiss import load_vs
from .prompts import PROMPT


# DOCUMENT FORMATTER - Format retrieved documents into readable context string with source references
def format_docs(docs):
    """
    Input: [doc1, doc2, doc3]
    """
    formatted_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        section = f"[{i + 1}] {source}\n{doc.page_content}"
        formatted_parts.append(section)
    return "\n\n".join(formatted_parts)


# CONTEXT COMBINER - Combine user question with formatted document context
def combine_context(x: Dict[str, Any]) -> Dict[str, str]:
    """
    Input: {"question": "How do I login?", "docs": [doc1, doc2, doc3]}
    Output: {"question": "How do I login?", "context": "[1] manual.pdf\nClick login..."}
    """
    return {
        "question": x["question"],
        "context": format_docs(x["docs"]),  # Format documents into readable context
    }


# CHAIN BUILDER - Create the complete RAG chain: retrieval + generation
def make_chain(
    index_dir: str = INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    llm_provider: str = LLM_PROVIDER,
):
    # Step 1: Load the knowledge database (filing cabinet)
    vs = load_vs(index_dir, embedding_model)

    # Step 2: Create a searcher that finds the 3 most relevant documents
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    # Step 3: Choose which AI brain to use - OpenAI or Ollama
    if llm_provider == "openai":
        llm = get_openai_llm(
            default_model=os.getenv("OPENAI_MODEL", OPENAI_MODEL), temperature=0
        )
    else:
        llm = get_ollama_llm(
            model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            temperature=0,
        )

    # Step 4: Build the complete pipeline - Language model + prompt + retriever + context combiner
    # The chain processes the question, retrieves documents, formats context, applies the prompt, and generates an answer
    chain = (
        RunnableMap(
            {
                "question": RunnablePassthrough(),  # Pass the question through unchanged
                "docs": retriever,  # Search for relevant documents
            }
        )
        | combine_context  # Combine question and documents into AI-readable format
        | PROMPT  # Apply the prompt template (give AI its instructions)
        | llm  # Let the AI generate the final answer
        | StrOutputParser()  # Ensure output is plain text
    )
    return chain, retriever


# STREAMING FUNCTION - Real-time answer generation (token-by-token)
async def stream_chain_answer(chain, question: str) -> AsyncIterator[str]:
    """
    Stream the model's answer token-by-token.
    """
    async for event in chain.astream_events(question, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            piece = getattr(chunk, "content", "") or ""
            if piece:
                yield piece
