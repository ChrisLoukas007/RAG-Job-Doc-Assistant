import os
from typing import Dict, Any, AsyncIterator

from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# NEW: we’ll stream using the LCEL event API
from langchain_core.messages import BaseMessage


# System prompt template that defines the AI assistant's behavior
PROMPT = ChatPromptTemplate.from_template("""
You are a concise AI helpdesk agent. Use ONLY the provided context to answer.
If the answer is not in context, say you don't know.

Question: {question}
Context:
{context}

Answer briefly, then list sources as bullet points with filenames.
""".strip())

def load_vs(index_dir: str, embedding_model: str):
    """
    Load the pre-built vector store from disk ( open the store and read the contents )
    """
    # Initialize the same embedding model used during ingestion
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Load FAISS vector database from disk (allow_dangerous_deserialization needed for FAISS)
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

def format_docs(docs):
    """
    Format retrieved documents into readable context string with source references
    """
    # Combine documents with source info and numbering for reference
    return "\n\n".join([
        f"[{i+1}] {d.metadata.get('source','unknown')}\n{d.page_content}" 
        for i, d in enumerate(docs)
    ])

def combine_context(x: Dict[str, Any]) -> Dict[str, str]:
    """
    Combine user question with formatted document context
    This replaces the lambda function for better readability and type safety
    """
    return {
        "question": x["question"],
        "context": format_docs(x["docs"])
    }

def make_chain(index_dir: str, embedding_model: str, llm_provider: str):
    """
    Create the complete RAG chain: retrieval + generation
    Uses RunnableMap instead of dict for better LangChain compatibility
    """
    # Load the vector store created by ingest.py
    vs = load_vs(index_dir, embedding_model)
    
    # Create retriever that finds top 4 most similar documents
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    
    # Choose LLM based on provider preference
    if llm_provider == "openai":
        # Use OpenAI's GPT models (requires API key in environment)
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    else:
        # Use local Ollama models (requires Ollama server running)
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    
    # Build the RAG chain pipeline where the LangChain pipeline philosophy kicks in 
    # 1. RunnableMap: Take user input and retrieve relevant docs in parallel
    # 2. combine_context: Format docs into context string with question
    # 3. PROMPT: Apply template to create final prompt
    # 4. llm: Generate answer using chosen language model
    chain = (
        RunnableMap({"question": RunnablePassthrough(), "docs": retriever})
        | combine_context
        | PROMPT
        | llm
    )
    
    # Return both chain and retriever for flexibility
    return chain, retriever

# ===========================
# streaming async generator
# ===========================
async def stream_chain_answer(chain, question: str) -> AsyncIterator[str]:
    """
    Stream the model's answer token-by-token.

    How it works:
    - `chain.astream_events(..., version="v1")` emits many events while the chain runs.
    - When the LLM sends a token, we receive an `on_chat_model_stream` event.
    - We extract `chunk.content` (the new text delta) and yield it immediately.
    """
    # Start the chain with the question and get events one by one
    async for event in chain.astream_events(question, version="v1"):
        
        # Check if this event is about the AI generating text
        if event["event"] == "on_chat_model_stream":
            
            # Get the data part that contains the new text piece
            chunk = event["data"]["chunk"]
            
            # Safely get the text content (or empty string if none)
            piece = getattr(chunk, "content", "") or ""
            
            # Only send back the text if it's not empty
            if piece:
                # Send this text piece to whoever called this function
                yield piece