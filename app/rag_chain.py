# IMPORTS - Getting all the AI tools we need
import os
from typing import Dict, Any, AsyncIterator

# LangChain imports - AI framework components
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings  # text to vector converter 
from langchain_core.output_parsers import StrOutputParser # for parsing AI output

# for handling AI responses
from langchain_core.messages import BaseMessage

# AI PROMPT TEMPLATE - Instructions for the AI
PROMPT = ChatPromptTemplate.from_template("""
You are a concise AI helpdesk agent. Use ONLY the provided context to answer.
If the answer is not in context, say you don't know.

Question: {question}
Context:
{context}

Answer briefly, then list sources as bullet points with filenames.
""".strip())

# VECTOR STORE LOADER - Load the pre-built vector store from disk
def load_vs(index_dir: str, embedding_model: str):
    # Initialize the same embedding model used during ingestion
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Load FAISS vector database from disk 
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

# DOCUMENT FORMATTER - Format retrieved documents into readable context string with source references
def format_docs(docs):
    """    
    Input: [doc1, doc2, doc3]
    Output: "[1] file1.pdf\nContent here...\n\n[2] file2.pdf\nMore content..."
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs):
        # Get the source file name (or "unknown" if missing)
        source = doc.metadata.get('source', 'unknown')
        
        # Create a numbered section with source and content
        section = f"[{i+1}] {source}\n{doc.page_content}"
        formatted_parts.append(section)
    
    # Join all sections with double newlines for readability
    return "\n\n".join(formatted_parts)

# CONTEXT COMBINER - Combine user question with formatted document context
def combine_context(x: Dict[str, Any]) -> Dict[str, str]:
    """   
    Input: {"question": "How do I login?", "docs": [doc1, doc2, doc3]}
    Output: {"question": "How do I login?", "context": "[1] manual.pdf\nClick login..."}
    """
    return {
        "question": x["question"],
        "context": format_docs(x["docs"]) # Format documents into readable context
    }

# CHAIN BUILDER - Create the complete RAG chain: retrieval + generation
def make_chain(index_dir: str, embedding_model: str, llm_provider: str):
   
    # Step 1: Load the knowledge database (filing cabinet)
    vs = load_vs(index_dir, embedding_model)
    
    # Step 2: Create a searcher that finds the 4 most relevant documents
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    
    # Step 3: Choose which AI brain to use
    if llm_provider == "openai":
        # Use OpenAI's GPT models (requires API key in environment)
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    else:
        # Use local Ollama models (requires Ollama server running)
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),  # Where Ollama is running
            temperature=0,          # Consistent answers
            num_predict=512,        # Reasonable length
            top_p=0.9,             # Focused but natural
            repeat_penalty=1.1,     # No repetition
        )

    # Step 4: Build the complete pipeline
    chain = (
        # 1: Take user input and search for relevant docs in parallel
        RunnableMap({
            "question": RunnablePassthrough(),    # Pass the question through unchanged
            "docs": retriever                     # Search for relevant documents
        })
        # 2: Combine question and documents into AI-readable format
        | combine_context
        # 3: Apply the prompt template (give AI its instructions)
        | PROMPT
        # 4: Let the AI generate the final answer
        | llm
        | StrOutputParser()  # Ensure output is plain text
    )
    
    # Return both the complete chain and the searcher
    return chain, retriever

# STREAMING FUNCTION - Real-time answer generation
async def stream_chain_answer(chain, question: str) -> AsyncIterator[str]:
    """
    Stream the model's answer token-by-token.
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