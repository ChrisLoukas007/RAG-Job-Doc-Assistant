from pathlib import Path  # Easy way to work with file paths

from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Tools to read PDF and text files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Breaks documents into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # Converts text to numbers (vectors)
from langchain_community.vectorstores import FAISS  # Database for storing and searching vectors

import os  # Access environment variables

def build_index(raw_dir: str, index_dir: str, embedding_model: str):
    """
    Processes documents into searchable vector database
    """
    # Convert string paths to Path objects for easier manipulation
    raw_path = Path(raw_dir)
    index_path = Path(index_dir)
    
    # Create output directory if it doesn't exist
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Container for all loaded documents
    docs = []
    
    # Recursively search through all files in directory and subdirectories
    for p in raw_path.rglob("*"):
        if p.suffix.lower() == ".pdf":
            # Load PDF files using specialized PDF loader
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in [".txt", ".md", ".markdown"]:
            # Load text-based files using generic text loader
            docs.extend(TextLoader(str(p)).load())
    
    # Initialize text splitter with optimal chunk size for embeddings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # Size of each text chunk
        chunk_overlap=120  # Overlap to preserve context between chunks
    )
    
    # Split all documents into smaller, manageable chunks
    chunks = splitter.split_documents(docs)
    
    # Initialize embedding model to convert text to vectors
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create FAISS vector store from document chunks
    vs = FAISS.from_documents(chunks, emb)
    
    # Save the vector database to disk for later use
    vs.save_local(index_dir)
    
    # Return number of chunks created for monitoring
    return len(chunks)

# Only run when script is executed directly
if __name__ == "__main__":
    # Load environment variables from .env file
    from dotenv import load_dotenv; load_dotenv()
    
    # Build the index using environment variables with fallback defaults
    built = build_index(
        os.getenv("DATA_DIR", "./data") + "/raw",                                    # Input directory
        os.getenv("INDEX_DIR", "./data/index"),                                      # Output directory
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),     # AI model for embeddings
    )
    
    # Report processing results
    print(f"Ingested chunks: {built}")
