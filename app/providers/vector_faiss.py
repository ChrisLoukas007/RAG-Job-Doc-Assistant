# VECTOR STORE LOADER - Load the pre-built vector store from disk (using FAISS)
from langchain_community.vectorstores import FAISS # FAISS vector database 
from langchain_huggingface import HuggingFaceEmbeddings  # text to vector converter 

def build_embeddings(model_name: str):
    # Initialize the same embedding model used during ingestion 
    return HuggingFaceEmbeddings(model_name=model_name) 

def load_vs(index_dir: str, embedding_model: str):
    emb = build_embeddings(embedding_model)
    # Load FAISS vector database from disk 
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
# Note: allow_dangerous_deserialization=True is needed for some FAISS versions