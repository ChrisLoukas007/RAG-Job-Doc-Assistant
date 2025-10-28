import os
import textwrap

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))


def load_vs():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


def show_results(vs, q: str, k: int = TOP_K):
    print(f"\nQ: {q}\nTop-{k} retrieved chunks:")
    docs = vs.similarity_search_with_score(q, k=k)
    for i, (d, score) in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        snippet = textwrap.shorten(d.page_content.replace("\n", " "), width=220)
        print(f"{i:>2}. score={score:.4f} | {src}\n    {snippet}")


if __name__ == "__main__":
    vs = load_vs()
    # Try a failing question from eval/predictions.jsonl
    q = input("Type a question to debug retrieval: ").strip()
    show_results(vs, q)
