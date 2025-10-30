import os
from pathlib import Path  # Makes working with file paths much easier than basic strings

from langchain_community.document_loaders import (  # Special tools to read PDF and text files
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import (
    FAISS,  # A special database that can find similar text quickly
)
from langchain_huggingface import (
    HuggingFaceEmbeddings,  # Updated tool to convert text into numbers (vectors)
)
from langchain_text_splitters import (  # Smart scissors to cut documents
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from opentelemetry import trace  # OpenTelemetry tracing tools for observability

from app.observability.phoenix import (
    configure_tracing,  # Import tracing configuration function
)

configure_tracing()  # Set up tracing for observability

tracer = trace.get_tracer("rag-ingest-pipeline")  # Create a tracer for this module


def build_index(raw_dir: str, index_dir: str, embedding_model: str):
    """
    Build a searchable index from documents in a folder.
    raw_dir: folder where your original documents live
    index_dir: folder where we'll save our searchable database
    embedding_model: the AI model that converts text to numbers
    """

    with tracer.start_as_current_span(
        "build_index",
        attributes={
            "rag.raw_dir": raw_dir,
            "rag.index_dir": index_dir,
            "rag.embedding_model": embedding_model,
        },
    ):
        # Convert text paths to Path objects - this makes file operations much safer and easier
        raw_path = Path(raw_dir)  # Where we read documents from
        index_path = Path(index_dir)  # Where we save our searchable database

        # Create the output folder if it doesn't exist yet
        index_path.mkdir(parents=True, exist_ok=True)

        # Create an empty list to hold all our loaded documents
        docs = []

        # Walk through every single file in the directory and all subdirectories
        # rglob("*") means "find every file, no matter how deep in folders"
        for file_path in raw_path.rglob("*"):
            with tracer.start_as_current_span(
                "load_file",
                attributes={
                    "rag.file_path": str(file_path),
                    "rag.file_suffix": file_path.suffix.lower(),
                },
            ):
                if file_path.suffix.lower() == ".pdf":
                    # Use special PDF loader that can extract text from PDF files
                    docs.extend(PyPDFLoader(str(file_path)).load())
                elif file_path.suffix.lower() in [".txt", ".md", ".markdown"]:
                    # Use generic text loader for plain text files
                    docs.extend(TextLoader(str(file_path)).load())

        # First scissors: cuts markdown files by headings (# ## ###)
        # This keeps related sections together instead of cutting randomly
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),  # Split on main headings
                ("##", "h2"),  # Split on sub-headings
                ("###", "h3"),  # Split on sub-sub-headings
            ]
        )

        # Second scissors: cuts any text into smaller, manageable chunks
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Each piece should be about 500 characters (smaller = more precise)
            chunk_overlap=100,  # Overlap pieces by 100 chars so we don't lose context at boundaries
        )

        # Create empty list to hold all our final text chunks
        chunks = []

        with tracer.start_as_current_span("split_documents"):
            # Process each document we loaded earlier
            for document in docs:
                # Check if this document is a markdown file by looking at its source path
                source_file = document.metadata.get("source", "").lower()

                if source_file.endswith((".md", ".markdown")):
                    # For markdown files: first split by headings, then split further by size
                    markdown_sections = md_splitter.split_text(
                        document.page_content
                    )
                    for section in markdown_sections:
                        section.metadata.update(document.metadata)
                        chunks.extend(char_splitter.split_documents([section]))
                else:
                    # For non-markdown files: just split by character count
                    chunks.extend(char_splitter.split_documents([document]))

        # Add helpful tracking information to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            source_path = chunk.metadata.get("source", "unknown")
            chunk.metadata["filename"] = Path(source_path).name
            trace.get_current_span().add_event(
                name="chunk_created",
                attributes={
                    "rag.chunk_id": i,
                    "rag.filename": chunk.metadata["filename"],
                    "rag.chunk_length": len(chunk.page_content),
                },
            )

        with tracer.start_as_current_span(
            "embed_chunks", attributes={"rag.chunk_count": len(chunks)}
        ):
            # Create our AI model that converts text into numbers (vectors)
            embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model)

            # Create the FAISS vector database from all our chunks
            vector_store = FAISS.from_documents(chunks, embeddings_model)

        with tracer.start_as_current_span(
            "persist_faiss", attributes={"rag.index_path": str(index_path)}
        ):
            # Save the entire searchable database to disk
            vector_store.save_local(str(index_path))

        # Return how many chunks we created (useful for monitoring)
        return len(chunks)


# This part only runs when you execute this script directly
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    total_chunks = build_index(
        raw_dir=os.getenv("DATA_DIR", "./data") + "/raw",
        index_dir=os.getenv("INDEX_DIR", "./data/index"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    print(f"Built index with {total_chunks} chunks")
