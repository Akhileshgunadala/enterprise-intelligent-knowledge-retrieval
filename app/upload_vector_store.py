from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

USER_INDEX_DIR = Path("user_index")

CHUNKS_FILE = USER_INDEX_DIR / "chunks.json"
INDEX_FILE = USER_INDEX_DIR / "faiss.index"
METADATA_FILE = USER_INDEX_DIR / "metadata.json"


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Create User Vector Store
# --------------------------------------------------

def create_vector_store(chunks):
    """
    Create a FAISS vector store from uploaded-document
    chunks.

    This does NOT modify the benchmark vector store.
    """

    print("\n===================================")
    print("CREATING USER VECTOR STORE")
    print("===================================\n")

    if not chunks:
        raise ValueError(
            "No chunks were provided."
        )

    print(
        f"Received {len(chunks)} chunks."
    )

    # Create directory
    USER_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save chunks
    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Chunks saved to: {CHUNKS_FILE}"
    )

    # Load embedding model
    print(
        f"\nLoading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # Extract text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nFAISS index saved to: "
        f"{INDEX_FILE}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )

    print(
        "\nUser vector store created successfully!"
    )

    return index


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    from upload_ingestion import (
        process_uploaded_files
    )

    print("\n===================================")
    print("USER VECTOR STORE TEST")
    print("===================================\n")

    documents_dir = Path(
        "data/documents"
    )

    files = list(
        documents_dir.glob("*.pdf")
    )

    if not files:

        print(
            "No documents found."
        )

    else:

        # Extract and chunk
        chunks = process_uploaded_files(
            files
        )

        # Build vector store
        create_vector_store(
            chunks
        )