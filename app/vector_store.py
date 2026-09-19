from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

CHUNKS_FILE = Path("data/chunks.json")
VECTORSTORE_DIR = Path("vectorstore")

INDEX_FILE = VECTORSTORE_DIR / "faiss.index"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# -----------------------------
# Embedding model
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    """
    Load chunks created by the ingestion pipeline.
    """

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def create_vector_store():
    """
    Generate embeddings and create a FAISS index.
    """

    print("\n===================================")
    print("CREATING VECTOR STORE")
    print("===================================\n")

    # Load chunks
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # Load embedding model
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Extract text
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Convert to float32 for FAISS
    embeddings = embeddings.astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    # Create directory
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(INDEX_FILE))

    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nFAISS index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

    print("\nVector store created successfully!")


def search(query, top_k=5):
    """
    Search the FAISS index for relevant chunks.
    """

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Load FAISS index
    index = faiss.read_index(str(INDEX_FILE))

    # Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Create query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Search
    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        chunk = metadata[index_position].copy()

        chunk["score"] = float(score)

        results.append(chunk)

    return results


if __name__ == "__main__":

    create_vector_store()

    print("\n===================================")
    print("TESTING SEMANTIC SEARCH")
    print("===================================\n")

    test_queries = [
        "How many work from home days are allowed?",
        "What are the password requirements?",
        "How much internet reimbursement can employees receive?",
        "How many annual leave days do employees get?",
    ]

    for query in test_queries:

        print(f"\nQUERY: {query}")
        print("-" * 70)

        results = search(query, top_k=3)

        for result in results:

            print(
                f"\nScore: {result['score']:.4f}"
            )

            print(
                f"Document: {result['document']}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Text: {result['text'][:250]}"
            )