import json
from pathlib import Path
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/chunks.json")
INDEX_FILE = Path("vectorstore/faiss.index")
METADATA_FILE = Path("vectorstore/metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    """
    Tokenize text while preserving identifiers
    such as HR-WFH-008.
    """

    text = text.lower()

    return re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text
    )


class HybridRetriever:

    def __init__(self):

        # -----------------------------
        # Load chunks
        # -----------------------------

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # -----------------------------
        # BM25
        # -----------------------------

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

        # -----------------------------
        # FAISS
        # -----------------------------

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # -----------------------------
        # Embedding model
        # -----------------------------

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            f"Hybrid retriever loaded with "
            f"{len(self.chunks)} chunks."
        )

    def semantic_search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            results.append(
                {
                    "index": int(idx),
                    "semantic_score": float(score),
                }
            )

        return results

    def keyword_search(self, query, top_k=5):

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:

            results.append(
                {
                    "index": int(idx),
                    "bm25_score": float(scores[idx]),
                }
            )

        return results

    def search(
        self,
        query,
        top_k=5,
        retrieval_k=10,
    ):

        # --------------------------------
        # Get semantic results
        # --------------------------------

        semantic_results = self.semantic_search(
            query,
            retrieval_k
        )

        # --------------------------------
        # Get BM25 results
        # --------------------------------

        keyword_results = self.keyword_search(
            query,
            retrieval_k
        )

        # --------------------------------
        # Normalize BM25 scores
        # --------------------------------

        bm25_scores = [
            result["bm25_score"]
            for result in keyword_results
        ]

        max_bm25 = max(
            bm25_scores
        ) if bm25_scores else 1

        min_bm25 = min(
            bm25_scores
        ) if bm25_scores else 0

        # --------------------------------
        # Combine scores
        # --------------------------------

        combined = {}

        for result in semantic_results:

            idx = result["index"]

            combined[idx] = {
                "semantic_score":
                    result["semantic_score"],
                "bm25_score": 0.0
            }

        for result in keyword_results:

            idx = result["index"]

            if idx not in combined:

                combined[idx] = {
                    "semantic_score": 0.0,
                    "bm25_score": 0.0
                }

            # Min-max normalization
            if max_bm25 != min_bm25:

                normalized = (
                    result["bm25_score"] - min_bm25
                ) / (
                    max_bm25 - min_bm25
                )

            else:

                normalized = 0.0

            combined[idx][
                "bm25_score"
            ] = normalized

        # --------------------------------
        # Weighted fusion
        # --------------------------------

        results = []

        for idx, scores in combined.items():

            semantic_score = scores[
                "semantic_score"
            ]

            bm25_score = scores[
                "bm25_score"
            ]

            hybrid_score = (
                0.6 * semantic_score
                +
                0.4 * bm25_score
            )

            chunk = self.metadata[
                idx
            ].copy()

            chunk["semantic_score"] = (
                semantic_score
            )

            chunk["bm25_score"] = (
                bm25_score
            )

            chunk["hybrid_score"] = (
                hybrid_score
            )

            results.append(chunk)

        # --------------------------------
        # Sort by hybrid score
        # --------------------------------

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]


if __name__ == "__main__":

    print("\n===================================")
    print("TESTING HYBRID RETRIEVAL")
    print("===================================\n")

    retriever = HybridRetriever()

    test_queries = [
        "How many work from home days are allowed?",
        "What are the password requirements?",
        "How much internet reimbursement can employees receive?",
        "How many annual leave days do employees get?",
        "What is policy HR-WFH-008?",
    ]

    for query in test_queries:

        print(f"\nQUERY: {query}")
        print("-" * 70)

        results = retriever.search(
            query,
            top_k=3,
            retrieval_k=10,
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRank: {rank}"
            )

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"BM25 Score: "
                f"{result['bm25_score']:.4f}"
            )

            print(
                f"Document: "
                f"{result['document']}"
            )

            print(
                f"Page: "
                f"{result['page']}"
            )

            print(
                f"Text: "
                f"{result['text'][:250]}"
            )