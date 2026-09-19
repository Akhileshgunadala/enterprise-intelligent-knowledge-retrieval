import re
import json
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/chunks.json")

def tokenize(text):
    """
    Tokenize text while preserving policy IDs such as:
    HR-WFH-008
    IT-SEC-012
    FIN-EXP-006
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text
    )

    return tokens


class BM25Retriever:

    def __init__(self):

        # Load chunks
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Tokenize documents
        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

        print(
            f"BM25 index created with "
            f"{len(self.chunks)} chunks."
        )

    def search(self, query, top_k=5):

        # Tokenize query
        tokenized_query = tokenize(query)

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            tokenized_query
        )

        # Get highest scoring indices
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index].copy()

            chunk["bm25_score"] = float(
                scores[index]
            )

            results.append(chunk)

        return results


if __name__ == "__main__":

    print("\n===================================")
    print("TESTING BM25 RETRIEVAL")
    print("===================================\n")

    retriever = BM25Retriever()

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
            top_k=3
        )

        for result in results:

            print(
                f"\nBM25 Score: "
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