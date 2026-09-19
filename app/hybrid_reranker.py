from hybrid_retriever import HybridRetriever
from reranker import Reranker


class HybridReranker:
    """
    Two-stage retrieval pipeline:

    Stage 1:
        Hybrid retrieval using FAISS + BM25

    Stage 2:
        Cross-encoder reranking
    """

    def __init__(self):
        print("Initializing Hybrid + Reranking pipeline...")

        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

        print("Hybrid + Reranking pipeline ready.")

    def search(
        self,
        query,
        retrieval_k=10,
        top_k=5
    ):
        """
        Retrieve candidate documents using hybrid retrieval
        and then rerank them using the cross-encoder.
        """

        # --------------------------------------------------
        # Stage 1: Hybrid candidate retrieval
        # --------------------------------------------------

        candidates = self.hybrid_retriever.search(
            query,
            top_k=retrieval_k,
            retrieval_k=retrieval_k
        )

        if not candidates:
            return []

        # --------------------------------------------------
        # Stage 2: Cross-encoder reranking
        # --------------------------------------------------

        reranked_results = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        return reranked_results


def run_tests():

    print("=" * 70)
    print("TESTING HYBRID RETRIEVAL + RERANKING")
    print("=" * 70)

    pipeline = HybridReranker()

    test_queries = [
        "How many work from home days are allowed?",
        "What are the password requirements?",
        "How much internet reimbursement can employees receive?",
        "How many annual leave days do employees get?",
        "What is policy HR-WFH-008?"
    ]

    for query in test_queries:

        print("\n")
        print("QUERY:", query)
        print("=" * 70)

        results = pipeline.search(
            query,
            retrieval_k=10,
            top_k=3
        )

        print("\nFINAL RERANKED RESULTS")
        print("-" * 70)

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")

            print(
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
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
                f"{result['text'][:400]}..."
            )


if __name__ == "__main__":
    run_tests()