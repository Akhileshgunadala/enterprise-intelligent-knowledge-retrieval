from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder based document reranker.

    The reranker receives a user query and candidate documents,
    then directly evaluates the relevance of each
    query-document pair.
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")

        self.model = CrossEncoder(model_name)

        print("Reranker model loaded successfully.")

    def rerank(self, query, results, top_k=5):
        """
        Rerank retrieved documents.

        Parameters:
            query: Original user question
            results: Candidate retrieval results
            top_k: Number of final results

        Returns:
            Reranked list of results
        """

        if not results:
            return []

        # Create query-document pairs
        pairs = []

        for result in results:
            pairs.append(
                [
                    query,
                    result["text"]
                ]
            )

        # Cross-encoder relevance scores
        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):

            updated_result = result.copy()

            updated_result["reranker_score"] = float(score)

            reranked_results.append(updated_result)

        # Highest relevance first
        reranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]


if __name__ == "__main__":

    print("=" * 70)
    print("RERANKER MODULE TEST")
    print("=" * 70)

    # Example candidates
    sample_results = [
        {
            "document": "work_from_home_policy.pdf",
            "page": 1,
            "text": (
                "Employees may work from home for up to "
                "8 days per calendar month."
            )
        },
        {
            "document": "employee_handbook.pdf",
            "page": 1,
            "text": (
                "Employees are expected to maintain "
                "professional attendance and working hours."
            )
        },
        {
            "document": "leave_policy.pdf",
            "page": 1,
            "text": (
                "Eligible employees receive 24 days "
                "of annual paid leave."
            )
        }
    ]

    query = "How many work from home days are allowed?"

    reranker = Reranker()

    results = reranker.rerank(
        query,
        sample_results,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nReranked Results:")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
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
            f"{result['text']}"
        )