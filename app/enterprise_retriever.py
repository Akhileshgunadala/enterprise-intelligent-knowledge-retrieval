from multi_query_retriever import MultiQueryRetriever
from reranker import Reranker


class EnterpriseRetriever:

    def __init__(self):
        print("=" * 70)
        print("INITIALIZING ENTERPRISE RETRIEVER")
        print("=" * 70)

        self.multi_query = MultiQueryRetriever()
        self.reranker = Reranker()

        print("\nEnterprise Retriever ready.")

    def search(self, query, retrieval_k=10, top_k=3):

        # =========================================================
        # STEP 1: MULTI-QUERY RETRIEVAL
        # =========================================================

        candidates = self.multi_query.search(
            query,
            top_k=retrieval_k
        )

        # =========================================================
        # STEP 2: CROSS-ENCODER RERANKING
        # =========================================================

        print("\n" + "=" * 70)
        print("CROSS-ENCODER RERANKING")
        print("=" * 70)

        reranked_results = self.reranker.rerank(
            query,
            candidates,
            top_k=retrieval_k
        )

        # =========================================================
        # STEP 3: RELEVANCE FILTERING
        # =========================================================

        if not reranked_results:
            return []

        best_score = reranked_results[0]["reranker_score"]

        # Keep documents reasonably close to the best result.
        score_margin = 3.0

        filtered_results = [
            result
            for result in reranked_results
            if result["reranker_score"] >= best_score - score_margin
        ]

        # =========================================================
        # STEP 4: REMOVE DUPLICATE SOURCES
        # =========================================================

        unique_results = []
        seen_sources = set()

        for result in filtered_results:

            source_key = (
                result["document"],
                result["page"]
            )

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_results.append(result)

        # =========================================================
        # STEP 5: FINAL TOP-K RESULTS
        # =========================================================

        final_results = unique_results[:top_k]

        # =========================================================
        # DISPLAY FINAL SOURCES
        # =========================================================

        print("\nFINAL RETRIEVED SOURCES")
        print("-" * 70)

        for rank, result in enumerate(final_results, start=1):

            print(
                f"{rank}. "
                f"{result['document']} "
                f"(Page {result['page']}) "
                f"| Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

        return final_results


if __name__ == "__main__":

    print("=" * 70)
    print("ENTERPRISE RETRIEVER TEST")
    print("=" * 70)

    retriever = EnterpriseRetriever()

    query = "How many work from home days are allowed?"

    results = retriever.search(
        query,
        retrieval_k=10,
        top_k=3
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )