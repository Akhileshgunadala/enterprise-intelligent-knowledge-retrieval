from upload_multi_query_retriever import UploadMultiQueryRetriever
from reranker import Reranker


class UploadEnterpriseRetriever:

    def __init__(self):

        print("\n===================================")
        print("INITIALIZING UPLOAD ENTERPRISE RETRIEVER")
        print("===================================\n")

        self.multi_query = UploadMultiQueryRetriever()

        self.reranker = Reranker()

        print("\nUpload Enterprise Retriever ready.")


    def search(
        self,
        query,
        retrieval_k=10,
        top_k=3,
    ):

        # ------------------------------------------
        # Multi-Query Retrieval
        # ------------------------------------------

        candidates = self.multi_query.search(
            query,
            top_k=retrieval_k,
            retrieval_k=retrieval_k,
        )

        print("\n===================================")
        print("CROSS-ENCODER RERANKING")
        print("===================================\n")

        # ------------------------------------------
        # Cross-Encoder Reranking
        # ------------------------------------------

        reranked_results = self.reranker.rerank(
            query,
            candidates,
            top_k=retrieval_k,
        )

        if not reranked_results:

            return []


        # ------------------------------------------
        # Remove very low-relevance results
        # ------------------------------------------

        best_score = (
            reranked_results[0]["reranker_score"]
        )

        score_margin = 3.0

        filtered_results = [
            result
            for result in reranked_results
            if result["reranker_score"]
            >= best_score - score_margin
        ]


        # ------------------------------------------
        # Remove duplicate document/page sources
        # ------------------------------------------

        unique_results = []

        seen_sources = set()

        for result in filtered_results:

            source_key = (
                result["document"],
                result["page"],
            )

            if source_key not in seen_sources:

                seen_sources.add(
                    source_key
                )

                unique_results.append(
                    result
                )


        # ------------------------------------------
        # Final Top-K
        # ------------------------------------------

        final_results = unique_results[
            :top_k
        ]


        # ------------------------------------------
        # Display Sources
        # ------------------------------------------

        print(
            "\nFINAL RETRIEVED SOURCES"
        )

        print(
            "-" * 70
        )

        for rank, result in enumerate(
            final_results,
            start=1,
        ):

            print(
                f"{rank}. "
                f"{result['document']} "
                f"(Page {result['page']}) "
                f"| Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )


        return final_results


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("TESTING UPLOAD ENTERPRISE RETRIEVER")
    print("===================================\n")

    retriever = UploadEnterpriseRetriever()


    test_queries = [
        "How many work from home days are allowed?",
        "What are the password requirements?",
        "How much internet reimbursement can employees receive?",
        "How many annual leave days do employees get?",
        "What is policy HR-WFH-008?",
    ]


    for query in test_queries:

        print(
            "\n\nQUERY:"
        )

        print(
            query
        )

        print(
            "=" * 70
        )


        results = retriever.search(
            query,
            retrieval_k=10,
            top_k=3,
        )


        print(
            "\nRESULT DETAILS"
        )

        print(
            "-" * 70
        )


        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nRank: {rank}"
            )

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
                f"{result['text'][:300]}"
            )