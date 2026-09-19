import json
import os
import sys

# Allow importing hybrid_retriever.py from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_retriever import HybridRetriever


class MultiQueryRetriever:
    def __init__(self):
        self.hybrid_retriever = HybridRetriever()

    def generate_queries(self, query):
        """
        Generate multiple search formulations for the same user question.

        In the first version, we use rule-based query expansion.
        Later, this can be replaced with an LLM-based query generator.
        """

        query_lower = query.lower()

        queries = [query]

        # Work-from-home related queries
        if "work from home" in query_lower or "remote work" in query_lower:
            queries.extend([
                "How many work from home days are allowed?",
                "What is the monthly remote work allowance?",
                "How many remote working days can an employee take?"
            ])

        # Password/security related queries
        elif "password" in query_lower:
            queries.extend([
                "What are the password requirements?",
                "What password rules must employees follow?",
                "What are the requirements for strong passwords?"
            ])

        # Reimbursement related queries
        elif "reimbursement" in query_lower:
            queries.extend([
                "What reimbursement amount can employees claim?",
                "What are the employee reimbursement limits?",
                "How much internet reimbursement is available?"
            ])

        # Leave related queries
        elif "leave" in query_lower:
            queries.extend([
                "How many annual leave days are available?",
                "What is the annual paid leave allowance?",
                "What are the employee leave entitlements?"
            ])

        # Policy ID queries
        elif "hr-wfh-008" in query_lower:
            queries.extend([
                "What is the Work From Home Policy?",
                "What does policy HR-WFH-008 cover?",
                "What are the rules for working from home?"
            ])

        # Generic expansion
        else:
            queries.extend([
                f"What does the company policy say about {query}?",
                f"What are the requirements related to {query}?",
                f"What rules apply to {query}?"
            ])

        # Remove duplicate queries while preserving order
        unique_queries = []

        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)

        return unique_queries

    def search(self, query, top_k=5):
        """
        Perform multi-query retrieval.

        Each generated query is sent through the existing
        Hybrid Retriever. Results are then combined and ranked.
        """

        generated_queries = self.generate_queries(query)

        print("\nGenerated Queries:")
        print("-" * 70)

        for i, q in enumerate(generated_queries, start=1):
            print(f"{i}. {q}")

        all_results = {}

        # Retrieve results for every generated query
        for generated_query in generated_queries:

            results = self.hybrid_retriever.search(
                generated_query,
                top_k=top_k,
                retrieval_k=10
            )

            for result in results:

                chunk_id = result["chunk_id"]

                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "chunk_id": result["chunk_id"],
                        "document": result["document"],
                        "page": result["page"],
                        "text": result["text"],
                        "scores": [],
                        "matched_queries": []
                    }

                all_results[chunk_id]["scores"].append(
                    result["hybrid_score"]
                )

                all_results[chunk_id]["matched_queries"].append(
                    generated_query
                )

        # Calculate final multi-query score
        #
        # We use the maximum score obtained across queries.
        # This prevents a relevant chunk from being penalized
        # simply because another query formulation matched poorly.

        final_results = []

        for chunk in all_results.values():

            chunk["multi_query_score"] = max(chunk["scores"])

            final_results.append(chunk)

        # Sort highest score first
        final_results.sort(
            key=lambda x: x["multi_query_score"],
            reverse=True
        )

        return final_results[:top_k]


def run_tests():

    print("=" * 70)
    print("TESTING MULTI-QUERY RETRIEVAL")
    print("=" * 70)

    retriever = MultiQueryRetriever()

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
        print("-" * 70)

        results = retriever.search(query, top_k=3)

        print("\nFinal Multi-Query Results:")
        print("-" * 70)

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(
                f"Multi-Query Score: "
                f"{result['multi_query_score']:.4f}"
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
                f"Matched by "
                f"{len(result['matched_queries'])} query(s)"
            )

            print("Matched Queries:")

            for matched_query in result["matched_queries"]:
                print(f"  - {matched_query}")

            print(
                f"Text: "
                f"{result['text'][:350]}..."
            )


if __name__ == "__main__":
    run_tests()