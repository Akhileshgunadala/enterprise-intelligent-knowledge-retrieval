import json
import re
import urllib.request
import urllib.error

from upload_hybrid_retriever import UploadHybridRetriever


class UploadMultiQueryRetriever:

    def __init__(self):
        self.hybrid = UploadHybridRetriever()

        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5:3b-instruct"

        print("Domain-independent Multi-Query Retriever initialized.")
        print(f"Local query generation model: {self.model}")

    # =====================================================
    # GENERATE RELATED QUERIES USING LOCAL QWEN
    # =====================================================

    def generate_queries(self, query):

        prompt = f"""
You are a search query transformation system.

The user will provide a question about an enterprise
knowledge base.

Generate exactly 3 alternative search queries.

IMPORTANT RULES:

1. Do NOT answer the question.
2. Preserve the exact meaning of the original question.
3. Do NOT introduce unrelated topics.
4. Do NOT assume facts that are not in the question.
5. Each query must be useful for searching documents.
6. Use different wording for each query.
7. Keep each query short and clear.
8. Return ONLY the 3 alternative queries.
9. Put each query on a separate line.
10. Do NOT use numbers, bullets, quotation marks, JSON,
    explanations, or extra text.

USER QUESTION:
{query}

Example:

USER QUESTION:
Who is eligible for remote work?

OUTPUT:
Who can work remotely?
What are the eligibility requirements for remote work?
Which employees are allowed to work from home?

Now generate exactly 3 alternative search queries.
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=120
            ) as response:

                response_data = json.loads(
                    response.read().decode("utf-8")
                )

            raw_response = response_data.get(
                "response",
                ""
            ).strip()

            # ---------------------------------------------
            # Parse one query per line
            # ---------------------------------------------

            lines = raw_response.splitlines()

            generated_queries = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # Remove common numbering/bullets
                line = re.sub(
                    r"^\s*(?:[-*•]|\d+[.)])\s*",
                    "",
                    line
                )

                # Remove surrounding quotes
                line = line.strip("\"'")

                if not line:
                    continue

                # Avoid duplicate original query
                if line.lower() == query.lower():
                    continue

                generated_queries.append(line)

            # Keep only the first 3 alternatives
            generated_queries = generated_queries[:3]

            # Always keep original query first
            queries = [query] + generated_queries

            print("\nGenerated Queries:")
            print("-" * 70)

            for i, q in enumerate(
                queries,
                start=1
            ):
                print(f"{i}. {q}")

            return queries

        except Exception as e:

            print(
                "\nLocal query generation failed."
            )

            print(
                f"Reason: {e}"
            )

            # Safe fallback
            return [query]

    # =====================================================
    # MULTI-QUERY SEARCH
    # =====================================================

    def search(
        self,
        query,
        retrieval_k=10,
        top_k=5
    ):

        queries = self.generate_queries(query)

        combined_results = {}

        # Search using every generated query
        for generated_query in queries:

            results = self.hybrid.search(
                generated_query,
                top_k=retrieval_k,
                retrieval_k=retrieval_k
            )

            for result in results:

                chunk_id = result["chunk_id"]

                if chunk_id not in combined_results:

                    combined_results[chunk_id] = result.copy()

                    combined_results[
                        chunk_id
                    ]["multi_query_score"] = (
                        result.get(
                            "hybrid_score",
                            0.0
                        )
                    )

                    combined_results[
                        chunk_id
                    ]["matched_queries"] = [
                        generated_query
                    ]

                else:

                    current_score = combined_results[
                        chunk_id
                    ]["multi_query_score"]

                    new_score = result.get(
                        "hybrid_score",
                        0.0
                    )

                    # Keep strongest score
                    if new_score > current_score:

                        combined_results[
                            chunk_id
                        ]["multi_query_score"] = new_score

                    if generated_query not in combined_results[
                        chunk_id
                    ]["matched_queries"]:

                        combined_results[
                            chunk_id
                        ]["matched_queries"].append(
                            generated_query
                        )

        # Sort by strongest retrieval score
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x[
                "multi_query_score"
            ],
            reverse=True
        )

        return final_results[:top_k]


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    import json

    retriever = UploadMultiQueryRetriever()

    query = (
        "What are the security requirements "
        "for employee accounts?"
    )

    results = retriever.search(
        query,
        top_k=5
    )

    print("\nFINAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['document']} "
            f"(Page {result['page']})"
        )

        print(
            f"Score: "
            f"{result['multi_query_score']:.4f}"
        )

        print(
            f"Matched queries: "
            f"{result['matched_queries']}"
        )

        print()