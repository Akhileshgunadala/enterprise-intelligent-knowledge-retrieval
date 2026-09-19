import json
import sys
from pathlib import Path

import numpy as np


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

TEST_QUERIES_FILE = ROOT / "evaluation" / "test_queries.json"
RESULTS_FILE = ROOT / "evaluation" / "evaluation_results.json"

# Allow imports from app/
sys.path.insert(0, str(APP_DIR))


from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from multi_query_retriever import MultiQueryRetriever
from reranker import Reranker


# =========================================================
# LOAD TEST QUERIES
# =========================================================

with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
    test_queries = json.load(f)


print("=" * 80)
print("RAG RETRIEVAL EVALUATION")
print("=" * 80)

print(f"\nTest queries: {len(test_queries)}")


# =========================================================
# INITIALIZE RETRIEVERS
# =========================================================

print("\nInitializing BM25...")
bm25 = BM25Retriever()

print("\nInitializing Hybrid Retriever...")
hybrid = HybridRetriever()

print("\nInitializing Multi-Query Retriever...")
multi_query = MultiQueryRetriever()

print("\nInitializing Cross-Encoder Reranker...")
reranker = Reranker()


# =========================================================
# METRIC FUNCTIONS
# =========================================================

def is_relevant(result, query_info):
    """
    A result is relevant when both document and page
    match the ground-truth document/page.
    """

    return (
        result.get("document") == query_info["relevant_document"]
        and result.get("page") == query_info["relevant_page"]
    )


def calculate_metrics(results, query_info):
    """
    Calculate Top-1, Recall@1/3/5 and Reciprocal Rank
    for one query.
    """

    relevance = [
        is_relevant(result, query_info)
        for result in results
    ]

    top1 = 1 if relevance[0] else 0

    recall1 = 1 if any(relevance[:1]) else 0
    recall3 = 1 if any(relevance[:3]) else 0
    recall5 = 1 if any(relevance[:5]) else 0

    reciprocal_rank = 0.0

    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "top1": top1,
        "recall1": recall1,
        "recall3": recall3,
        "recall5": recall5,
        "mrr": reciprocal_rank,
    }


# =========================================================
# EVALUATION STORAGE
# =========================================================

methods = {
    "BM25": [],
    "Semantic / Vector": [],
    "Hybrid": [],
    "Multi-Query": [],
    "Multi-Query + Reranking": [],
}


# =========================================================
# RUN BENCHMARK
# =========================================================

for query_number, query_info in enumerate(test_queries, start=1):

    query = query_info["query"]

    print("\n" + "-" * 80)
    print(f"QUERY {query_number}/{len(test_queries)}")
    print(f"Question: {query}")

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_results = bm25.search(
        query,
        top_k=5
    )

    methods["BM25"].append(
        calculate_metrics(
            bm25_results,
            query_info
        )
    )

    # -----------------------------------------------------
    # Semantic / Vector
    # -----------------------------------------------------

    semantic_results = hybrid.semantic_search(
        query,
        top_k=5
    )

    # Convert FAISS results into normal document records
    semantic_full_results = []

    for result in semantic_results:

        index = result["index"]

        chunk = hybrid.chunks[index].copy()

        chunk["semantic_score"] = result["semantic_score"]

        semantic_full_results.append(chunk)

    methods["Semantic / Vector"].append(
        calculate_metrics(
            semantic_full_results,
            query_info
        )
    )

    # -----------------------------------------------------
    # Hybrid
    # -----------------------------------------------------

    hybrid_results = hybrid.search(
        query,
        top_k=5,
        retrieval_k=10
    )

    methods["Hybrid"].append(
        calculate_metrics(
            hybrid_results,
            query_info
        )
    )

    # -----------------------------------------------------
    # Multi-Query
    # -----------------------------------------------------

    multi_results = multi_query.search(
        query,
        top_k=5
    )

    methods["Multi-Query"].append(
        calculate_metrics(
            multi_results,
            query_info
        )
    )

    # -----------------------------------------------------
    # Multi-Query + Reranking
    # -----------------------------------------------------

    reranked_results = reranker.rerank(
        query,
        multi_results,
        top_k=5
    )

    methods["Multi-Query + Reranking"].append(
        calculate_metrics(
            reranked_results,
            query_info
        )
    )


# =========================================================
# AGGREGATE RESULTS
# =========================================================

final_results = []

for method_name, metric_list in methods.items():

    top1 = np.mean([
        item["top1"]
        for item in metric_list
    ]) * 100

    recall1 = np.mean([
        item["recall1"]
        for item in metric_list
    ]) * 100

    recall3 = np.mean([
        item["recall3"]
        for item in metric_list
    ]) * 100

    recall5 = np.mean([
        item["recall5"]
        for item in metric_list
    ]) * 100

    mrr = np.mean([
        item["mrr"]
        for item in metric_list
    ])

    final_results.append({
        "Method": method_name,
        "Top-1 Accuracy": round(float(top1), 3),
        "Recall@1": round(float(recall1), 3),
        "Recall@3": round(float(recall3), 3),
        "Recall@5": round(float(recall5), 3),
        "MRR": round(float(mrr), 4),
    })


# =========================================================
# SAVE RESULTS
# =========================================================

output = {
    "query_count": len(test_queries),
    "results": final_results
}

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n")
print("=" * 80)
print("FINAL EVALUATION RESULTS")
print("=" * 80)

print(
    f"\n{'Method':<30}"
    f"{'Top-1':>10}"
    f"{'R@1':>10}"
    f"{'R@3':>10}"
    f"{'R@5':>10}"
    f"{'MRR':>10}"
)

print("-" * 80)

for result in final_results:

    print(
        f"{result['Method']:<30}"
        f"{result['Top-1 Accuracy']:>9.1f}%"
        f"{result['Recall@1']:>9.1f}%"
        f"{result['Recall@3']:>9.1f}%"
        f"{result['Recall@5']:>9.1f}%"
        f"{result['MRR']:>10.3f}"
    )

print("\n" + "=" * 80)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_FILE}"
)

print("\nEvaluation completed successfully.")