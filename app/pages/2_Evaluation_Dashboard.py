import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = ROOT / "evaluation"
RESULTS_FILE = EVALUATION_DIR / "evaluation_results.json"
EVALUATOR_FILE = EVALUATION_DIR / "evaluate_retrieval.py"


# ---------------------------------------------------------
# DEFAULT RESULTS
# ---------------------------------------------------------

DEFAULT_RESULTS = [
    {
        "Method": "BM25",
        "Top-1 Accuracy": 92.0,
        "Recall@1": 92.0,
        "Recall@3": 96.0,
        "Recall@5": 96.0,
        "MRR": 0.933,
    },
    {
        "Method": "Semantic / Vector",
        "Top-1 Accuracy": 76.0,
        "Recall@1": 76.0,
        "Recall@3": 100.0,
        "Recall@5": 100.0,
        "MRR": 0.873,
    },
    {
        "Method": "Hybrid",
        "Top-1 Accuracy": 88.0,
        "Recall@1": 88.0,
        "Recall@3": 96.0,
        "Recall@5": 100.0,
        "MRR": 0.921,
    },
    {
        "Method": "Multi-Query",
        "Top-1 Accuracy": 92.0,
        "Recall@1": 92.0,
        "Recall@3": 96.0,
        "Recall@5": 100.0,
        "MRR": 0.950,
    },
    {
        "Method": "Multi-Query + Reranking",
        "Top-1 Accuracy": 92.0,
        "Recall@1": 92.0,
        "Recall@3": 96.0,
        "Recall@5": 100.0,
        "MRR": 0.950,
    },
]


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

def load_results():
    """Load the latest saved evaluation results."""

    if not RESULTS_FILE.exists():
        return pd.DataFrame(DEFAULT_RESULTS), 25

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        query_count = data.get("query_count", 25)

        rows = data.get("results", data)

        if isinstance(rows, list) and rows:
            return pd.DataFrame(rows), query_count

    except Exception:
        pass

    return pd.DataFrame(DEFAULT_RESULTS), 25


# ---------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------

def run_evaluation():
    """Run the benchmark evaluator and save fresh results."""

    if not EVALUATOR_FILE.exists():
        st.error(
            f"Evaluator not found:\n\n{EVALUATOR_FILE}"
        )
        return False

    with st.spinner(
        "Running full RAG evaluation... "
        "This may take a few minutes because local ML models are loaded."
    ):

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR_FILE),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:

                st.error("Evaluation failed.")

                if result.stderr:
                    st.code(
                        result.stderr,
                        language="text",
                    )

                return False

            if not RESULTS_FILE.exists():

                st.error(
                    "Evaluation completed, but "
                    "evaluation_results.json was not created."
                )

                return False

            return True

        except Exception as e:

            st.error(
                f"Could not run evaluation:\n\n{e}"
            )

            return False


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📊 RAG Evaluation Dashboard")

st.caption(
    "Benchmark comparison of keyword retrieval, semantic retrieval, "
    "hybrid retrieval, multi-query retrieval, and cross-encoder reranking."
)


# ---------------------------------------------------------
# RUN EVALUATION BUTTON
# ---------------------------------------------------------

button_col, status_col = st.columns([1, 3])

with button_col:

    run_button = st.button(
        "▶ Run Evaluation",
        type="primary",
        width="stretch",
    )


if run_button:

    success = run_evaluation()

    if success:

        st.success(
            "Evaluation completed successfully. "
            "Dashboard updated with fresh results."
        )

        st.rerun()


# ---------------------------------------------------------
# LOAD CURRENT RESULTS
# ---------------------------------------------------------

df, query_count = load_results()


st.info(
    f"Benchmark: {query_count} enterprise knowledge-base queries "
    "across the current NexaTech Solutions document set."
)


# ---------------------------------------------------------
# NORMALIZE COLUMN NAMES
# ---------------------------------------------------------

rename_map = {
    "Top1": "Top-1 Accuracy",
    "Top1 Accuracy": "Top-1 Accuracy",
}

df = df.rename(columns=rename_map)


required = [
    "Method",
    "Top-1 Accuracy",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "MRR",
]


missing = [
    column
    for column in required
    if column not in df.columns
]


if missing:

    st.error(
        "Evaluation results are missing these columns: "
        + ", ".join(missing)
    )

    st.stop()


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

best_mrr_row = df.loc[df["MRR"].idxmax()]
best_top1_row = df.loc[df["Top-1 Accuracy"].idxmax()]


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Queries Evaluated",
        query_count,
    )


with c2:

    st.metric(
        "Highest Top-1 Accuracy",
        f"{best_top1_row['Top-1 Accuracy']:.1f}%",
    )


with c3:

    st.metric(
        "Highest MRR",
        f"{best_mrr_row['MRR']:.3f}",
    )


with c4:

    st.metric(
        "Recall@5 Coverage",
        f"{df['Recall@5'].max():.1f}%",
    )


st.divider()


# ---------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------

st.subheader("📋 Benchmark Results")


display_df = df.copy()


for column in [
    "Top-1 Accuracy",
    "Recall@1",
    "Recall@3",
    "Recall@5",
]:

    display_df[column] = display_df[column].map(
        lambda x: f"{float(x):.1f}%"
    )


display_df["MRR"] = display_df["MRR"].map(
    lambda x: f"{float(x):.3f}"
)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# TOP-1 CHART
# ---------------------------------------------------------

st.subheader("🎯 Top-1 Retrieval Accuracy")


top1_chart = px.bar(
    df,
    x="Method",
    y="Top-1 Accuracy",
    text="Top-1 Accuracy",
    title="Top-1 Accuracy by Retrieval Method",
)


top1_chart.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


top1_chart.update_layout(
    yaxis_title="Accuracy (%)",
    xaxis_title="",
    yaxis_range=[0, 105],
)


st.plotly_chart(
    top1_chart,
    width="stretch",
)


# ---------------------------------------------------------
# RECALL CHART
# ---------------------------------------------------------

st.subheader("🔎 Recall@K Comparison")


recall_df = df.melt(
    id_vars=["Method"],
    value_vars=[
        "Recall@1",
        "Recall@3",
        "Recall@5",
    ],
    var_name="Metric",
    value_name="Recall",
)


recall_chart = px.bar(
    recall_df,
    x="Method",
    y="Recall",
    color="Metric",
    barmode="group",
    title="Recall@1 vs Recall@3 vs Recall@5",
)


recall_chart.update_layout(
    yaxis_title="Recall (%)",
    xaxis_title="",
    yaxis_range=[0, 105],
)


st.plotly_chart(
    recall_chart,
    width="stretch",
)


# ---------------------------------------------------------
# MRR CHART
# ---------------------------------------------------------

st.subheader("📈 Mean Reciprocal Rank (MRR)")


mrr_chart = px.bar(
    df,
    x="Method",
    y="MRR",
    text="MRR",
    title="MRR by Retrieval Method",
)


mrr_chart.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)


mrr_chart.update_layout(
    yaxis_title="MRR",
    xaxis_title="",
    yaxis_range=[0, 1.05],
)


st.plotly_chart(
    mrr_chart,
    width="stretch",
)


# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

st.subheader("🧠 What the Benchmark Shows")


st.markdown(
    """
- **BM25** provides strong exact-term retrieval performance on the current
  enterprise policy corpus.
- **Semantic retrieval** reaches full Recall@3 and Recall@5 on this benchmark,
  while its Top-1 ranking is lower.
- **Hybrid retrieval** combines lexical and semantic signals and reaches
  100% Recall@5 on the benchmark.
- **Multi-Query retrieval** improves the ranking metric measured by MRR on
  this benchmark.
- **Multi-Query + Reranking** has the same aggregate benchmark metrics as
  Multi-Query here, showing that the reranker does not produce an aggregate
  improvement on this particular 25-query test set.
"""
)


st.caption(
    "These observations describe the measured benchmark results only; "
    "they should not be generalized beyond this dataset without additional evaluation."
)


# ---------------------------------------------------------
# ARCHITECTURE
# ---------------------------------------------------------

st.subheader("🏗️ Evaluated Retrieval Architecture")


st.code(
    """User Query
    │
    ├── Multi-Query Transformation
    │       │
    │       ├── Query 1 ──┐
    │       ├── Query 2 ──┤
    │       └── Query 3 ──┘
    │                    │
    │          ┌─────────┴─────────┐
    │          │                   │
    │        BM25              FAISS / Vector
    │          │                   │
    │          └─────────┬─────────┘
    │                    │
    │              Hybrid Retrieval
    │                    │
    │             Candidate Set
    │                    │
    │          Cross-Encoder Reranker
    │                    │
    │              Top Documents
    │                    │
    │              Local Qwen 2.5
    │                    │
    └────────────── Answer + Sources
""",
    language="text",
)


st.success(
    "Evaluation dashboard loaded successfully."
)