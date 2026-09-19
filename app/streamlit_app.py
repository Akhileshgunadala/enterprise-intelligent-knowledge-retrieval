import sys
import shutil
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"
USER_INDEX_DIR = PROJECT_ROOT / "user_index"

APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from upload_ingestion import process_uploaded_files
from upload_vector_store import create_vector_store
from upload_rag_pipeline import UploadRAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       THEME-AWARE COLORS
       ====================================================== */

    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.72;
        margin-bottom: 1.5rem;
    }


    /* ======================================================
       DASHBOARD CARDS
       ====================================================== */

    .metric-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    .metric-title {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.65;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
        color: var(--text-color);
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        color: var(--text-color);
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }


    /* ======================================================
       PIPELINE
       ====================================================== */

    .pipeline-box {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 20px;
    }

    .pipeline-step {
        padding: 7px 0;
        color: var(--text-color);
        font-size: 0.95rem;
    }


    /* ======================================================
       ANSWER
       ====================================================== */

    .answer-box {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 5px solid #6366f1;
        border-radius: 14px;
        padding: 24px;
        margin-top: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
    }


    /* ======================================================
       SOURCE
       ====================================================== */

    .source-box {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }


    /* ======================================================
       GENERAL TEXT
       ====================================================== */

    .stMarkdown,
    .stText,
    label,
    p,
    span {
        color: var(--text-color);
    }


    /* ======================================================
       TEXT INPUT
       ====================================================== */

    input,
    textarea {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        caret-color: var(--text-color) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--text-color) !important;
        opacity: 0.55 !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        caret-color: var(--text-color) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-color) !important;
        opacity: 0.55 !important;
    }


    /* ======================================================
       CODE / EXAMPLE QUESTIONS
       ====================================================== */

    code {
        color: var(--text-color) !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-color: rgba(128, 128, 128, 0.25);
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: var(--secondary-background-color);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-color);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: var(--text-color);
        opacity: 0.55;
        font-size: 0.85rem;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = []

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "build_success" not in st.session_state:
    st.session_state.build_success = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:1.45rem;font-weight:800;">
        📚 Knowledge Base
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Build a private document knowledge base "
        "and ask questions using RAG."
    )

    st.divider()


    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select enterprise documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX and TXT",
    )


    # --------------------------------------------------------
    # Knowledge base status
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Knowledge Base")

    if st.session_state.chunks_count > 0:

        st.success("● Knowledge Base Ready")

        st.metric(
            "Documents",
            len(st.session_state.documents_indexed),
        )

        st.metric(
            "Text Chunks",
            st.session_state.chunks_count,
        )

    else:

        st.warning("● No Knowledge Base")

        st.caption(
            "Upload documents and build the "
            "knowledge base."
        )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.divider()

    st.subheader("⚙️ Retrieval Pipeline")

    pipeline_steps = [
        "✓ Document extraction",
        "✓ Text chunking",
        "✓ BM25 keyword retrieval",
        "✓ FAISS semantic retrieval",
        "✓ Hybrid retrieval",
        "✓ Multi-query retrieval",
        "✓ Cross-encoder reranking",
        "✓ Local Qwen generation",
    ]

    for step in pipeline_steps:

        st.markdown(
            f'<div class="pipeline-step">{step}</div>',
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # Clear button
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base",
        width="stretch",
    ):

        st.session_state.pipeline = None
        st.session_state.documents_indexed = []
        st.session_state.chunks_count = 0
        st.session_state.chat_history = []
        st.session_state.build_success = False

        if USER_INDEX_DIR.exists():

            try:
                shutil.rmtree(USER_INDEX_DIR)

            except Exception:
                pass

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 Enterprise Intelligent Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Hybrid RAG • Multi-Query Retrieval • Cross-Encoder Reranking • Local LLM
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Documents</div>
            <div class="metric-value">
                {len(st.session_state.documents_indexed)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Indexed Chunks</div>
            <div class="metric-value">
                {st.session_state.chunks_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    status_text = (
        "Ready"
        if st.session_state.pipeline is not None
        else "Waiting"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">System Status</div>
            <div class="metric-value">
                🟢 {status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">LLM</div>
            <div class="metric-value">
                Qwen 2.5
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DOCUMENT UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📁 Build Knowledge Base</div>',
    unsafe_allow_html=True,
)


if uploaded_files:

    st.info(
        f"📄 {len(uploaded_files)} document(s) selected."
    )


    with st.expander(
        "View selected documents",
        expanded=False,
    ):

        for file in uploaded_files:

            st.write(
                f"📄 **{file.name}** "
                f"({file.size / 1024:.1f} KB)"
            )


    if st.button(
        "⚡ Build Knowledge Base",
        type="primary",
        width="stretch",
    ):

        try:

            # ------------------------------------------------
            # Create upload directory
            # ------------------------------------------------

            UPLOAD_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved_paths = []


            # ------------------------------------------------
            # Processing status
            # ------------------------------------------------

            with st.status(
                "Building knowledge base...",
                expanded=True,
            ) as status:


                # --------------------------------------------
                # Save documents
                # --------------------------------------------

                st.write(
                    "📥 Saving uploaded documents..."
                )
                # Remove previously uploaded documents
                for old_file in UPLOAD_DIR.iterdir():
                    if old_file.is_file():
                        old_file.unlink()
                        
                for uploaded_file in uploaded_files:

                    file_path = (
                        UPLOAD_DIR
                        / uploaded_file.name
                    )

                    with open(
                        file_path,
                        "wb",
                    ) as output_file:

                        output_file.write(
                            uploaded_file.getbuffer()
                        )

                    saved_paths.append(
                        file_path
                    )


                st.write(
                    f"Saved {len(saved_paths)} document(s)."
                )


                # --------------------------------------------
                # Extraction
                # --------------------------------------------

                st.write(
                    "📖 Extracting document text..."
                )

                chunks = process_uploaded_files(
                    saved_paths
                )


                if not chunks:

                    raise ValueError(
                        "No readable text was found "
                        "in the uploaded documents."
                    )


                st.write(
                    f"🧩 Created {len(chunks)} text chunks."
                )


                # --------------------------------------------
                # Vector store
                # --------------------------------------------

                st.write(
                    "🧠 Generating embeddings..."
                )

                create_vector_store(
                    chunks
                )


                st.write(
                    "⚡ FAISS vector index created."
                )


                # --------------------------------------------
                # RAG pipeline
                # --------------------------------------------

                st.write(
                    "🔄 Initializing retrieval pipeline..."
                )

                pipeline = UploadRAGPipeline()


                # --------------------------------------------
                # Update session state
                # --------------------------------------------

                st.session_state.pipeline = pipeline

                st.session_state.documents_indexed = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.chunks_count = len(
                    chunks
                )

                st.session_state.chat_history = []

                st.session_state.build_success = True


                status.update(
                    label="Knowledge base ready!",
                    state="complete",
                    expanded=False,
                )


            # ------------------------------------------------
            # Refresh dashboard immediately
            # ------------------------------------------------

            st.rerun()


        except Exception as e:

            st.error(
                "❌ Failed to build the knowledge base."
            )

            st.exception(e)


else:

    st.info(
        "Upload PDF, DOCX or TXT documents from "
        "the sidebar to create your knowledge base."
    )


# ============================================================
# BUILD SUCCESS MESSAGE
# ============================================================

if st.session_state.build_success:

    st.success(
        "✅ Knowledge base built successfully!"
    )

    st.session_state.build_success = False


# ============================================================
# QUESTION / CHAT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask Your Documents</div>',
    unsafe_allow_html=True,
)


if st.session_state.pipeline is None:

    st.info(
        "🔒 The assistant is waiting for a knowledge base. "
        "Upload documents and click **Build Knowledge Base**."
    )

else:

    # --------------------------------------------------------
    # Example questions
    # --------------------------------------------------------

    st.caption("Try asking")

    example_col1, example_col2, example_col3 = st.columns(3)

    examples = [
        "What is the eligibility criteria for WFH?",
        "How many annual leave days do employees receive?",
        "What are the password requirements?",
    ]


    with example_col1:

        st.code(
            examples[0],
            language=None,
        )


    with example_col2:

        st.code(
            examples[1],
            language=None,
        )


    with example_col3:

        st.code(
            examples[2],
            language=None,
        )


    # ========================================================
    # DISPLAY PREVIOUS CHAT HISTORY
    # ========================================================

    if st.session_state.chat_history:

        for conversation in st.session_state.chat_history:

            # ------------------------------------------------
            # User message
            # ------------------------------------------------

            with st.chat_message("user"):

                st.markdown(
                    conversation["question"]
                )


            # ------------------------------------------------
            # Assistant message
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.markdown(
                    "### 💡 Answer"
                )

                st.markdown(
                    conversation["answer"]
                )


                # ------------------------------------------------
                # Sources
                # ------------------------------------------------

                sources = conversation.get(
                    "sources",
                    []
                )


                if sources:

                    st.markdown(
                        "### 📚 Sources"
                    )


                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        document = source.get(
                            "document",
                            "Unknown document",
                        )

                        page = source.get(
                            "page",
                            "Unknown",
                        )

                        reranker_score = source.get(
                            "reranker_score",
                            None,
                        )


                        with st.expander(
                            f"📄 {index}. {document} — Page {page}"
                        ):

                            st.markdown(
                                source.get(
                                    "text",
                                    "No source text available.",
                                )
                            )


                            st.markdown("---")


                            st.caption(
                                "Retrieval Details"
                            )


                            score_col1, score_col2, score_col3 = (
                                st.columns(3)
                            )


                            with score_col1:

                                semantic_score = source.get(
                                    "semantic_score"
                                )

                                if semantic_score is not None:

                                    st.metric(
                                        "Semantic",
                                        f"{semantic_score:.4f}",
                                    )


                            with score_col2:

                                bm25_score = source.get(
                                    "bm25_score"
                                )

                                if bm25_score is not None:

                                    st.metric(
                                        "BM25",
                                        f"{bm25_score:.4f}",
                                    )


                            with score_col3:

                                if reranker_score is not None:

                                    st.metric(
                                        "Reranker",
                                        f"{reranker_score:.4f}",
                                    )


                else:

                    st.info(
                        "No relevant sources were found "
                        "in the knowledge base."
                    )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your documents..."
    )


    # ========================================================
    # PROCESS NEW QUESTION
    # ========================================================

    if question:

        question = question.strip()


        if question:

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # ------------------------------------------------
            # ASSISTANT MESSAGE
            # ------------------------------------------------

            with st.chat_message("assistant"):

                status_placeholder = st.empty()

                answer_placeholder = st.empty()


                try:

                    # ----------------------------------------
                    # STEP 1 — RETRIEVAL
                    # ----------------------------------------

                    status_placeholder.info(
                        "🔎 Searching your knowledge base..."
                    )


                    # ----------------------------------------
                    # STEP 2 — STREAM QWEN RESPONSE
                    # ----------------------------------------

                    status_placeholder.info(
                        "🧠 Generating a grounded answer..."
                    )


                    streamed_answer = ""


                    for event in (
                        st.session_state
                        .pipeline
                        .ask_stream(
                            question,
                            retrieval_k=10,
                            top_k=3,
                        )
                    ):


                        # ------------------------------------
                        # TOKEN
                        # ------------------------------------

                        if event["type"] == "token":

                            token = event["content"]

                            streamed_answer += token

                            answer_placeholder.markdown(
                                streamed_answer
                                + "▌"
                            )


                        # ------------------------------------
                        # COMPLETE
                        # ------------------------------------

                        elif event["type"] == "complete":

                            final_answer = event[
                                "answer"
                            ]

                            sources = event[
                                "sources"
                            ]

                            streamed_answer = final_answer

                            answer_placeholder.markdown(
                                streamed_answer
                            )


                    # ----------------------------------------
                    # REMOVE STATUS
                    # ----------------------------------------

                    status_placeholder.empty()


                    # ----------------------------------------
                    # DISPLAY SOURCES
                    # ----------------------------------------

                    if sources:

                        st.markdown(
                            "### 📚 Sources"
                        )


                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            document = source.get(
                                "document",
                                "Unknown document",
                            )

                            page = source.get(
                                "page",
                                "Unknown",
                            )

                            reranker_score = source.get(
                                "reranker_score",
                                None,
                            )


                            with st.expander(
                                f"📄 {index}. {document} — Page {page}"
                            ):

                                st.markdown(
                                    source.get(
                                        "text",
                                        "No source text available.",
                                    )
                                )


                                st.markdown("---")


                                st.caption(
                                    "Retrieval Details"
                                )


                                score_col1, score_col2, score_col3 = (
                                    st.columns(3)
                                )


                                with score_col1:

                                    semantic_score = source.get(
                                        "semantic_score"
                                    )

                                    if semantic_score is not None:

                                        st.metric(
                                            "Semantic",
                                            f"{semantic_score:.4f}",
                                        )


                                with score_col2:

                                    bm25_score = source.get(
                                        "bm25_score"
                                    )

                                    if bm25_score is not None:

                                        st.metric(
                                            "BM25",
                                            f"{bm25_score:.4f}",
                                        )


                                with score_col3:

                                    if reranker_score is not None:

                                        st.metric(
                                            "Reranker",
                                            f"{reranker_score:.4f}",
                                        )


                    else:

                        st.info(
                            "No relevant sources were found "
                            "in the knowledge base."
                        )


                    # ----------------------------------------
                    # SAVE CHAT HISTORY
                    # ----------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": streamed_answer,
                            "sources": sources,
                        }
                    )


                except Exception as e:

                    status_placeholder.empty()

                    st.error(
                        "❌ An error occurred while "
                        "processing your question."
                    )

                    st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
    Enterprise Intelligent Knowledge Retrieval System
    <br>
    Hybrid RAG • BM25 • FAISS • Multi-Query Retrieval
    • Cross-Encoder Reranking • Local Qwen
    </div>
    """,
    unsafe_allow_html=True,
)