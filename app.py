"""Streamlit enterprise web interface inspired by IBM Carbon Design System."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from rag_langchain_chatbot.config import get_settings
from rag_langchain_chatbot.evaluation import RAGEvaluator
from rag_langchain_chatbot.loaders import get_loader_registry
from rag_langchain_chatbot.monitoring import get_metrics_tracker
from rag_langchain_chatbot.pipeline import HybridRAGPipeline
from rag_langchain_chatbot.ui.theme import apply_carbon_theme, render_carbon_header

logger = logging.getLogger(__name__)

# Streamlit Page Setup
st.set_page_config(
    page_title="Database-Driven Hybrid RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply IBM Carbon Theme
apply_carbon_theme()
render_carbon_header()


@st.cache_resource
def load_pipeline() -> HybridRAGPipeline:
    """Cache and return the Hybrid RAG pipeline singleton."""
    return HybridRAGPipeline()


settings = get_settings()
pipeline = load_pipeline()
metrics_tracker = get_metrics_tracker(settings)
registry = get_loader_registry()

# ==============================================================================
# Sidebar: System Telemetry & Hyperparameter Controls
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Telemetry")
    st.markdown(f"""
        **Embedding**: `{settings.embedding_model_name.split('/')[-1]}`  
        **Reranker**: `{settings.reranker_model_name.split('/')[-1]}`  
        **Dense DB**: `ChromaDB ({settings.collection_name})`  
        **Lexical DB**: `BM25Plus (Dynamic Corpus)`  
        **Security Guard**: `{'Active (Prompt Shield)' if settings.enable_prompt_injection_detection else 'Disabled'}`  
        **Tracing**: `{'LangSmith Active' if settings.langchain_api_key else 'Local SQLite'}`
    """)
    st.divider()

    st.markdown("### 🎛️ Pipeline Configuration")
    enable_rerank = st.toggle("Cross-Encoder Reranking", value=True, help="Enable deep cross-attention reranker")
    top_k = st.slider("Final Output Chunks (Top-K)", min_value=1, max_value=10, value=settings.reranker_top_k)
    fused_k = st.slider("Fused Candidate Pool", min_value=5, max_value=20, value=settings.fused_top_k)
    
    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# Main Navigation Tabs
# ==============================================================================
tab_chat, tab_docs, tab_dashboard, tab_eval = st.tabs([
    "💬 RAG Assistant",
    "📁 Database & Document Management",
    "📊 Observability & Cost Analytics",
    "🧪 LangSmith Evaluation"
])

# ------------------------------------------------------------------------------
# Tab 1: Interactive Chatbot & Source Inspection
# ------------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                meta = msg["metadata"]
                st.caption(
                    f"⚡ Latency: {meta.get('total_latency_ms', 0):.1f}ms "
                    f"(Retrieval: {meta.get('retrieval_latency_ms', 0):.1f}ms | Gen: {meta.get('generation_latency_ms', 0):.1f}ms) · "
                    f"🔤 Tokens: {meta.get('total_tokens', 0)} · "
                    f"💰 Est. Cost: ${meta.get('estimated_cost_usd', 0):.6f}"
                )

                if meta.get("security_risk") and meta.get("security_risk") != "LOW":
                    st.warning(f"⚠️ Security notice: {', '.join(meta.get('flagged_patterns', []))}")

                # Source Attribution Drawer
                chunks = meta.get("chunks", [])
                if chunks:
                    with st.expander(f"🔍 Inspect {len(chunks)} Retrieved Sources & Provenance"):
                        for c in chunks:
                            dense_info = f"Dense #{c.get('dense_rank') or 'N/A'}"
                            sparse_info = f"BM25 #{c.get('sparse_rank') or 'N/A'}"
                            rrf_info = f"RRF: {c.get('rrf_score') or 0.0:.4f}"
                            rerank_info = f"Rerank: {c.get('rerank_score') or c['score']:.4f}"

                            loc = f"p. {c['page'] + 1}" if c.get('page') is not None else (f"rec #{c.get('record_id')}" if c.get('record_id') else "N/A")
                            st.markdown(f"""
                                <div class="carbon-source-card">
                                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                        <span class="carbon-badge badge-blue">Rank {c['rank']}</span>
                                        <span class="carbon-badge badge-green">{rerank_info}</span>
                                        <span class="carbon-badge badge-gray">{dense_info}</span>
                                        <span class="carbon-badge badge-gray">{sparse_info}</span>
                                        <span class="carbon-badge badge-gray">{rrf_info}</span>
                                        <span class="carbon-badge badge-gray">{Path(c['source']).name} ({loc})</span>
                                    </div>
                                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #f4f4f4;">
                                        {c['content']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

    # User Input
    if user_query := st.chat_input("Ask any question against the connected database and documents..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Executing hybrid retrieval, reranking, and generation..."):
                response = pipeline.query(user_query, top_k=top_k, rerank=enable_rerank)
                st.markdown(response.answer)

                meta_info = {
                    "retrieval_latency_ms": response.retrieval_latency_ms,
                    "generation_latency_ms": response.generation_latency_ms,
                    "total_latency_ms": response.total_latency_ms,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "estimated_cost_usd": response.estimated_cost_usd,
                    "security_risk": response.security_risk,
                    "flagged_patterns": response.flagged_security_patterns,
                    "chunks": [
                        {
                            "rank": c.rank,
                            "score": c.score,
                            "source": c.source,
                            "page": c.page,
                            "record_id": c.record_id,
                            "chunk_id": c.chunk_id,
                            "content": c.content,
                            "dense_rank": c.dense_rank,
                            "dense_score": c.dense_score,
                            "sparse_rank": c.sparse_rank,
                            "sparse_score": c.sparse_score,
                            "rrf_score": c.rrf_score,
                            "rerank_score": c.rerank_score,
                        }
                        for c in response.chunks
                    ]
                }

                st.caption(
                    f"⚡ Latency: {response.total_latency_ms:.1f}ms "
                    f"(Retrieval: {response.retrieval_latency_ms:.1f}ms | Gen: {response.generation_latency_ms:.1f}ms) · "
                    f"🔤 Tokens: {response.total_tokens} · "
                    f"💰 Est. Cost: ${response.estimated_cost_usd:.6f}"
                )

                if response.security_risk != "LOW":
                    st.warning(f"⚠️ Security scan: {', '.join(response.flagged_security_patterns)}")

                if response.chunks:
                    with st.expander(f"🔍 Inspect {len(response.chunks)} Retrieved Sources & Provenance"):
                        for c in response.chunks:
                            dense_info = f"Dense #{c.dense_rank or 'N/A'}"
                            sparse_info = f"BM25 #{c.sparse_rank or 'N/A'}"
                            rrf_info = f"RRF: {c.rrf_score or 0.0:.4f}"
                            rerank_info = f"Rerank: {c.rerank_score or c.score:.4f}"
                            loc = f"p. {c.page + 1}" if c.page is not None else (f"rec #{c.record_id}" if c.record_id else "N/A")

                            st.markdown(f"""
                                <div class="carbon-source-card">
                                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                        <span class="carbon-badge badge-blue">Rank {c.rank}</span>
                                        <span class="carbon-badge badge-green">{rerank_info}</span>
                                        <span class="carbon-badge badge-gray">{dense_info}</span>
                                        <span class="carbon-badge badge-gray">{sparse_info}</span>
                                        <span class="carbon-badge badge-gray">{rrf_info}</span>
                                        <span class="carbon-badge badge-gray">{Path(c.source).name} ({loc})</span>
                                    </div>
                                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #f4f4f4;">
                                        {c.content}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": meta_info
                })

# ------------------------------------------------------------------------------
# Tab 2: Document Management & Multi-Format Ingestion
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("### 📁 Adaptive Ingestion & Incremental Synchronization")
    st.markdown("""
        The system dynamically synchronizes the vector database (**ChromaDB**) and lexical index (**BM25**) with the connected database and files.
        - **Supported formats**: PDF (`.pdf`), Word (`.docx`), Markdown/Text (`.md`, `.txt`), Tabular (`.csv`, `.json`), SQLite (`.sqlite3`, `.db`).
        - **Synchronization**: Detects content changes via SHA-256 hashes, indexes new items, updates changed sources, and prunes deleted records.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Document or Data File",
            type=["pdf", "docx", "txt", "md", "csv", "json"]
        )
        if uploaded_file is not None:
            save_path = settings.data_dir / uploaded_file.name
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded '{uploaded_file.name}' to `{settings.data_dir}`.")

            if st.button("🚀 Synchronize Database Now"):
                with st.spinner("Performing change detection, incremental chunking, and dual-index synchronization..."):
                    stats = pipeline.index(force_reindex=False)
                    st.success("Synchronization completed successfully!")
                    st.json(stats)

    with col2:
        st.markdown("#### ⚡ Maintenance Controls")
        if st.button("🔄 Force Rebuild Entire Collection", use_container_width=True):
            with st.spinner("Purging collections and rebuilding from all discovered sources..."):
                stats = pipeline.index(force_reindex=True)
                st.success("Entire collection rebuilt from ground up!")
                st.json(stats)

    st.divider()
    st.markdown("#### 📑 Discovered Files and Data Sources")
    discovered_files = registry.discover_sources(settings.data_dir)
    if discovered_files:
        for f in discovered_files:
            file_size_kb = f.stat().st_size / 1024
            st.markdown(f"- 📄 **{f.name}** (`{f.suffix}`) — {file_size_kb:.1f} KB")
    else:
        st.info("No documents currently found in the configured data directory.")

# ------------------------------------------------------------------------------
# Tab 3: API Consumption & Observability Dashboard
# ------------------------------------------------------------------------------
with tab_dashboard:
    st.markdown("### 📊 Operational Observability & Cost Analytics")
    st.caption("Telemetry tracked locally in SQLite with WAL mode and synchronized to LangSmith cloud.")

    kpis = metrics_tracker.get_summary_kpis()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Total Requests</div>
                <div class="carbon-kpi-value">{kpis['total_requests']}</div>
                <div class="carbon-kpi-subtitle">Completed calls</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Total Tokens</div>
                <div class="carbon-kpi-value">{kpis['total_tokens']:,}</div>
                <div class="carbon-kpi-subtitle">In: {kpis['prompt_tokens']:,} | Out: {kpis['completion_tokens']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Mean Latency</div>
                <div class="carbon-kpi-value">{kpis['avg_latency_ms']:.1f} ms</div>
                <div class="carbon-kpi-subtitle">End-to-end pipeline</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Estimated Cost</div>
                <div class="carbon-kpi-value">${kpis['total_cost_usd']:.4f}</div>
                <div class="carbon-kpi-subtitle">Published rate models</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Error Rate</div>
                <div class="carbon-kpi-value">{kpis['error_rate_pct']:.1f}%</div>
                <div class="carbon-kpi-subtitle">{kpis['error_count']} failed runs</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📜 Recent Request Inspection Logs")
    recent_requests = metrics_tracker.get_recent_requests(limit=25)
    if recent_requests:
        try:
            import pandas as pd
            df = pd.DataFrame(recent_requests)
            display_cols = [
                "id", "timestamp", "query", "model_name",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "total_latency_ms", "estimated_cost_usd", "retrieved_chunks_count"
            ]
            valid_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[valid_cols], use_container_width=True, height=350)
        except ImportError:
            st.json(recent_requests)
    else:
        st.info("No requests recorded yet. Execute queries in the RAG Assistant tab to populate telemetry.")

# ------------------------------------------------------------------------------
# Tab 4: LangSmith Evaluation & Benchmarks
# ------------------------------------------------------------------------------
with tab_eval:
    st.markdown("### 🧪 Evaluation & Quality Benchmark Suite")
    st.markdown("""
        Evaluates the end-to-end RAG pipeline against benchmark specifications across three core dimensions:
        - **Faithfulness**: Measures whether answer statements are strictly grounded in retrieved context (penalizing hallucinations).
        - **Answer Relevance**: Measures whether responses directly address question intent.
        - **Context Precision**: Evaluates whether retrieved and reranked chunks contain the target facts.
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Benchmark Evaluation", use_container_width=True):
            evaluator = RAGEvaluator(settings=settings)
            with st.spinner("Evaluating pipeline against benchmark dataset and syncing to LangSmith..."):
                report = evaluator.evaluate_pipeline(pipeline)
                st.session_state["eval_report"] = report
                st.success("Evaluation complete!")

    if "eval_report" in st.session_state:
        report = st.session_state["eval_report"]

        st.divider()
        st.markdown("#### 🏆 Benchmark Performance Scorecard")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Faithfulness", f"{report['mean_faithfulness'] * 100:.1f}%")
        m2.metric("Mean Answer Relevance", f"{report['mean_answer_relevance'] * 100:.1f}%")
        m3.metric("Mean Context Precision", f"{report['mean_context_precision'] * 100:.1f}%")
        m4.metric("Overall RAG Score", f"{report['mean_overall_score'] * 100:.1f}%")

        st.markdown("#### 📋 Sample-by-Sample Breakdown")
        try:
            import pandas as pd
            samples_df = pd.DataFrame(report["samples"])
            st.dataframe(samples_df, use_container_width=True)
        except ImportError:
            st.json(report["samples"])

        st.download_button(
            label="📥 Download Benchmark Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="rag_evaluation_report.json",
            mime="application/json"
        )
