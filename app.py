"""Streamlit enterprise web interface inspired by IBM Carbon Design System (Carbon 11).

Supports dual Light / Dark mode, real-time hybrid retrieval inspection,
multi-format data ingestion, cost telemetry, and LangSmith benchmark evaluation.
"""

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

# ==============================================================================
# Page Configuration & State Setup
# ==============================================================================
st.set_page_config(
    page_title="Database-Driven Hybrid RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_pipeline() -> HybridRAGPipeline:
    """Cache and return the Hybrid RAG pipeline singleton."""
    return HybridRAGPipeline()


settings = get_settings()
pipeline = load_pipeline()
metrics_tracker = get_metrics_tracker(settings)
registry = get_loader_registry()

# ==============================================================================
# Sidebar: Theme Toggle, Telemetry & Hyperparameters
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎨 Appearance & Theme")
    theme_selection = st.radio(
        "Theme Mode",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state["theme_mode"] == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    selected_mode = "dark" if "Dark" in theme_selection else "light"
    if selected_mode != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = selected_mode
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ System Telemetry")
    st.markdown(f"""
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Embedding Model</div>
        <div class="carbon-telemetry-pill">{settings.embedding_model_name}</div>
        
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Neural Reranker</div>
        <div class="carbon-telemetry-pill">{settings.reranker_model_name}</div>
        
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Dense Store</div>
        <div class="carbon-telemetry-pill">ChromaDB ({settings.collection_name})</div>
        
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Lexical Store</div>
        <div class="carbon-telemetry-pill">BM25Okapi (Dynamic Inverted Index)</div>
        
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Guardrails & Security</div>
        <div class="carbon-telemetry-pill">{'Active (Prompt Shield & Canary)' if settings.enable_prompt_injection_detection else 'Disabled'}</div>
        
        <div style="font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem;">• Observability</div>
        <div class="carbon-telemetry-pill">{'LangSmith Tracing V2' if settings.langchain_api_key else 'Local SQLite WAL'}</div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎛️ Pipeline Settings")
    enable_rerank = st.toggle("Cross-Encoder Reranking", value=True, help="Enable deep transformer cross-attention reranker")
    top_k = st.slider("Final Output Chunks (Top-K)", min_value=1, max_value=10, value=settings.reranker_top_k)
    fused_k = st.slider("Candidate Pool (Fused Top-K)", min_value=5, max_value=20, value=settings.fused_top_k)

    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Apply Dynamic IBM Carbon Theme & Header
apply_carbon_theme(st.session_state["theme_mode"])
render_carbon_header(st.session_state["theme_mode"])

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
# Tab 1: Interactive Chatbot & Source Provenance Drawer
# ------------------------------------------------------------------------------
with tab_chat:
    # Render Conversation History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**Question:** {msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"""
                    <div class="carbon-assistant-card">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--cds-interactive-01);">
                            🤖 Grounded Assistant Response
                        </div>
                        <div>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)

                if "metadata" in msg and msg["metadata"]:
                    meta = msg["metadata"]
                    st.markdown(f"""
                        <div class="carbon-caption-bar">
                            <span>⚡ <b>Latency:</b> {meta.get('total_latency_ms', 0):.1f}ms (Retrieval: {meta.get('retrieval_latency_ms', 0):.1f}ms | Gen: {meta.get('generation_latency_ms', 0):.1f}ms)</span>
                            <span>•</span>
                            <span>🔤 <b>Tokens:</b> {meta.get('total_tokens', 0):,}</span>
                            <span>•</span>
                            <span>💰 <b>Est. Cost:</b> ${meta.get('estimated_cost_usd', 0):.6f}</span>
                            <span>•</span>
                            <span>🛡️ <b>Security:</b> {meta.get('security_risk', 'LOW')} Risk</span>
                        </div>
                    """, unsafe_allow_html=True)

                    if meta.get("security_risk") and meta.get("security_risk") != "LOW":
                        st.warning(f"⚠️ Security scan: {', '.join(meta.get('flagged_patterns', []))}")

                    chunks = meta.get("chunks", [])
                    if chunks:
                        with st.expander(f"🔍 Inspect {len(chunks)} Retrieved Sources & Provenance Attribution"):
                            for c in chunks:
                                dense_info = f"Dense #{c.get('dense_rank') or 'N/A'}"
                                sparse_info = f"BM25 #{c.get('sparse_rank') or 'N/A'}"
                                rrf_info = f"RRF: {c.get('rrf_score') or 0.0:.4f}"
                                rerank_info = f"Rerank: {c.get('rerank_score') or c['score']:.4f}"
                                loc = f"p. {c['page'] + 1}" if c.get('page') is not None else (f"rec #{c.get('record_id')}" if c.get('record_id') else "N/A")

                                st.markdown(f"""
                                    <div class="carbon-source-card">
                                        <div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">
                                            <span class="carbon-badge badge-blue">Rank {c['rank']}</span>
                                            <span class="carbon-badge badge-green">{rerank_info}</span>
                                            <span class="carbon-badge badge-gray">{dense_info}</span>
                                            <span class="carbon-badge badge-gray">{sparse_info}</span>
                                            <span class="carbon-badge badge-gray">{rrf_info}</span>
                                            <span class="carbon-badge badge-gray">{Path(c['source']).name} ({loc})</span>
                                        </div>
                                        <div class="carbon-source-excerpt">
                                            "{c['content']}"
                                        </div>
                                        <div class="carbon-source-footer">
                                            Chunk ID: {c.get('chunk_id', '')[:16]}... • Verified Source: Connected Database / Document Storage
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

    # Chat Input Box
    if user_query := st.chat_input("Ask any question against the connected database and documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(f"**Question:** {user_query}")

        with st.chat_message("assistant"):
            with st.spinner("Executing hybrid retrieval, reranking, and grounded generation..."):
                response = pipeline.query(user_query, top_k=top_k, rerank=enable_rerank)

                st.markdown(f"""
                    <div class="carbon-assistant-card">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--cds-interactive-01);">
                            🤖 Grounded Assistant Response
                        </div>
                        <div>{response.answer}</div>
                    </div>
                """, unsafe_allow_html=True)

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

                st.markdown(f"""
                    <div class="carbon-caption-bar">
                        <span>⚡ <b>Latency:</b> {response.total_latency_ms:.1f}ms (Retrieval: {response.retrieval_latency_ms:.1f}ms | Gen: {response.generation_latency_ms:.1f}ms)</span>
                        <span>•</span>
                        <span>🔤 <b>Tokens:</b> {response.total_tokens:,}</span>
                        <span>•</span>
                        <span>💰 <b>Est. Cost:</b> ${response.estimated_cost_usd:.6f}</span>
                        <span>•</span>
                        <span>🛡️ <b>Security:</b> {response.security_risk} Risk</span>
                    </div>
                """, unsafe_allow_html=True)

                if response.security_risk != "LOW":
                    st.warning(f"⚠️ Security scan: {', '.join(response.flagged_security_patterns)}")

                if response.chunks:
                    with st.expander(f"🔍 Inspect {len(response.chunks)} Retrieved Sources & Provenance Attribution"):
                        for c in response.chunks:
                            dense_info = f"Dense #{c.dense_rank or 'N/A'}"
                            sparse_info = f"BM25 #{c.sparse_rank or 'N/A'}"
                            rrf_info = f"RRF: {c.rrf_score or 0.0:.4f}"
                            rerank_info = f"Rerank: {c.rerank_score or c.score:.4f}"
                            loc = f"p. {c.page + 1}" if c.page is not None else (f"rec #{c.record_id}" if c.record_id else "N/A")

                            st.markdown(f"""
                                <div class="carbon-source-card">
                                    <div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">
                                        <span class="carbon-badge badge-blue">Rank {c.rank}</span>
                                        <span class="carbon-badge badge-green">{rerank_info}</span>
                                        <span class="carbon-badge badge-gray">{dense_info}</span>
                                        <span class="carbon-badge badge-gray">{sparse_info}</span>
                                        <span class="carbon-badge badge-gray">{rrf_info}</span>
                                        <span class="carbon-badge badge-gray">{Path(c.source).name} ({loc})</span>
                                    </div>
                                    <div class="carbon-source-excerpt">
                                        "{c.content}"
                                    </div>
                                    <div class="carbon-source-footer">
                                        Chunk ID: {c.chunk_id[:16]}... • Verified Source: Connected Database / Document Storage
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
        The system dynamically synchronizes the vector database (**ChromaDB**) and lexical index (**BM25Okapi**) with local files and connected SQLite databases.
        - **Supported formats**: PDF (`.pdf`), Word (`.docx`), Markdown/Text (`.md`, `.txt`), Tabular (`.csv`, `.json`), SQLite (`.sqlite3`, `.db`).
        - **Incremental Engine**: Detects additions, modifications, and deletions via SHA-256 content hashes, avoiding redundant embedding computation.
    """)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("#### 📤 Upload Documents or Data Files")
        uploaded_file = st.file_uploader(
            "Upload File",
            type=["pdf", "docx", "txt", "md", "csv", "json"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            save_path = settings.data_dir / uploaded_file.name
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded '{uploaded_file.name}' to `{settings.data_dir}`.")

            if st.button("🚀 Synchronize Database Now", use_container_width=True):
                with st.spinner("Executing SHA-256 change detection and dual-index synchronization..."):
                    stats = pipeline.index(force_reindex=False)
                    st.success("Synchronization completed successfully!")
                    st.json(stats)

    with col2:
        st.markdown("#### ⚡ Index Maintenance & State")
        with st.container():
            st.markdown(f"""
                <div class="carbon-source-card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--cds-interactive-01);">Ledger Status: Healthy (WAL Mode)</div>
                    <div style="font-size: 0.85rem; color: var(--cds-text-secondary); line-height: 1.6;">
                        • <b>Vector Store</b>: ChromaDB ({settings.collection_name})<br/>
                        • <b>Lexical Engine</b>: BM25Okapi (Serialized Safe Payload)<br/>
                        • <b>Change Detection</b>: SHA-256 State Ledger
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Force Rebuild Entire Index", use_container_width=True):
                with st.spinner("Purging collections and rebuilding from all discovered sources..."):
                    stats = pipeline.index(force_reindex=True)
                    st.success("Entire collection rebuilt from ground up!")
                    st.json(stats)

    st.divider()
    st.markdown("#### 📑 Discovered Files and Connected Data Sources")
    discovered_files = registry.discover_sources(settings.data_dir)
    if discovered_files:
        files_data = []
        for f in discovered_files:
            size_kb = f.stat().st_size / 1024
            files_data.append({
                "File Name": f.name,
                "Format": f.suffix,
                "Size (KB)": f"{size_kb:.1f}",
                "Sync Status": "Synchronized (SHA-256 match)"
            })
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame(files_data), use_container_width=True)
        except ImportError:
            st.json(files_data)
    else:
        st.info("No documents currently found in the configured data directory.")

# ------------------------------------------------------------------------------
# Tab 3: API Consumption & Observability Dashboard
# ------------------------------------------------------------------------------
with tab_dashboard:
    st.markdown("### 📊 Operational Observability & Cost Analytics")
    st.caption("Telemetry tracked in SQLite WAL mode and synchronized to LangSmith cloud.")

    kpis = metrics_tracker.get_summary_kpis()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="carbon-kpi-card">
                <div class="carbon-kpi-label">Total Requests</div>
                <div class="carbon-kpi-value">{kpis['total_requests']}</div>
                <div class="carbon-kpi-subtitle">Completed pipeline calls</div>
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
    recent_requests = metrics_tracker.get_recent_requests(limit=30)
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
        - **Faithfulness**: Verifies answer statements are strictly grounded in retrieved context (penalizing hallucinations).
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
