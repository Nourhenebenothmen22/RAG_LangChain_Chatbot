"""IBM Carbon Design System styling and CSS theme for Streamlit."""

import streamlit as st

CARBON_CSS = """
<style>
/* ==========================================================================
   IBM Carbon Design System (Carbon 11 Theme - Gray 100 / Gray 90)
   ========================================================================== */

/* Root Variables */
:root {
    --cds-background: #161616;
    --cds-layer-01: #262626;
    --cds-layer-02: #393939;
    --cds-layer-03: #525252;
    --cds-text-primary: #f4f4f4;
    --cds-text-secondary: #c6c6c6;
    --cds-text-muted: #8d8d8d;
    --cds-interactive-01: #0f62fe;
    --cds-interactive-hover: #0353e9;
    --cds-border-subtle: #393939;
    --cds-border-strong: #6f6f6f;
    --cds-support-success: #24a148;
    --cds-support-warning: #f1c21b;
    --cds-support-error: #da1e28;
    --cds-support-info: #4589ff;
}

/* Global App Container */
.stApp {
    background-color: var(--cds-background);
    color: var(--cds-text-primary);
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Carbon Header Banner */
.carbon-header {
    background-color: var(--cds-layer-01);
    border-bottom: 1px solid var(--cds-border-subtle);
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.carbon-header h1 {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: var(--cds-text-primary);
    letter-spacing: -0.02em;
}

.carbon-header .tagline {
    font-size: 0.85rem;
    color: var(--cds-text-secondary);
}

/* Metric / KPI Cards */
.carbon-kpi-card {
    background-color: var(--cds-layer-01);
    border: 1px solid var(--cds-border-subtle);
    border-top: 3px solid var(--cds-interactive-01);
    padding: 1.25rem;
    margin-bottom: 1rem;
    border-radius: 0px;
}

.carbon-kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cds-text-secondary);
    margin-bottom: 0.25rem;
}

.carbon-kpi-value {
    font-size: 1.85rem;
    font-weight: 600;
    color: var(--cds-text-primary);
    line-height: 1.2;
}

.carbon-kpi-subtitle {
    font-size: 0.75rem;
    color: var(--cds-text-muted);
    margin-top: 0.35rem;
}

/* Source Inspection Cards */
.carbon-source-card {
    background-color: var(--cds-layer-01);
    border: 1px solid var(--cds-border-subtle);
    border-left: 3px solid var(--cds-interactive-01);
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-radius: 0px;
}

.carbon-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-radius: 2px;
    margin-right: 0.5rem;
}

.badge-blue {
    background-color: rgba(15, 98, 254, 0.2);
    color: #78a9ff;
    border: 1px solid #0f62fe;
}

.badge-green {
    background-color: rgba(36, 161, 72, 0.2);
    color: #6fdc8c;
    border: 1px solid #24a148;
}

.badge-gray {
    background-color: #393939;
    color: #c6c6c6;
    border: 1px solid #525252;
}

/* Carbon Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--cds-layer-01);
    border-bottom: 1px solid var(--cds-border-subtle);
    padding: 0;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1.5rem;
    color: var(--cds-text-secondary);
    border-radius: 0;
    border: none;
    font-weight: 500;
    font-size: 0.875rem;
}

.stTabs [aria-selected="true"] {
    color: var(--cds-text-primary);
    background-color: var(--cds-background);
    border-bottom: 2px solid var(--cds-interactive-01);
}

/* Buttons */
.stButton > button {
    background-color: var(--cds-interactive-01);
    color: #ffffff;
    border: none;
    border-radius: 0px;
    padding: 0.6rem 1.25rem;
    font-size: 0.875rem;
    font-weight: 500;
    transition: background-color 70ms cubic-bezier(0, 0, 0.38, 0.9);
}

.stButton > button:hover {
    background-color: var(--cds-interactive-hover);
    color: #ffffff;
}

/* Chat Input & Text Inputs */
.stTextInput > div > div > input,
.stChatInputContainer textarea {
    background-color: var(--cds-layer-01) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    color: var(--cds-text-primary) !important;
    border-radius: 0px !important;
}

.stTextInput > div > div > input:focus,
.stChatInputContainer textarea:focus {
    border-color: var(--cds-interactive-01) !important;
    box-shadow: 0 0 0 1px var(--cds-interactive-01) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--cds-layer-01) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    border-radius: 0px !important;
    color: var(--cds-text-primary) !important;
    font-size: 0.85rem !important;
}

/* Code block adjustments */
pre, code {
    border-radius: 0px !important;
    background-color: var(--cds-layer-01) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    font-family: 'IBM Plex Mono', Menlo, Consolas, monospace !important;
    font-size: 0.82rem !important;
}
</style>
"""


def apply_carbon_theme() -> None:
    """Inject IBM Carbon Design System CSS styles into current Streamlit page."""
    st.markdown(CARBON_CSS, unsafe_allow_html=True)


def render_carbon_header() -> None:
    """Render top enterprise header inspired by IBM Carbon Shell."""
    st.markdown("""
        <div class="carbon-header">
            <div>
                <h1>ClaimEye AI <span>/ Hybrid RAG Enterprise Studio</span></h1>
                <div class="tagline">Production-grade Retrieval-Augmented Generation · LangSmith Observability · Dual Vector + BM25</div>
            </div>
            <div>
                <span class="carbon-badge badge-blue">Carbon 11 Theme</span>
                <span class="carbon-badge badge-green">Dual Engine Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
