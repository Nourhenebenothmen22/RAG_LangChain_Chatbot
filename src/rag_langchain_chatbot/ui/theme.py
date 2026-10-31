"""IBM Carbon Design System styling and CSS theme for Streamlit with Dark and Light mode support."""

import streamlit as st
from typing import Literal

ThemeMode = Literal["dark", "light"]

# ==============================================================================
# CSS Theme Definitions (IBM Carbon 11 Design System)
# ==============================================================================

DARK_TOKENS = """
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
    --cds-badge-blue-bg: rgba(15, 98, 254, 0.2);
    --cds-badge-blue-text: #78a9ff;
    --cds-badge-blue-border: #0f62fe;
    --cds-badge-green-bg: rgba(36, 161, 72, 0.2);
    --cds-badge-green-text: #6fdc8c;
    --cds-badge-green-border: #24a148;
    --cds-badge-gray-bg: #393939;
    --cds-badge-gray-text: #c6c6c6;
    --cds-badge-gray-border: #525252;
    --cds-card-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
"""

LIGHT_TOKENS = """
    --cds-background: #f4f4f4;
    --cds-layer-01: #ffffff;
    --cds-layer-02: #e0e0e0;
    --cds-layer-03: #c6c6c6;
    --cds-text-primary: #161616;
    --cds-text-secondary: #525252;
    --cds-text-muted: #6f6f6f;
    --cds-interactive-01: #0f62fe;
    --cds-interactive-hover: #0353e9;
    --cds-border-subtle: #e0e0e0;
    --cds-border-strong: #8d8d8d;
    --cds-support-success: #198038;
    --cds-support-warning: #f1c21b;
    --cds-support-error: #da1e28;
    --cds-support-info: #0043ce;
    --cds-badge-blue-bg: rgba(15, 98, 254, 0.12);
    --cds-badge-blue-text: #0043ce;
    --cds-badge-blue-border: #0f62fe;
    --cds-badge-green-bg: rgba(25, 128, 56, 0.12);
    --cds-badge-green-text: #0e6027;
    --cds-badge-green-border: #198038;
    --cds-badge-gray-bg: #e0e0e0;
    --cds-badge-gray-text: #393939;
    --cds-badge-gray-border: #c6c6c6;
    --cds-card-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
"""


def get_carbon_css(theme_mode: ThemeMode = "dark") -> str:
    """Generate dynamic IBM Carbon CSS styles according to selected mode."""
    tokens = DARK_TOKENS if theme_mode == "dark" else LIGHT_TOKENS

    return f"""
<style>
/* ==========================================================================
   IBM Carbon Design System (Carbon 11) - Dynamic Theme
   ========================================================================== */

:root {{
{tokens}
}}

/* Global App Container */
.stApp {{
    background-color: var(--cds-background);
    color: var(--cds-text-primary);
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    transition: background-color 150ms ease, color 150ms ease;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: var(--cds-layer-01) !important;
    border-right: 1px solid var(--cds-border-subtle) !important;
}}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {{
    color: var(--cds-text-primary) !important;
}}

/* Carbon Header Banner */
.carbon-header {{
    background-color: var(--cds-layer-01);
    border-bottom: 1px solid var(--cds-border-subtle);
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--cds-card-shadow);
    flex-wrap: wrap;
    gap: 1rem;
}}

.carbon-header h1 {{
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: var(--cds-text-primary);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.carbon-header .tagline {{
    font-size: 0.85rem;
    color: var(--cds-text-secondary);
    margin-top: 0.25rem;
}}

.carbon-header-badges {{
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
}}

/* Metric / KPI Cards */
.carbon-kpi-card {{
    background-color: var(--cds-layer-01);
    border: 1px solid var(--cds-border-subtle);
    border-top: 3px solid var(--cds-interactive-01);
    padding: 1.25rem;
    margin-bottom: 1rem;
    border-radius: 0px;
    box-shadow: var(--cds-card-shadow);
    transition: transform 120ms ease, box-shadow 120ms ease;
}}

.carbon-kpi-card:hover {{
    transform: translateY(-2px);
    border-top-color: var(--cds-interactive-hover);
}}

.carbon-kpi-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cds-text-secondary);
    margin-bottom: 0.25rem;
    font-weight: 600;
}}

.carbon-kpi-value {{
    font-size: 1.85rem;
    font-weight: 600;
    color: var(--cds-text-primary);
    line-height: 1.2;
}}

.carbon-kpi-subtitle {{
    font-size: 0.75rem;
    color: var(--cds-text-muted);
    margin-top: 0.35rem;
}}

/* Assistant Message Card */
.carbon-assistant-card {{
    background-color: var(--cds-layer-01);
    border: 1px solid var(--cds-border-subtle);
    border-left: 4px solid var(--cds-interactive-01);
    padding: 1.25rem;
    margin-bottom: 1rem;
    border-radius: 0px;
    color: var(--cds-text-primary);
    box-shadow: var(--cds-card-shadow);
}}

/* Source Inspection Cards */
.carbon-source-card {{
    background-color: var(--cds-layer-01);
    border: 1px solid var(--cds-border-subtle);
    border-left: 3px solid var(--cds-interactive-01);
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-radius: 0px;
    box-shadow: var(--cds-card-shadow);
}}

.carbon-source-excerpt {{
    font-size: 0.88rem;
    margin-top: 0.6rem;
    color: var(--cds-text-primary);
    line-height: 1.45;
}}

.carbon-source-footer {{
    font-size: 0.75rem;
    margin-top: 0.5rem;
    color: var(--cds-text-muted);
}}

/* IBM Carbon Badge Pills */
.carbon-badge {{
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-radius: 2px;
    margin-right: 0.4rem;
    margin-bottom: 0.25rem;
}}

.badge-blue {{
    background-color: var(--cds-badge-blue-bg);
    color: var(--cds-badge-blue-text);
    border: 1px solid var(--cds-badge-blue-border);
}}

.badge-green {{
    background-color: var(--cds-badge-green-bg);
    color: var(--cds-badge-green-text);
    border: 1px solid var(--cds-badge-green-border);
}}

.badge-gray {{
    background-color: var(--cds-badge-gray-bg);
    color: var(--cds-badge-gray-text);
    border: 1px solid var(--cds-badge-gray-border);
}}

/* Carbon Tabs Navigation */
.stTabs [data-baseweb="tab-list"] {{
    background-color: var(--cds-layer-01);
    border-bottom: 1px solid var(--cds-border-subtle);
    padding: 0;
    gap: 0;
}}

.stTabs [data-baseweb="tab"] {{
    padding: 0.75rem 1.5rem;
    color: var(--cds-text-secondary) !important;
    border-radius: 0;
    border: none;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 120ms ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--cds-text-primary) !important;
    background-color: var(--cds-layer-02);
}}

.stTabs [aria-selected="true"] {{
    color: var(--cds-text-primary) !important;
    background-color: var(--cds-background) !important;
    border-bottom: 2px solid var(--cds-interactive-01) !important;
    font-weight: 600 !important;
}}

/* Action Buttons */
.stButton > button {{
    background-color: var(--cds-interactive-01) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0px !important;
    padding: 0.65rem 1.25rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: background-color 100ms ease;
    box-shadow: none !important;
}}

.stButton > button:hover {{
    background-color: var(--cds-interactive-hover) !important;
    color: #ffffff !important;
}}

.stButton > button:active {{
    background-color: #002d9c !important;
}}

/* Form Inputs & Chat Input */
.stTextInput > div > div > input,
.stChatInputContainer textarea,
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: var(--cds-layer-01) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    color: var(--cds-text-primary) !important;
    border-radius: 0px !important;
}}

.stTextInput > div > div > input:focus,
.stChatInputContainer textarea:focus {{
    border-color: var(--cds-interactive-01) !important;
    box-shadow: 0 0 0 1px var(--cds-interactive-01) !important;
}}

/* Telemetry caption */
.carbon-caption-bar {{
    background-color: var(--cds-layer-02);
    border: 1px solid var(--cds-border-subtle);
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: var(--cds-text-secondary);
    margin-top: 0.5rem;
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    align-items: center;
}}

/* Expanders */
div[data-testid="stExpander"] {{
    background-color: var(--cds-layer-01) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    border-radius: 0px !important;
}}

div[data-testid="stExpander"] summary {{
    color: var(--cds-text-primary) !important;
    font-weight: 600;
}}

/* Code Blocks & Monospace */
pre, code {{
    border-radius: 0px !important;
    background-color: var(--cds-layer-02) !important;
    color: var(--cds-text-primary) !important;
    border: 1px solid var(--cds-border-subtle) !important;
    font-family: 'IBM Plex Mono', Menlo, Consolas, monospace !important;
    font-size: 0.82rem !important;
}}

/* Monospace telemetry pill in sidebar */
.carbon-telemetry-pill {{
    background-color: var(--cds-layer-02);
    border: 1px solid var(--cds-border-subtle);
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--cds-text-primary);
    margin-bottom: 0.4rem;
    word-break: break-all;
}}

/* Responsive Design Rules */
@media (max-width: 768px) {{
    .carbon-header {{
        flex-direction: column;
        align-items: flex-start;
        padding: 1rem;
    }}
    .carbon-kpi-card {{
        margin-bottom: 0.75rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
    }}
    .carbon-badge {{
        margin-bottom: 0.35rem;
    }}
}}
</style>
"""


def apply_carbon_theme(theme_mode: ThemeMode = "dark") -> None:
    """Inject IBM Carbon Design System CSS styles into the current Streamlit page."""
    st.markdown(get_carbon_css(theme_mode), unsafe_allow_html=True)


def render_carbon_header(theme_mode: ThemeMode = "dark") -> None:
    """Render top enterprise header inspired by IBM Carbon Shell."""
    mode_label = "🌙 Dark Mode" if theme_mode == "dark" else "☀️ Light Mode"
    badge_mode_class = "badge-blue" if theme_mode == "dark" else "badge-green"

    st.markdown(f"""
        <div class="carbon-header">
            <div>
                <h1>⚡ DATABASE-DRIVEN HYBRID RAG STUDIO</h1>
                <div class="tagline">Production-Grade Dual-Engine RAG · LangSmith Observability · ChromaDB + BM25Okapi</div>
            </div>
            <div class="carbon-header-badges">
                <span class="carbon-badge badge-blue">Carbon 11 Theme</span>
                <span class="carbon-badge badge-green">Dual Engine Active</span>
                <span class="carbon-badge {badge_mode_class}">{mode_label}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
