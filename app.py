"""
PSX AI Investment Advisory System
BS Computer Science — 6th Semester AI Project
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os

from scoring_engine import (
    score_all, allocate_portfolio, portfolio_metrics,
    hill_climbing, simulated_annealing
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PSX AI Investment Advisory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Portfolio"
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* ── App shell ── */
  .stApp {
    background: #0D1B2A;
    color: #E2E8F0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2236 0%, #0D1B2A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="stSidebar"] .stTextInput > div > div > input,
  [data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stSidebar"] .stSlider > div > div > div {
    color: #38BDF8 !important;
  }
  [data-testid="stSidebar"] .stMultiSelect > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(30,136,229,0.4) !important;
    transition: all 0.2s ease !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #42A5F5, #1E88E5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(30,136,229,0.55) !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #38BDF8 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
  }

  /* ── Top header bar ── */
  .top-header {
    background: linear-gradient(90deg, #0F2236 0%, #112840 100%);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    position: sticky;
    top: 0;
    z-index: 999;
  }
  .header-title {
    font-size: 16px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: 0.2px;
  }
  .header-title span { color: #38BDF8; }
  .nav-tabs {
    display: flex;
    gap: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 4px;
  }
  .nav-tab {
    padding: 7px 18px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 500;
    color: #94A3B8;
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all 0.15s ease;
    white-space: nowrap;
    text-decoration: none;
  }
  .nav-tab:hover { color: #E2E8F0; background: rgba(255,255,255,0.08); }
  .nav-tab.active {
    color: #F1F5F9 !important;
    background: #1E3A5F !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }

  /* ── Main content area ── */
  .main-content {
    padding: 28px 36px 40px;
    background: #0D1B2A;
    min-height: calc(100vh - 60px);
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(145deg, #112840 0%, #0F2236 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 24px;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
  }
  .metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
  }
  .metric-card.blue::after  { background: linear-gradient(90deg, #38BDF8, #1E88E5); }
  .metric-card.green::after { background: linear-gradient(90deg, #34D399, #10B981); }
  .metric-card.amber::after { background: linear-gradient(90deg, #FBBF24, #F59E0B); }
  .metric-card.purple::after{ background: linear-gradient(90deg, #A78BFA, #7C3AED); }
  .metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin: 4px 0;
  }
  .metric-badge {
    font-size: 11px;
    font-weight: 600;
  }
  .metric-badge.up   { color: #34D399; }
  .metric-badge.ok   { color: #34D399; }
  .metric-badge.info { color: #94A3B8; }
  .metric-badge.star { color: #FBBF24; }

  /* ── Section heading ── */
  .section-heading {
    font-size: 16px;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 16px;
    margin-top: 28px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin-left: 8px;
  }

  /* ── Panel/Card ── */
  .panel {
    background: linear-gradient(145deg, #112840 0%, #0F2236 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 16px;
  }

  /* ── Stock table ── */
  .stock-table { width: 100%; border-collapse: collapse; }
  .stock-table th {
    font-size: 10px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    text-align: left;
  }
  .stock-table td {
    padding: 11px 12px;
    font-size: 13px;
    color: #CBD5E1;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .stock-table tr:last-child td { border-bottom: none; }
  .stock-table tr:hover td { background: rgba(255,255,255,0.03); }
  .sym-badge {
    font-weight: 700;
    color: #38BDF8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }
  .green-text { color: #34D399; font-weight: 600; }
  .amber-text { color: #FBBF24; font-weight: 600; }
  .score-pill {
    background: rgba(30,136,229,0.15);
    color: #38BDF8;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Reason box ── */
  .reason-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 12px;
    padding: 20px 24px;
  }
  .reason-box .reason-title {
    color: #38BDF8;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 14px;
  }
  .reason-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 13px;
    color: #94A3B8;
  }
  .reason-item:last-child { border-bottom: none; }
  .reason-item .label { color: #94A3B8; }
  .reason-item .val   { color: #F1F5F9; font-weight: 600; }

  /* ── Sidebar investor badge ── */
  .welcome-card {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(30,136,229,0.08));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  .welcome-name {
    font-size: 16px;
    font-weight: 700;
    color: #F1F5F9 !important;
  }
  .welcome-sub {
    font-size: 11px;
    color: #64748B !important;
    margin-top: 2px;
  }

  /* ── Algo button group ── */
  .algo-group {
    display: flex;
    gap: 0;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    margin-top: 4px;
  }
  .algo-btn {
    flex: 1;
    text-align: center;
    padding: 8px 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    color: #64748B !important;
    background: transparent;
    border: none;
  }
  .algo-btn.active-algo {
    background: #1E3A5F !important;
    color: #38BDF8 !important;
  }

  /* ── Footer ── */
  .app-footer {
    text-align: center;
    color: #2D4A63;
    font-size: 11px;
    padding: 24px 0 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 40px;
  }

  /* Streamlit internal overrides for main area */
  .main .block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
  }
  div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

  /* Dataframe styling */
  .stDataFrame { background: transparent !important; }
  iframe { border-radius: 10px !important; }

  /* Spinner */
  .stSpinner > div { color: #38BDF8 !important; }

  /* Radio button override */
  [data-testid="stRadio"] > div { flex-direction: row !important; gap: 4px !important; }
  [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.05);
    border-radius: 7px;
    padding: 5px 14px !important;
    font-size: 12px !important;
    cursor: pointer;
  }
  [data-testid="stRadio"] label[data-checked="true"] {
    background: #1E3A5F !important;
    color: #38BDF8 !important;
  }

  /* Alert/error */
  .stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "stocks_data.csv")
    return pd.read_csv(path)

df = load_data()
ALL_SECTORS = sorted(df["Sector"].unique().tolist())

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;padding:4px 0;">
      <div style="width:34px;height:34px;background:linear-gradient(135deg,#1E88E5,#38BDF8);
                  border-radius:9px;display:flex;align-items:center;justify-content:center;
                  font-size:18px;">📈</div>
      <div>
        <div style="font-size:15px;font-weight:800;color:#F1F5F9;letter-spacing:-0.3px;">PSX InvestAI</div>
        <div style="font-size:10px;color:#475569;letter-spacing:0.5px;">ADVISORY SYSTEM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # User name
    user_name = st.text_input("Your Name", value="", placeholder="Enter your name…",
                               label_visibility="collapsed")

    if user_name.strip():
        st.markdown(f"""
        <div class="welcome-card">
          <div class="welcome-name">👋 {user_name.strip()}</div>
          <div class="welcome-sub">Welcome to your investment dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome-card">
          <div class="welcome-name">👋 Investor</div>
          <div class="welcome-sub">Enter your name above</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── INVESTOR SETTINGS ──
    st.markdown("### ⚙️ Investor Settings")

    amount = st.number_input(
        "Investment Amount (PKR)",
        min_value=100_000, max_value=100_000_000,
        value=5_000_000, step=100_000,
        help="Total capital to invest"
    )
    # Show formatted amount
    st.markdown(f"<div style='font-size:13px;color:#38BDF8;font-weight:700;margin:-10px 0 10px 0;font-family:JetBrains Mono,monospace;'>Rs {amount:,}</div>", unsafe_allow_html=True)

    duration = st.slider("Duration (Years)", 1, 15, 5)
    target_return = st.slider("Target Annual Return (%)", 5, 50, 20)

    risk = st.radio("Risk Appetite", ["Low", "Medium", "High"], index=1, horizontal=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── SECTOR PREFERENCES ──
    st.markdown("### 🏭 Preferred Sectors")
    preferred = st.multiselect(
        "Select sectors", options=ALL_SECTORS,
        default=["Banking", "Energy"],
        label_visibility="collapsed"
    )
    excluded = st.multiselect(
        "Excluded Sectors",
        options=[s for s in ALL_SECTORS if s not in preferred],
        default=[], placeholder="Exclude sectors…"
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── PORTFOLIO SIZE ──
    st.markdown("### 📦 Portfolio Size")
    n_stocks = st.slider("Stocks", 3, 12, 5, label_visibility="collapsed",
                          format="%d stocks")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── ALGORITHM ──
    st.markdown("### 🧠 Algorithm")
    algo = st.radio(
        "Algorithm",
        ["Hill Climbing", "Sim. Annealing", "Both"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    # Map short label back
    if algo == "Sim. Annealing":
        algo = "Simulated Annealing"

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  TOP NAV BAR
# ─────────────────────────────────────────────
TABS = [
    ("Portfolio",    "📊"),
    ("AI Reasoning", "🤖"),
    ("Optimization", "📈"),
    ("All Stocks",   "📋"),
]

# Build nav HTML
nav_html = '<div class="top-header">'
nav_html += '<div class="header-title">PSX <span>AI Investment Advisory System</span> — Streamlit Dashboard</div>'
nav_html += '<div class="nav-tabs">'
for tab_name, icon in TABS:
    active_class = "active" if st.session_state.active_tab == tab_name else ""
    nav_html += f'<button class="nav-tab {active_class}" onclick="void(0)">{icon} {tab_name}</button>'
nav_html += '</div></div>'

st.markdown(nav_html, unsafe_allow_html=True)

# Actual tab switcher using columns + buttons (these drive state)
nav_cols = st.columns(len(TABS))
for i, (tab_name, icon) in enumerate(TABS):
    with nav_cols[i]:
        if st.button(
            f"{icon} {tab_name}",
            key=f"nav_{tab_name}",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.active_tab = tab_name
            st.rerun()

# Style the functional nav buttons to be invisible (header HTML handles visuals)
st.markdown("""
<style>
  div[data-testid="stHorizontalBlock"] > div:first-child > div > div > button,
  div[data-testid="stHorizontalBlock"] > div > div > div > button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    height: 1px !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 0 !important;
    overflow: hidden !important;
    min-height: 0 !important;
  }
</style>
""", unsafe_allow_html=True)

active = st.session_state.active_tab

# ─────────────────────────────────────────────
#  MAIN WRAPPER
# ─────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  WELCOME SCREEN (no run yet)
# ─────────────────────────────────────────────
if not run_btn and not st.session_state.results_ready:
    st.markdown("""
    <div style="text-align:center;padding:60px 0 40px;">
      <div style="font-size:56px;margin-bottom:16px;">📈</div>
      <h1 style="font-size:28px;font-weight:800;color:#F1F5F9;margin:0 0 8px;">PSX AI Investment Advisory</h1>
      <p style="color:#64748B;font-size:15px;max-width:480px;margin:0 auto 32px;">
        Configure your investor profile in the sidebar and click <b style="color:#38BDF8">Run Analysis</b> to generate an optimised portfolio.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, step, desc in [
        (c1, "🎯", "Step 1", "Fill investor profile in sidebar"),
        (c2, "🏭", "Step 2", "Select preferred sectors"),
        (c3, "🚀", "Step 3", "Click Run Analysis"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#112840,#0F2236);border:1px solid rgba(255,255,255,0.08);
                        border-radius:14px;padding:24px;text-align:center;">
              <div style="font-size:32px;margin-bottom:12px;">{icon}</div>
              <div style="font-size:11px;font-weight:700;color:#38BDF8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{step}</div>
              <div style="font-size:13px;color:#94A3B8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Available Stock Database</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=380, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  SCORING + OPTIMISATION
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("⚙️ Scoring stocks…"):
        time.sleep(0.3)
        scored_df = score_all(df, preferred, excluded, risk)

    if scored_df.empty:
        st.error("No stocks found after applying sector filters. Please adjust your preferences.")
        st.stop()

    top_stocks = scored_df.head(n_stocks).reset_index(drop=True)

    with st.spinner("🧠 Running optimisation algorithm(s)…"):
        time.sleep(0.4)

        if algo in ["Hill Climbing", "Both"]:
            hc_weights, hc_history, hc_iters = hill_climbing(top_stocks, risk, n_stocks)
            hc_metrics = portfolio_metrics(top_stocks, hc_weights)
            hc_alloc = [{
                "Symbol": row["Symbol"], "Name": row["Name"], "Sector": row["Sector"],
                "Score": row["Total_Score"],
                "Allocation %": round(hc_weights[i] * 100, 1),
                "Amount (PKR)": int(hc_weights[i] * amount),
                "Return": round(row.get("Expected_Return", row.get("Growth_5yr", 0)) * 100, 1),
                "Div %": round(row.get("Dividend_Yield", 0), 1),
            } for i, (_, row) in enumerate(top_stocks.iterrows())]

        if algo in ["Simulated Annealing", "Both"]:
            sa_weights, sa_history, sa_iters = simulated_annealing(top_stocks, risk, n_stocks)
            sa_metrics = portfolio_metrics(top_stocks, sa_weights)
            sa_alloc = [{
                "Symbol": row["Symbol"], "Name": row["Name"], "Sector": row["Sector"],
                "Score": row["Total_Score"],
                "Allocation %": round(sa_weights[i] * 100, 1),
                "Amount (PKR)": int(sa_weights[i] * amount),
                "Return": round(row.get("Expected_Return", row.get("Growth_5yr", 0)) * 100, 1),
                "Div %": round(row.get("Dividend_Yield", 0), 1),
            } for i, (_, row) in enumerate(top_stocks.iterrows())]

    if algo == "Hill Climbing":
        display_alloc, display_weights, display_metrics = hc_alloc, hc_weights, hc_metrics
        hc_alloc_stored = hc_alloc; hc_history_stored = hc_history; hc_iters_stored = hc_iters; hc_metrics_stored = hc_metrics
        sa_alloc_stored = sa_alloc_stored = None; sa_history_stored = None; sa_iters_stored = None; sa_metrics_stored = None
    elif algo == "Simulated Annealing":
        display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics
        hc_alloc_stored = None; hc_history_stored = None; hc_iters_stored = None; hc_metrics_stored = None
        sa_alloc_stored = sa_alloc; sa_history_stored = sa_history; sa_iters_stored = sa_iters; sa_metrics_stored = sa_metrics
    else:
        display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics
        hc_alloc_stored = hc_alloc; hc_history_stored = hc_history; hc_iters_stored = hc_iters; hc_metrics_stored = hc_metrics
        sa_alloc_stored = sa_alloc; sa_history_stored = sa_history; sa_iters_stored = sa_iters; sa_metrics_stored = sa_metrics

    st.session_state.analysis_data = {
        "scored_df": scored_df, "top_stocks": top_stocks,
        "display_alloc": display_alloc, "display_weights": display_weights, "display_metrics": display_metrics,
        "algo": algo, "preferred": preferred,
        "hc_alloc": hc_alloc_stored, "hc_history": hc_history_stored,
        "hc_iters": hc_iters_stored, "hc_metrics": hc_metrics_stored,
        "sa_alloc": sa_alloc_stored, "sa_history": sa_history_stored,
        "sa_iters": sa_iters_stored, "sa_metrics": sa_metrics_stored,
        "amount": amount, "target_return": target_return,
    }
    st.session_state.results_ready = True
    st.session_state.active_tab = "Portfolio"
    st.rerun()

# ─────────────────────────────────────────────
#  UNPACK CACHED RESULTS
# ─────────────────────────────────────────────
if not st.session_state.results_ready or st.session_state.analysis_data is None:
    st.info("Configure settings in the sidebar and click **Run Analysis**.")
    st.stop()

D = st.session_state.analysis_data
scored_df        = D["scored_df"]
top_stocks       = D["top_stocks"]
display_alloc    = D["display_alloc"]
display_metrics  = D["display_metrics"]
algo             = D["algo"]
preferred        = D["preferred"]
amount_display   = D["amount"]
target_return    = D["target_return"]

# ══════════════════════════════════════════════
#  TAB: PORTFOLIO
# ══════════════════════════════════════════════
if active == "Portfolio":

    # ── Metrics row ──
    exp_ret  = display_metrics.get("Expected Return (%)", 0)
    port_risk= display_metrics.get("Portfolio Risk", 0)
    div_yld  = display_metrics.get("Avg Dividend Yield", 0)
    sharpe   = display_metrics.get("Sharpe-like Ratio", 0)
    above    = "▲ Above target" if exp_ret >= target_return else "▼ Below target"
    above_cls= "up" if exp_ret >= target_return else "amber-text"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card blue">
          <div class="metric-label">Expected Return</div>
          <div class="metric-value">{exp_ret:.1f}%</div>
          <div class="metric-badge up">{above}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        risk_ok = port_risk < 0.25
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-label">Portfolio Risk</div>
          <div class="metric-value">{port_risk:.2f}</div>
          <div class="metric-badge ok">{'✓ Within tolerance' if risk_ok else '⚠ Above tolerance'}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card amber">
          <div class="metric-label">Avg Dividend Yield</div>
          <div class="metric-value">{div_yld:.1f}%</div>
          <div class="metric-badge info">Annual income</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        sh_lbl = "★ Excellent" if sharpe >= 1.5 else "✓ Good" if sharpe >= 1.0 else "◇ Fair"
        st.markdown(f"""
        <div class="metric-card purple">
          <div class="metric-label">Sharpe Ratio</div>
          <div class="metric-value">{sharpe:.2f}</div>
          <div class="metric-badge star">{sh_lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Portfolio Allocation</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Donut Chart</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame(display_alloc)
        colors = ["#1E88E5","#26A69A","#EF9A9A","#FFA726","#AB47BC",
                  "#42A5F5","#66BB6A","#FF7043","#EC407A","#78909C"]
        fig_donut = go.Figure(go.Pie(
            labels=pie_df["Symbol"],
            values=pie_df["Allocation %"],
            hole=0.58,
            marker=dict(colors=colors[:len(pie_df)], line=dict(color="#0D1B2A", width=3)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#E2E8F0"),
            hovertemplate="<b>%{label}</b><br>Allocation: %{value}%<extra></extra>",
        ))
        n_items = len(pie_df)
        fig_donut.add_annotation(
            text=f"<b>{n_items} Stocks</b><br><span style='font-size:11px;color:#64748B'>Diversified</span>",
            x=0.5, y=0.5, showarrow=False, align="center",
            font=dict(size=14, color="#F1F5F9"),
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(
                font=dict(color="#94A3B8", size=11),
                bgcolor="rgba(0,0,0,0)",
                orientation="v", x=1, y=0.5
            ),
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Stock Breakdown</div>', unsafe_allow_html=True)
        rows_html = ""
        for item in display_alloc:
            ret_val = item.get("Return", 0)
            ret_cls = "green-text" if ret_val >= 15 else "amber-text" if ret_val >= 8 else ""
            score_val = f"{item['Score']:.0f}/100"
            rows_html += f"""
            <tr>
              <td><span class="sym-badge">{item['Symbol']}</span></td>
              <td style="color:#94A3B8;font-size:12px;">{item['Sector']}</td>
              <td><span class="score-pill">{score_val}</span></td>
              <td style="color:#E2E8F0;font-weight:600;">{item['Allocation %']}%</td>
              <td class="{ret_cls}">{ret_val:.1f}%</td>
              <td style="color:#FBBF24;font-weight:600;">{item['Div %']:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <table class="stock-table">
          <thead><tr>
            <th>Stock</th><th>Sector</th><th>Score</th>
            <th>Alloc %</th><th>Return</th><th>Div %</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Sector bar ──
    st.markdown('<div class="section-heading">Allocation by Sector</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    sector_alloc = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    sector_alloc = sector_alloc.sort_values("Allocation %", ascending=True)
    bar_colors = ["#1E88E5","#26A69A","#EF9A9A","#FFA726","#AB47BC","#42A5F5","#66BB6A","#FF7043"]
    fig_bar = go.Figure(go.Bar(
        x=sector_alloc["Allocation %"],
        y=sector_alloc["Sector"],
        orientation="h",
        marker=dict(
            color=bar_colors[:len(sector_alloc)],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        text=[f"{v:.0f}%" for v in sector_alloc["Allocation %"]],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=12),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color="#64748B"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#94A3B8",
                   tickfont=dict(size=12)),
        height=max(180, len(sector_alloc) * 46 + 40),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB: AI REASONING
# ══════════════════════════════════════════════
elif active == "AI Reasoning":

    st.markdown('<div class="section-heading">🤖 AI Recommendation Reasoning</div>', unsafe_allow_html=True)

    stock_options = [f"{r['Symbol']} — {r['Name']}" for r in display_alloc]

    # Preserve selected stock in session state to avoid re-render jump
    if st.session_state.selected_stock not in stock_options:
        st.session_state.selected_stock = stock_options[0]

    def _on_stock_change():
        st.session_state.selected_stock = st.session_state._stock_selector
        # Do NOT change active_tab

    selected = st.selectbox(
        "Select a stock to explain:",
        options=stock_options,
        index=stock_options.index(st.session_state.selected_stock),
        key="_stock_selector",
        on_change=_on_stock_change,
    )
    sym = st.session_state.selected_stock.split(" — ")[0]

    stock_row = scored_df[scored_df["Symbol"] == sym].iloc[0]
    breakdown = stock_row["Breakdown"]

    col_reason, col_chart = st.columns([1, 1], gap="medium")

    with col_reason:
        vol = stock_row["Volatility"]
        vol_label = "✅ Low" if vol < 0.15 else "⚠️ Moderate" if vol < 0.25 else "🔴 High"
        sector_label = "✅ Preferred" if stock_row["Sector"] in preferred else "🔵 Neutral"

        st.markdown(f"""
        <div class="reason-box">
          <div class="reason-title">Why {sym} was selected</div>
          <div class="reason-item">
            <span class="label">Growth (5yr)</span>
            <span class="val">{stock_row['Growth_5yr']*100:.1f}%</span>
          </div>
          <div class="reason-item">
            <span class="label">Volatility</span>
            <span class="val">{vol:.2f} &nbsp; {vol_label}</span>
          </div>
          <div class="reason-item">
            <span class="label">Dividend Yield</span>
            <span class="val">{stock_row['Dividend_Yield']:.1f}%</span>
          </div>
          <div class="reason-item">
            <span class="label">Sector</span>
            <span class="val">{stock_row['Sector']} &nbsp; {sector_label}</span>
          </div>
          <div class="reason-item">
            <span class="label">Momentum Score</span>
            <span class="val">{stock_row['Momentum_Score']:.2f} / 1.0</span>
          </div>
          <div class="reason-item" style="border-bottom:none;">
            <span class="label">AI Total Score</span>
            <span class="val" style="color:#38BDF8;font-size:15px;">{stock_row['Total_Score']:.1f}/100</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        components = list(breakdown.keys())
        scores_bd  = list(breakdown.values())
        maxes      = [30, 20, 15, 20, 15]

        fig_score = go.Figure()
        fig_score.add_trace(go.Bar(
            x=scores_bd, y=components, orientation="h",
            marker=dict(color=["#1E88E5","#26A69A","#FFA726","#AB47BC","#EF9A9A"],
                        line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.1f}" for v in scores_bd],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
            hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
            name="Score",
        ))
        # Max markers
        for i, mx in enumerate(maxes[:len(components)]):
            fig_score.add_trace(go.Scatter(
                x=[mx], y=[components[i]],
                mode="markers",
                marker=dict(symbol="line-ns", size=14, color="#475569",
                            line=dict(width=2, color="#475569")),
                showlegend=False,
                hovertemplate=f"Max: {mx}<extra></extra>",
            ))
        fig_score.update_layout(
            title=dict(text="Heuristic Score Breakdown", font=dict(color="#94A3B8", size=13), x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=36, b=10, l=10, r=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#94A3B8",
                       tickfont=dict(size=11)),
            height=260,
            bargap=0.3,
        )
        st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
#  TAB: OPTIMIZATION
# ══════════════════════════════════════════════
elif active == "Optimization":

    st.markdown('<div class="section-heading">⚙️ Optimisation Algorithm Results</div>', unsafe_allow_html=True)

    hc_alloc    = D["hc_alloc"];    hc_history = D["hc_history"]
    hc_iters    = D["hc_iters"];    hc_metrics = D["hc_metrics"]
    sa_alloc    = D["sa_alloc"];    sa_history = D["sa_history"]
    sa_iters    = D["sa_iters"];    sa_metrics = D["sa_metrics"]

    def convergence_fig(history, title, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history, mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.07)").replace("rgb", "rgba") if "rgb" in color else color + "12",
            hovertemplate="Iter %{x}: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#94A3B8", size=13), x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=36, b=20, l=10, r=10),
            xaxis=dict(color="#64748B", gridcolor="rgba(255,255,255,0.04)",
                       title="Iteration", titlefont=dict(size=11)),
            yaxis=dict(color="#64748B", gridcolor="rgba(255,255,255,0.04)",
                       title="Objective Score", titlefont=dict(size=11)),
            height=240,
        )
        return fig

    if algo == "Both" and hc_metrics and sa_metrics:
        oc1, oc2 = st.columns(2, gap="medium")
        with oc1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">🔵 Hill Climbing</div>', unsafe_allow_html=True)
            for k, v in hc_metrics.items():
                st.metric(k, v)
            st.caption(f"Iterations: {hc_iters}  |  ⚠️ May get stuck at local optima")
            st.plotly_chart(convergence_fig(hc_history, "Hill Climbing Convergence", "#1E88E5"),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with oc2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">🟢 Simulated Annealing</div>', unsafe_allow_html=True)
            for k, v in sa_metrics.items():
                st.metric(k, v)
            st.caption(f"Iterations: {sa_iters}  |  ✅ Escapes local optima")
            st.plotly_chart(convergence_fig(sa_history, "Simulated Annealing Convergence", "#26A69A"),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Scatter
        st.markdown('<div class="section-heading">Risk vs Return Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_scatter = go.Figure()
        for label, metrics, clr in [
            ("Hill Climbing", hc_metrics, "#1E88E5"),
            ("Simulated Annealing", sa_metrics, "#26A69A"),
        ]:
            fig_scatter.add_trace(go.Scatter(
                x=[metrics["Portfolio Risk"]], y=[metrics["Expected Return (%)"]],
                mode="markers+text",
                marker=dict(size=18, color=clr, line=dict(color="white", width=2)),
                text=[label], textposition="top center",
                textfont=dict(color="#94A3B8", size=11),
                name=label,
                hovertemplate=f"<b>{label}</b><br>Risk: %{{x:.3f}}<br>Return: %{{y:.1f}}%<extra></extra>",
            ))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(color="#64748B", gridcolor="rgba(255,255,255,0.06)", title="Risk"),
            yaxis=dict(color="#64748B", gridcolor="rgba(255,255,255,0.06)", title="Return (%)"),
            legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            height=280,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        history = hc_history if algo == "Hill Climbing" else sa_history
        n_iters = hc_iters   if algo == "Hill Climbing" else sa_iters
        metrics = hc_metrics if algo == "Hill Climbing" else sa_metrics
        clr     = "#1E88E5"  if algo == "Hill Climbing" else "#26A69A"

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for k, v in metrics.items():
            st.metric(k, v)
        st.caption(f"Total iterations: {n_iters}")
        if history:
            st.plotly_chart(convergence_fig(history, f"{algo} Convergence", clr),
                            use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB: ALL STOCKS
# ══════════════════════════════════════════════
elif active == "All Stocks":

    st.markdown('<div class="section-heading">📋 All Scored & Ranked Stocks</div>', unsafe_allow_html=True)

    display_cols = ["Symbol","Name","Sector","Total_Score",
                    "Growth_5yr","Volatility","Dividend_Yield","Momentum_Score"]
    show_df = scored_df[display_cols].copy()
    show_df["Growth_5yr"]     = (show_df["Growth_5yr"] * 100).round(1).astype(str) + "%"
    show_df["Dividend_Yield"] = show_df["Dividend_Yield"].round(1).astype(str) + "%"
    show_df["Total_Score"]    = show_df["Total_Score"].round(1)

    st.dataframe(
        show_df.style.background_gradient(subset=["Total_Score"], cmap="Blues"),
        use_container_width=True, hide_index=True, height=520
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  PSX AI Investment Advisory System &nbsp;•&nbsp; Work 1 Demo &nbsp;•&nbsp;
  Team A, B &amp; C &nbsp;•&nbsp; BS Computer Science, Semester 6
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .main-content
