"""
AI Investment Advisory System
BS Computer Science — 6th Semester AI Project
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    page_title="InvestAI — Advisory System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background:#F7F9FC; border-right:1px solid #E2E8F0; }
    .metric-box { background:#EBF4FF; border-radius:10px; padding:14px 18px;
                  border-left:4px solid #185FA5; margin-bottom:8px; }
    .metric-box h4 { color:#185FA5; margin:0 0 4px 0; font-size:13px; }
    .metric-box p  { color:#1a1a2e; margin:0; font-size:22px; font-weight:700; }
    .reason-box { background:#E6F1FB; border-radius:8px; padding:12px 16px; }
    .reason-box ul { margin:6px 0 0 16px; }
    .reason-box li { color:#0C447C; font-size:13px; margin-bottom:4px; }
    .footer { text-align:center; color:#9CA3AF; font-size:11px; padding:20px 0 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME HELPERS
# ─────────────────────────────────────────────
PALETTE   = ["#185FA5","#1D9E75","#EF9F27","#D4537E","#534AB7",
             "#47B8E0","#F06B6B","#6BCB77","#FFD166","#A78BFA"]
BG_COLOR  = "#FFFFFF"
GRID_CLR  = "#E2E8F0"
TEXT_CLR  = "#1a1a2e"

def _base_fig(w=6, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

def make_pie(labels, values):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG_COLOR)
    colors = PALETTE[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        textprops=dict(color=TEXT_CLR, fontsize=8)
    )
    for at in autotexts:
        at.set_fontsize(7); at.set_color("white")
    ax.set_title("Allocation by Symbol", fontsize=10, color=TEXT_CLR, pad=8)
    plt.tight_layout()
    return fig

def make_hbar(categories, values, title="", color=None):
    n = len(categories)
    fig, ax = _base_fig(w=6, h=max(2.4, n * 0.42 + 0.8))
    colors = color if color else PALETTE[:n]
    bars = ax.barh(categories, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8, color=TEXT_CLR)
    ax.set_xlabel("Allocation %", fontsize=8, color=TEXT_CLR)
    ax.set_title(title, fontsize=10, color=TEXT_CLR, pad=6)
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig

def make_score_hbar(components, scores, maxes):
    n = len(components)
    fig, ax = _base_fig(w=5.5, h=max(2.4, n * 0.42 + 0.8))
    colors = PALETTE[:n]
    bars = ax.barh(components, scores, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for i, mx in enumerate(maxes):
        ax.plot(mx, i, "|", color="#9CA3AF", markersize=12, markeredgewidth=1.5, zorder=4)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=8, color=TEXT_CLR)
    ax.set_xlabel("Score", fontsize=8, color=TEXT_CLR)
    ax.set_title("Heuristic Score Breakdown", fontsize=10, color=TEXT_CLR, pad=6)
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig

def make_line(history, title="", color="#185FA5"):
    fig, ax = _base_fig(w=5.5, h=2.8)
    ax.plot(history, color=color, linewidth=1.6, zorder=3)
    ax.set_xlabel("Iteration", fontsize=8, color=TEXT_CLR)
    ax.set_ylabel("Objective Score", fontsize=8, color=TEXT_CLR)
    ax.set_title(title, fontsize=10, color=TEXT_CLR, pad=6)
    plt.tight_layout()
    return fig

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
#  CACHED PIPELINE (so tab clicks are instant)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(preferred, excluded, risk, n_stocks, algo):
    scored = score_all(df, list(preferred), list(excluded), risk)
    if scored.empty:
        return scored, None, None, None
    top = scored.head(n_stocks).reset_index(drop=True)
    hc = sa = None
    if algo in ("Hill Climbing", "Both"):
        hc = hill_climbing(top, risk, n_stocks)
    if algo in ("Simulated Annealing", "Both"):
        sa = simulated_annealing(top, risk, n_stocks)
    return scored, top, hc, sa

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 InvestAI")
    st.markdown("---")
    st.markdown("### 👤 Investor Profile")

    amount = st.number_input("Investment Amount (PKR)",
        min_value=100_000, max_value=100_000_000,
        value=5_000_000, step=100_000)
    duration = st.slider("Duration (Years)", 1, 15, 5)
    target_return = st.slider("Target Annual Return (%)", 5, 50, 25)
    risk = st.radio("Risk Appetite", ["Low", "Medium", "High"], index=1, horizontal=True)

    st.markdown("---")
    st.markdown("### 🏭 Sector Preferences")
    preferred = st.multiselect("Preferred Sectors", options=ALL_SECTORS,
                               default=["Banking", "Energy"])
    excluded = st.multiselect("Excluded Sectors",
        options=[s for s in ALL_SECTORS if s not in preferred], default=[])

    st.markdown("---")
    n_stocks = st.slider("Portfolio Size (# Stocks)", 3, 12, 7)
    algo     = st.selectbox("Optimization Algorithm",
                            ["Hill Climbing", "Simulated Annealing", "Both"])

    run_btn = st.button("🚀 Generate Portfolio",
                        use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  PERSIST RUN STATE  (THE FIX)
# ─────────────────────────────────────────────
if run_btn:
    st.session_state.generated = True
    st.session_state.params = dict(
        amount=amount, risk=risk,
        preferred=tuple(preferred), excluded=tuple(excluded),
        n_stocks=n_stocks, algo=algo,
    )

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.title("🤖 AI Investment Advisory System")
st.caption("BS Computer Science — 6th Semester AI Project  |  PSX Stocks")

if not st.session_state.get("generated"):
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1** — Fill investor profile in sidebar")
    c2.info("**Step 2** — Select sectors & algorithm")
    c3.info("**Step 3** — Click Generate Portfolio")
    st.markdown("---")
    st.subheader("📋 Available Stock Database")
    st.dataframe(df, use_container_width=True, height=400)
    st.stop()

# Use saved params (immune to widget changes after generation)
p = st.session_state.params
amount    = p["amount"]
risk      = p["risk"]
preferred = p["preferred"]
excluded  = p["excluded"]
n_stocks  = p["n_stocks"]
algo      = p["algo"]

# ─────────────────────────────────────────────
#  RUN (cached — instant on tab switch)
# ─────────────────────────────────────────────
with st.spinner("⚙️ Scoring & optimizing..."):
    scored_df, top_stocks, hc_result, sa_result = run_pipeline(
        preferred, excluded, risk, n_stocks, algo
    )

if scored_df.empty:
    st.error("No stocks found after applying sector filters. Adjust your preferences.")
    st.stop()

def build_alloc(weights):
    out = []
    for i, row in top_stocks.iterrows():
        out.append({
            "Symbol": row["Symbol"], "Name": row["Name"], "Sector": row["Sector"],
            "Score": row["Total_Score"],
            "Allocation %": round(weights[i] * 100, 1),
            "Amount (PKR)": int(weights[i] * amount),
        })
    return out

if hc_result:
    hc_weights, hc_history, hc_iters = hc_result
    hc_metrics = portfolio_metrics(top_stocks, hc_weights)
    hc_alloc = build_alloc(hc_weights)
if sa_result:
    sa_weights, sa_history, sa_iters = sa_result
    sa_metrics = portfolio_metrics(top_stocks, sa_weights)
    sa_alloc = build_alloc(sa_weights)

if algo == "Hill Climbing":
    display_alloc, display_weights, display_metrics = hc_alloc, hc_weights, hc_metrics
elif algo == "Simulated Annealing":
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics
else:
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics

# ─────────────────────────────────────────────
#  NAVIGATION
# ─────────────────────────────────────────────
page = st.radio("", ["Portfolio", "AI Reasoning", "Optimization", "All Scored Stocks"],
                horizontal=True, key="nav_page")

# ── PORTFOLIO ─────────────────────────────────
if page == "Portfolio":
    st.subheader("Portfolio Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-box'><h4>Expected Return</h4><p>{display_metrics['Expected Return (%)']:.1f}%</p></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-box'><h4>Portfolio Risk</h4><p>{display_metrics['Portfolio Risk']:.3f}</p></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-box'><h4>Avg Dividend Yield</h4><p>{display_metrics['Avg Dividend Yield']:.1f}%</p></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-box'><h4>Sharpe Ratio</h4><p>{display_metrics['Sharpe-like Ratio']:.2f}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    col_pie, col_tbl = st.columns([1, 1])
    with col_pie:
        st.subheader("Allocation Pie Chart")
        pie_df = pd.DataFrame(display_alloc)
        show(make_pie(pie_df["Symbol"].tolist(), pie_df["Allocation %"].tolist()))
    with col_tbl:
        st.subheader("Recommended Stocks")
        alloc_df = pd.DataFrame(display_alloc)
        alloc_df["Amount (PKR)"] = alloc_df["Amount (PKR)"].apply(lambda x: f"₨ {x:,}")
        alloc_df["Score"] = alloc_df["Score"].apply(lambda x: f"{x:.0f}/100")
        st.dataframe(alloc_df, use_container_width=True, hide_index=True, height=320)

    st.markdown("---")
    st.subheader("Sector Distribution")
    sector_alloc = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    show(make_hbar(sector_alloc["Sector"].tolist(),
                   sector_alloc["Allocation %"].tolist(),
                   title="Allocation by Sector"))

# ── AI REASONING ──────────────────────────────
elif page == "AI Reasoning":
    st.subheader("AI Recommendation Reasoning")
    selected = st.selectbox("Select a stock to explain:",
        options=[f"{r['Symbol']} — {r['Name']}" for r in display_alloc],
        key="stock_explain")
    sym = selected.split(" — ")[0]
    stock_row = scored_df[scored_df["Symbol"] == sym].iloc[0]
    breakdown = stock_row["Breakdown"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='reason-box'>
        <b>Why {sym} was selected</b>
        <ul>
            <li>Growth (5yr): {stock_row['Growth_5yr']*100:.1f}%</li>
            <li>Volatility: {stock_row['Volatility']:.2f}</li>
            <li>Dividend Yield: {stock_row['Dividend_Yield']:.1f}%</li>
            <li>Sector: {stock_row['Sector']}</li>
            <li>Momentum Score: {stock_row['Momentum_Score']:.2f}</li>
            <li>AI Total Score: {stock_row['Total_Score']:.1f}/100</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        components = list(breakdown.keys())
        scores = list(breakdown.values())
        maxes = [30, 20, 15, 20, 15]
        show(make_score_hbar(components, scores, maxes))

# ── OPTIMIZATION ──────────────────────────────
elif page == "Optimization":
    st.subheader("Optimization Algorithm Results")
    if algo == "Both":
        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown("### Hill Climbing")
            for k, v in hc_metrics.items(): st.metric(k, v)
            show(make_line(hc_history, "Hill Climbing Convergence"))
        with oc2:
            st.markdown("### Simulated Annealing")
            for k, v in sa_metrics.items(): st.metric(k, v)
            show(make_line(sa_history, "Simulated Annealing Convergence"))
    else:
        metrics = hc_metrics if algo == "Hill Climbing" else sa_metrics
        history = hc_history if algo == "Hill Climbing" else sa_history
        clr = "#185FA5" if algo == "Hill Climbing" else "#1D9E75"
        for k, v in metrics.items(): st.metric(k, v)
        show(make_line(history, f"{algo} Convergence", clr))

# ── ALL SCORED STOCKS ─────────────────────────
elif page == "All Scored Stocks":
    st.subheader("All Scored & Ranked Stocks")
    display_cols = ["Symbol","Name","Sector","Total_Score",
                    "Growth_5yr","Volatility","Dividend_Yield","Momentum_Score"]
    show_df = scored_df[display_cols].copy()
    show_df["Growth_5yr"] = (show_df["Growth_5yr"] * 100).round(1).astype(str) + "%"
    show_df["Dividend_Yield"] = show_df["Dividend_Yield"].round(1).astype(str) + "%"
    st.dataframe(show_df.style.background_gradient(subset=["Total_Score"], cmap="Blues"),
                 use_container_width=True, hide_index=True, height=500)

# ─────────────────────────────────────────────
st.markdown("<div class='footer'>AI Investment Advisory System | BS CS 6th Semester | PSX Stocks — Simulated Data</div>", unsafe_allow_html=True)
