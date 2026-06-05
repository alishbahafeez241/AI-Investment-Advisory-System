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
    page_title="PSX AI Investment Advisory System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark navy sidebar, clean cards
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Sidebar: dark navy ── */
[data-testid="stSidebar"] {
    background: #0F1B2D !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .sidebar-header { color: #F1F5F9 !important; }

/* Sidebar section labels */
[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8 !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }

/* Sidebar number input / slider labels */
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 12px !important; }

/* Sidebar radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #CBD5E1 !important; font-size: 13px !important; text-transform: none !important; letter-spacing: 0 !important; }

/* Sidebar multiselect tags */
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #1E3A5F !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #93C5FD !important; }

/* Sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="select"] { background: #162032 !important; border-color: #2D4A6A !important; color: #E2E8F0 !important; }

/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #3B82F6 !important; }

/* Run button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D9E75) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 0 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #15866A) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
}

/* ── Top header bar ── */
.page-header {
    background: linear-gradient(90deg, #0F1B2D 0%, #162032 100%);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid #1E3A5F;
}
.page-header h1 { color: #F1F5F9; font-size: 22px; font-weight: 700; margin: 0; }
.page-header span { color: #64748B; font-size: 13px; }

/* ── Tab nav override ── */
[data-testid="stRadio"] > div {
    background: #F1F5F9;
    border-radius: 10px;
    padding: 4px;
    gap: 2px !important;
    display: flex;
}
[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    cursor: pointer;
    transition: all 0.2s;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: white !important;
    color: #0F1B2D !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563EB, #1D9E75);
}
.kpi-card .kpi-label { color: #64748B; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-card .kpi-value { color: #0F1B2D; font-size: 28px; font-weight: 700; font-family: 'DM Mono', monospace; line-height: 1; }
.kpi-card .kpi-badge { display: inline-block; margin-top: 6px; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 20px; }
.badge-green { background: #D1FAE5; color: #065F46; }
.badge-blue  { background: #DBEAFE; color: #1E40AF; }
.badge-amber { background: #FEF3C7; color: #92400E; }

/* ── Section cards ── */
.section-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.section-card h3 { color: #0F1B2D; font-size: 15px; font-weight: 600; margin: 0 0 16px 0; }

/* ── Stock table badges ── */
.badge-top  { background: #DBEAFE; color: #1E40AF; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-rec  { background: #D1FAE5; color: #065F46; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-sel  { background: #F0FDF4; color: #166534; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }

/* ── AI Reasoning cards ── */
.reason-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.reason-card .stock-symbol { font-size: 22px; font-weight: 800; color: #0F1B2D; }
.reason-card .stock-name   { font-size: 12px; color: #64748B; margin-bottom: 4px; }
.reason-card .sector-pill  {
    display: inline-block;
    background: #EFF6FF;
    color: #2563EB;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.score-bar-wrap { margin-bottom: 6px; }
.score-bar-label { font-size: 11px; color: #64748B; margin-bottom: 3px; display: flex; justify-content: space-between; }
.score-bar-bg { background: #F1F5F9; border-radius: 4px; height: 6px; }
.score-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #2563EB, #1D9E75); }
.stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 12px; }
.stat-row .stat-label { color: #64748B; }
.stat-row .stat-val   { font-weight: 600; color: #0F1B2D; font-family: 'DM Mono', monospace; }
.risk-low  { color: #16A34A !important; }
.risk-med  { color: #D97706 !important; }
.risk-high { color: #DC2626 !important; }
.alloc-tag { background: #EFF6FF; color: #2563EB; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px; }

/* ── Score breakdown bar (AI reasoning page) ── */
.comp-score-row { margin-bottom: 10px; }
.comp-score-row .cs-label { font-size: 12px; color: #374151; font-weight: 500; margin-bottom: 3px; display: flex; justify-content: space-between; }
.comp-score-row .cs-bg { background: #F1F5F9; border-radius: 4px; height: 8px; }
.comp-score-row .cs-fill { height: 8px; border-radius: 4px; }

/* ── Optimization page ── */
.algo-metric { text-align: center; background: white; border-radius: 12px; border: 1px solid #E2E8F0; padding: 20px; }
.algo-metric .am-label { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.algo-metric .am-value { font-size: 32px; font-weight: 800; color: #0F1B2D; font-family: 'DM Mono', monospace; }
.algo-metric .am-sub   { font-size: 11px; color: #94A3B8; margin-top: 4px; }

/* ── Main bg ── */
.stApp { background: #F8FAFC !important; }

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; }

/* ── Footer ── */
.footer { text-align:center; color:#94A3B8; font-size:11px; padding:24px 0 8px; }

/* ── Info boxes on landing ── */
.info-step { background: white; border-radius: 12px; border: 1px solid #E2E8F0; padding: 20px; text-align: center; }
.info-step .step-num { background: #EFF6FF; color: #2563EB; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; margin: 0 auto 10px; }
.info-step p { color: #64748B; font-size: 13px; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME HELPERS
# ─────────────────────────────────────────────
PALETTE   = ["#2563EB","#1D9E75","#EF9F27","#D4537E","#534AB7",
             "#47B8E0","#F06B6B","#6BCB77","#FFD166","#A78BFA"]
BG_COLOR  = "#FFFFFF"
GRID_CLR  = "#E2E8F0"
TEXT_CLR  = "#0F1B2D"

def _base_fig(w=6, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="#64748B", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

def make_pie(labels, values):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor(BG_COLOR)
    colors = PALETTE[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT_CLR, fontsize=8)
    )
    for at in autotexts:
        at.set_fontsize(7); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Allocation by Symbol", fontsize=10, color=TEXT_CLR, pad=8, fontweight="600")
    plt.tight_layout()
    return fig

def make_hbar(categories, values, title="", color=None):
    n = len(categories)
    fig, ax = _base_fig(w=6, h=max(2.4, n * 0.48 + 0.8))
    colors = color if color else PALETTE[:n]
    bars = ax.barh(categories, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3, height=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8, color=TEXT_CLR, fontweight="600")
    ax.set_xlabel("Allocation %", fontsize=8, color="#64748B")
    ax.set_title(title, fontsize=10, color=TEXT_CLR, pad=6, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig

def make_score_hbar(components, scores, maxes):
    n = len(components)
    fig, ax = _base_fig(w=5.5, h=max(2.4, n * 0.48 + 0.8))
    colors = PALETTE[:n]
    bars = ax.barh(components, scores, color=colors, edgecolor="white", linewidth=0.8, zorder=3, height=0.55)
    for i, mx in enumerate(maxes):
        ax.plot(mx, i, "|", color="#CBD5E1", markersize=14, markeredgewidth=2, zorder=4)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=8, color=TEXT_CLR, fontweight="600")
    ax.set_xlabel("Score", fontsize=8, color="#64748B")
    ax.set_title("Score Component Breakdown", fontsize=10, color=TEXT_CLR, pad=6, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig

def make_line(history, title="", color="#2563EB"):
    fig, ax = _base_fig(w=5.5, h=3.0)
    ax.plot(history, color=color, linewidth=2, zorder=3)
    ax.fill_between(range(len(history)), history, alpha=0.08, color=color)
    ax.set_xlabel("Iteration", fontsize=8, color="#64748B")
    ax.set_ylabel("Objective Score", fontsize=8, color="#64748B")
    ax.set_title(title, fontsize=10, color=TEXT_CLR, pad=6, fontweight="600")
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
#  CACHED PIPELINE
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
#  SIDEBAR — dark navy
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Investor Settings")
    st.markdown("---")

    amount = st.number_input("Investment Amount (PKR)",
        min_value=100_000, max_value=100_000_000,
        value=5_000_000, step=100_000)
    duration = st.slider("Duration (Years)", 1, 15, 5)
    target_return = st.slider("Target Annual Return (%)", 5, 50, 20)
    risk = st.radio("Risk Appetite", ["Low", "Medium", "High"], index=1, horizontal=True)

    st.markdown("---")
    preferred = st.multiselect("Preferred Sectors", options=ALL_SECTORS,
                               default=["Banking", "Energy"])
    excluded = st.multiselect("Excluded Sectors",
        options=[s for s in ALL_SECTORS if s not in preferred], default=[])

    st.markdown("---")
    n_stocks = st.slider("Portfolio Size", 3, 12, 5)
    algo     = st.selectbox("Algorithm",
                            ["Hill Climbing", "Simulated Annealing", "Both"])

    st.markdown("&nbsp;")
    run_btn = st.button("▶  Run Analysis",
                        use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  PERSIST RUN STATE
# ─────────────────────────────────────────────
if run_btn:
    st.session_state.generated = True
    st.session_state.params = dict(
        amount=amount, risk=risk,
        preferred=tuple(preferred), excluded=tuple(excluded),
        n_stocks=n_stocks, algo=algo,
    )

# ─────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <span style="font-size:28px;">📈</span>
  <div>
    <h1 style="margin:0;color:#F1F5F9;">PSX AI Investment Advisory System</h1>
    <span style="color:#64748B;font-size:12px;">BS Computer Science — 6th Semester AI Project &nbsp;·&nbsp; PSX Stocks</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LANDING
# ─────────────────────────────────────────────
if not st.session_state.get("generated"):
    c1, c2, c3 = st.columns(3)
    for col, step, desc in [
        (c1, "1", "Fill your investor profile in the settings panel"),
        (c2, "2", "Select preferred sectors and optimization algorithm"),
        (c3, "3", "Click Run Analysis to generate your AI portfolio"),
    ]:
        col.markdown(f"""
        <div class="info-step">
          <div class="step-num">{step}</div>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Available Stock Database")
    st.dataframe(df, use_container_width=True, height=400)
    st.stop()

# ─────────────────────────────────────────────
#  LOAD SAVED PARAMS
# ─────────────────────────────────────────────
p = st.session_state.params
amount    = p["amount"]
risk      = p["risk"]
preferred = p["preferred"]
excluded  = p["excluded"]
n_stocks  = p["n_stocks"]
algo      = p["algo"]

# ─────────────────────────────────────────────
#  RUN PIPELINE
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
#  NAV TABS
# ─────────────────────────────────────────────
page = st.radio("", ["📊 Portfolio", "🤖 AI Reasoning", "⚡ Optimization", "📋 All Stocks"],
                horizontal=True, key="nav_page")

st.markdown("&nbsp;")

# ══════════════════════════════════════════════
#  PORTFOLIO TAB
# ══════════════════════════════════════════════
if page == "📊 Portfolio":

    ret  = display_metrics['Expected Return (%)']
    vrisk = display_metrics['Portfolio Risk']
    div  = display_metrics['Avg Dividend Yield']
    sharpe = display_metrics['Sharpe-like Ratio']

    ret_badge  = ("Above Target", "badge-green") if ret >= target_return else ("Below Target", "badge-amber")
    risk_badge = ("Within tolerance", "badge-blue")
    div_badge  = ("Annual income", "badge-green")
    sharpe_lbl = "Excellent" if sharpe >= 1.5 else ("Good" if sharpe >= 1.0 else "Fair")
    sharpe_cls = "badge-green" if sharpe >= 1.5 else "badge-blue"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Expected Return</div>
        <div class="kpi-value">{ret:.1f}%</div>
        <span class="kpi-badge {ret_badge[1]}">▲ {ret_badge[0]}</span>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Portfolio Risk</div>
        <div class="kpi-value">{vrisk:.3f}</div>
        <span class="kpi-badge {risk_badge[1]}">{risk_badge[0]}</span>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Dividend Yield</div>
        <div class="kpi-value">{div:.1f}%</div>
        <span class="kpi-badge {div_badge[1]}">{div_badge[0]}</span>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value">{sharpe:.2f}</div>
        <span class="kpi-badge {sharpe_cls}">{sharpe_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_pie, col_tbl = st.columns([1, 1], gap="large")
    with col_pie:
        st.markdown('<div class="section-card"><h3>Portfolio Allocation</h3>', unsafe_allow_html=True)
        pie_df = pd.DataFrame(display_alloc)
        show(make_pie(pie_df["Symbol"].tolist(), pie_df["Allocation %"].tolist()))
        # Legend dots
        legend_html = " &nbsp; ".join(
            f'<span style="color:{PALETTE[i]};font-weight:700;">●</span> <span style="font-size:12px;color:#374151;">{r["Symbol"]} — {r["Allocation %"]}%</span>'
            for i, r in enumerate(display_alloc)
        )
        st.markdown(f'<div style="margin-top:8px;">{legend_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tbl:
        st.markdown('<div class="section-card"><h3>Recommended Stocks</h3>', unsafe_allow_html=True)
        alloc_df = pd.DataFrame(display_alloc)
        alloc_df_show = alloc_df.copy()
        alloc_df_show["Amount (PKR)"] = alloc_df_show["Amount (PKR)"].apply(lambda x: f"₨ {x:,}")
        alloc_df_show["Score"] = alloc_df_show["Score"].apply(lambda x: f"{x:.0f}/100")
        alloc_df_show["Alloc %"] = alloc_df_show["Allocation %"].apply(lambda x: f"{x}%")
        st.dataframe(alloc_df_show[["Symbol","Sector","Score","Alloc %","Amount (PKR)"]],
                     use_container_width=True, hide_index=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Allocation by Sector</h3>', unsafe_allow_html=True)
    sector_alloc = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    show(make_hbar(sector_alloc["Sector"].tolist(),
                   sector_alloc["Allocation %"].tolist(),
                   title="Sector Distribution"))
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  AI REASONING TAB
# ══════════════════════════════════════════════
elif page == "🤖 AI Reasoning":
    st.markdown("### AI Recommendation Reasoning")

    COMP_COLORS = ["#2563EB","#1D9E75","#EF9F27","#D4537E","#534AB7"]
    COMP_MAXES  = [30, 20, 15, 20, 15]

    # Show top 3 stocks as cards
    top3 = display_alloc[:3]
    cols = st.columns(len(top3), gap="large")
    for col, rec in zip(cols, top3):
        sym = rec["Symbol"]
        row = scored_df[scored_df["Symbol"] == sym].iloc[0]
        breakdown = row["Breakdown"]
        vol = row["Volatility"]
        risk_cls = "risk-low" if vol < 0.15 else ("risk-med" if vol < 0.25 else "risk-high")
        risk_lbl = "Low" if vol < 0.15 else ("Medium" if vol < 0.25 else "High")

        comp_bars = ""
        for (comp, score), mx, clr in zip(breakdown.items(), COMP_MAXES, COMP_COLORS):
            pct = min(score / mx * 100, 100)
            comp_bars += f"""
            <div class="comp-score-row">
              <div class="cs-label"><span>{comp}</span><span style="font-family:'DM Mono',monospace;">{score:.1f}</span></div>
              <div class="cs-bg"><div class="cs-fill" style="width:{pct}%;background:{clr};"></div></div>
            </div>"""

        col.markdown(f"""
        <div class="reason-card">
          <div class="stock-name">{row['Name']}</div>
          <div class="stock-symbol">{sym}</div>
          <div><span class="sector-pill">{row['Sector']}</span></div>
          <div style="margin-bottom:14px;">
            <div style="font-size:11px;color:#64748B;margin-bottom:4px;">Composite Score</div>
            {comp_bars}
          </div>
          <div class="stat-row"><span class="stat-label">5yr Growth</span><span class="stat-val">{row['Growth_5yr']*100:.0f}%</span></div>
          <div class="stat-row"><span class="stat-label">Volatility</span><span class="stat-val {risk_cls}">{risk_lbl}</span></div>
          <div class="stat-row"><span class="stat-label">Dividend Yield</span><span class="stat-val">{row['Dividend_Yield']:.1f}%</span></div>
          <div class="stat-row" style="border:none;"><span class="stat-label">Momentum</span><span class="stat-val">{row['Momentum_Score']:.2f}</span></div>
          <div style="margin-top:12px;font-size:11px;color:#64748B;">
            {sym} ranks highly due to strong 5-year growth and consistent dividend payout; sector preference bonus applied.
          </div>
          <div style="margin-top:8px;"><span class="alloc-tag">Recommended allocation {rec['Allocation %']}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # Detailed breakdown for any stock
    st.markdown("---")
    st.markdown("#### Score Component Breakdown — Detailed View")
    selected = st.selectbox("Select a stock:",
        options=[f"{r['Symbol']} — {r['Name']}" for r in display_alloc],
        key="stock_explain")
    sym = selected.split(" — ")[0]
    stock_row = scored_df[scored_df["Symbol"] == sym].iloc[0]
    breakdown = stock_row["Breakdown"]

    components = list(breakdown.keys())
    scores     = list(breakdown.values())
    show(make_score_hbar(components, scores, COMP_MAXES))
    st.caption("AI Reasoning Tab — Scores computed by scoring_engine using 5 weighted components")

# ══════════════════════════════════════════════
#  OPTIMIZATION TAB
# ══════════════════════════════════════════════
elif page == "⚡ Optimization":
    st.markdown("### Optimization Algorithm Results")

    if algo == "Both":
        # Convergence curves side by side
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-card"><h3>Convergence Curves</h3>', unsafe_allow_html=True)
            fig, ax = _base_fig(w=5.5, h=3.0)
            ax.plot(hc_history, color="#2563EB", linewidth=2, label="Hill Climbing", zorder=3)
            ax.plot(sa_history, color="#1D9E75", linewidth=2, linestyle="--", label="Simulated Annealing", zorder=3)
            ax.fill_between(range(len(hc_history)), hc_history, alpha=0.07, color="#2563EB")
            ax.fill_between(range(len(sa_history)), sa_history, alpha=0.07, color="#1D9E75")
            ax.set_xlabel("Iterations", fontsize=8, color="#64748B")
            ax.set_ylabel("Objective Score", fontsize=8, color="#64748B")
            ax.legend(fontsize=8, framealpha=0.8)
            plt.tight_layout()
            show(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card"><h3>Algorithm Comparison</h3>', unsafe_allow_html=True)
            hc_best = max(hc_history)
            sa_best = max(sa_history)
            st.markdown(f"""
            <div style="display:flex;gap:16px;margin-bottom:16px;">
              <div class="algo-metric" style="flex:1;">
                <div class="am-label">HC Best Score</div>
                <div class="am-value" style="color:#2563EB;">{hc_best:.3f}</div>
                <div class="am-sub">Iterations: {hc_iters}</div>
              </div>
              <div class="algo-metric" style="flex:1;">
                <div class="am-label">SA Best Score</div>
                <div class="am-value" style="color:#1D9E75;">{sa_best:.3f}</div>
                <div class="am-sub">Temp: 1000 → 0.01</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            winner = "SA" if sa_best > hc_best else "HC"
            st.info(f"**{winner}** found a marginally better portfolio. Both are appropriate for this portfolio size. Recommendation: use SA for large portfolios (N > 8 stocks).")
            st.markdown('</div>', unsafe_allow_html=True)

        # Metrics below
        mc1, mc2 = st.columns(2, gap="large")
        with mc1:
            st.markdown("**Hill Climbing Metrics**")
            for k, v in hc_metrics.items():
                st.metric(k, v)
        with mc2:
            st.markdown("**Simulated Annealing Metrics**")
            for k, v in sa_metrics.items():
                st.metric(k, v)

    else:
        metrics = hc_metrics if algo == "Hill Climbing" else sa_metrics
        history = hc_history if algo == "Hill Climbing" else sa_history
        clr = "#2563EB" if algo == "Hill Climbing" else "#1D9E75"

        best_score = max(history)
        iters = hc_iters if algo == "Hill Climbing" else sa_iters

        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:20px;">
          <div class="algo-metric" style="min-width:160px;">
            <div class="am-label">Best Score</div>
            <div class="am-value" style="color:{clr};">{best_score:.3f}</div>
            <div class="am-sub">Iterations: {iters}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        show(make_line(history, f"{algo} Convergence", clr))
        st.markdown("---")
        for k, v in metrics.items():
            st.metric(k, v)

    st.caption(f"Optimization Tab — {algo} convergence comparison")

# ══════════════════════════════════════════════
#  ALL STOCKS TAB
# ══════════════════════════════════════════════
elif page == "📋 All Stocks":
    total = len(scored_df)
    st.markdown(f"### All {total} Stocks — Ranked by Score")
    st.caption(f"Showing top 10 of {total}")

    display_cols = ["Symbol","Name","Sector","Total_Score",
                    "Growth_5yr","Volatility","Dividend_Yield","Momentum_Score"]
    show_df = scored_df[display_cols].copy()
    show_df["Growth_5yr"]     = (show_df["Growth_5yr"] * 100).round(1).astype(str) + "%"
    show_df["Dividend_Yield"] = show_df["Dividend_Yield"].round(1).astype(str) + "%"
    show_df["Volatility"]     = show_df["Volatility"].round(2)
    show_df["Total_Score"]    = show_df["Total_Score"].round(0).astype(int)
    show_df = show_df.rename(columns={
        "Total_Score":"Score","Growth_5yr":"Growth","Dividend_Yield":"Div Yield","Momentum_Score":"Momentum"
    })

    selected_syms = set(r["Symbol"] for r in display_alloc)
    top_syms = set(scored_df.head(1)["Symbol"].tolist())

    st.dataframe(
        show_df.style.background_gradient(subset=["Score"], cmap="Blues"),
        use_container_width=True, hide_index=True, height=500
    )
    st.caption("All Stocks Table — Scores calculated using 5-component heuristic function with risk appetite adjustment")

# ─────────────────────────────────────────────
st.markdown("<div class='footer'>AI Investment Advisory System &nbsp;·&nbsp; BS CS 6th Semester &nbsp;·&nbsp; PSX Stocks — Simulated Data</div>", unsafe_allow_html=True)
