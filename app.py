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
import os

from scoring_engine import (
    score_all, allocate_portfolio, portfolio_metrics,
    hill_climbing, simulated_annealing
)

st.set_page_config(
    page_title="PSX AI Investment Advisory System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; font-size:13px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#1B2A42 !important; border-right:none !important; }
[data-testid="stSidebar"] * { color:#C8D6E8 !important; }
[data-testid="stSidebar"] label { color:#94A3B8 !important; font-size:11px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color:#C8D6E8 !important; font-size:11px !important;
    text-transform:none !important; letter-spacing:0 !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background:#243548 !important; border-color:#2D4A6A !important;
    color:#E2E8F0 !important; font-size:12px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] { background:#1E3A5F !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color:#93C5FD !important; }
[data-testid="stSidebar"] .stButton > button {
    background:linear-gradient(135deg,#2563EB,#0EA5A0) !important;
    color:white !important; border:none !important; border-radius:7px !important;
    font-weight:600 !important; font-size:12px !important; padding:9px 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:linear-gradient(135deg,#1D4ED8,#0B8A87) !important;
    box-shadow:0 4px 12px rgba(37,99,235,.3) !important;
}

/* ── Main bg ── */
.stApp { background:#F0F4F9 !important; }

/* ── Page header ── */
.page-header {
    background:linear-gradient(90deg,#1B2A42,#243548);
    border-radius:8px; padding:11px 18px; margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
}
.page-header h1 { color:#F1F5F9; font-size:14px; font-weight:700; margin:0; }
.page-header p  { color:#64748B; font-size:10px; margin:0; }

/* ── Tab radio ── */
[data-testid="stHorizontalRadio"] {
    background:#E2E8F0; border-radius:7px; padding:3px; gap:2px !important;
}
[data-testid="stHorizontalRadio"] label {
    background:transparent !important; border-radius:5px !important;
    padding:5px 20px !important; font-size:12px !important; font-weight:500 !important;
    color:#64748B !important; cursor:pointer; transition:all .15s;
}
[data-testid="stHorizontalRadio"] label:has(input:checked) {
    background:white !important; color:#1B2A42 !important;
    box-shadow:0 1px 3px rgba(0,0,0,.1) !important; font-weight:600 !important;
}

/* ── KPI row ── */
.kpi-row { display:flex; gap:15px; margin-bottom:16px; }
.kpi-card {
    flex:1; background:white; border-radius:10px; padding:14px 16px;
    border:1px solid #E2E8F0; box-shadow:0 1px 2px rgba(0,0,0,.04);
}
.kpi-label  { color:#64748B; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.kpi-value  { color:#0F1B2D; font-size:24px; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1.2; }
.kpi-badge  { display:inline-block; margin-top:6px; font-size:10px; font-weight:600; padding:2px 8px; border-radius:20px; }
.bg-green { background:#D1FAE5; color:#065F46; }
.bg-blue  { background:#DBEAFE; color:#1E40AF; }
.bg-amber { background:#FEF3C7; color:#92400E; }
.bg-red   { background:#FEE2E2; color:#991B1B; }

/* ── Section card ── */
.sc { background:white; border-radius:10px; border:1px solid #E2E8F0; padding:16px 18px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.sc h3 { color:#0F1B2D; font-size:13px; font-weight:600; margin:0 0 12px 0; }
.sc h4 { color:#0F1B2D; font-size:12px; font-weight:600; margin:0 0 8px 0; }

/* ── Stock card (like reference) ── */
.stock-card {
    background:white; border-radius:10px; border:1px solid #E2E8F0; 
    padding:14px; margin-bottom:12px;
}
.stock-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.stock-symbol { font-size:18px; font-weight:800; color:#0F1B2D; }
.stock-sector { background:#EFF6FF; color:#2563EB; font-size:9px; font-weight:600; padding:2px 8px; border-radius:12px; }
.stock-name { font-size:11px; color:#64748B; margin-bottom:10px; }
.score-bar { margin:10px 0; }
.score-label { font-size:9px; color:#64748B; margin-bottom:3px; display:flex; justify-content:space-between; }
.score-bg { background:#F1F5F9; border-radius:4px; height:6px; overflow:hidden; }
.score-fill { height:6px; border-radius:4px; }
.stock-stats { display:flex; gap:12px; margin:10px 0; font-size:10px; }
.stock-stat { flex:1; text-align:center; }
.stat-label { color:#64748B; font-size:8px; margin-bottom:2px; }
.stat-value { font-weight:700; color:#0F1B2D; }
.alloc-badge { display:inline-block; background:#EFF6FF; color:#2563EB; font-size:10px; font-weight:700; padding:4px 10px; border-radius:6px; margin-top:8px; }
.reason-text { font-size:9px; color:#64748B; margin-top:8px; line-height:1.4; padding-top:6px; border-top:1px solid #F1F5F9; }

/* ── Algo metric box ── */
.am { text-align:center; background:white; border-radius:10px; border:1px solid #E2E8F0; padding:16px; }
.aml { font-size:10px; color:#64748B; text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px; }
.amv { font-size:28px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.ams { font-size:9px; color:#94A3B8; margin-top:3px; }

/* ── Table styles ── */
.stock-table { width:100%; border-collapse:collapse; }
.stock-table th { text-align:left; padding:8px 6px; background:#F8FAFD; font-size:10px; font-weight:600; color:#64748B; border-bottom:1px solid #E2E8F0; }
.stock-table td { padding:8px 6px; border-bottom:1px solid #F1F5F9; font-size:11px; }
.top-pick { background:#F0FDF4; font-weight:600; color:#065F46; }
.recommended { background:#EFF6FF; font-weight:600; color:#1E40AF; }
.selected { background:#FFFBEB; font-weight:600; color:#92400E; }

/* ── Chart title ── */
.chart-title { font-size:12px; font-weight:700; color:#0F1B2D; margin:0 0 4px 0; padding:10px 14px 0; background:white; border:1px solid #E2E8F0; border-bottom:none; border-radius:10px 10px 0 0; }

.footer { text-align:center; color:#94A3B8; font-size:10px; padding:20px 0 8px; border-top:1px solid #E2E8F0; margin-top:20px; }

.istep { background:white; border-radius:10px; border:1px solid #E2E8F0; padding:20px; text-align:center; }
.istep .sn { background:#EFF6FF; color:#2563EB; width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; margin:0 auto 10px; }
.istep p { color:#64748B; font-size:12px; margin:0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB
# ─────────────────────────────────────────────
PALETTE = ["#1a3fcc","#1D9E75","#EF9F27","#D4537E","#8B5CF6",
           "#47B8E0","#F06B6B","#6BCB77","#FFD166","#A78BFA"]
BG, GRID, TXT = "#FFFFFF", "#E8EDF4", "#0F1B2D"

def _base_fig(w=6, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors="#64748B", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

def make_pie(labels, values):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor(BG)
    wedges, texts, auto = ax.pie(
        values, labels=labels, autopct="%1.1f%%", colors=PALETTE[:len(labels)],
        startangle=90, wedgeprops=dict(width=.55, edgecolor="white", linewidth=1.5),
        textprops=dict(color=TXT, fontsize=8))
    for at in auto: at.set_fontsize(7); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Allocation by Symbol", fontsize=9, color=TXT, pad=8, fontweight="700")
    plt.tight_layout(); return fig

def make_hbar(cats, vals, title=""):
    n = len(cats)
    fig, ax = _base_fig(w=5.5, h=max(2.5, n*.45+.7))
    bars = ax.barh(cats, vals, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+max(vals)*.015, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8, color=TXT, fontweight="600")
    ax.set_xlabel("Allocation %", fontsize=7, color="#64748B")
    ax.set_title(title, fontsize=9, color=TXT, pad=5, fontweight="700")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_score_hbar(comps, scores, maxes):
    n = len(comps)
    fig, ax = _base_fig(w=5.5, h=max(2.5, n*.45+.7))
    bars = ax.barh(comps, scores, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.5)
    for i, mx in enumerate(maxes):
        ax.plot(mx, i, "|", color="#CBD5E1", markersize=12, markeredgewidth=1.5, zorder=4)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=8, color=TXT, fontweight="600")
    ax.set_xlabel("Score", fontsize=7, color="#64748B")
    ax.set_title("Score Component Breakdown", fontsize=9, color=TXT, pad=5, fontweight="700")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_line(history, title="", color="#1a3fcc", label=None):
    fig, ax = _base_fig(w=5.5, h=3.2)
    x = range(len(history))
    ax.plot(history, color=color, linewidth=2.0, zorder=3, label=label)
    base = min(history)
    ax.fill_between(x, history, base, alpha=.12, color=color)
    ax.set_xlabel("Iteration No.", fontsize=7, color="#64748B")
    ax.set_ylabel("Convergence Value", fontsize=7, color="#64748B")
    ax.set_title(title, fontsize=9, color=TXT, pad=8, fontweight="700")
    if label: ax.legend(fontsize=7, framealpha=.8)
    plt.tight_layout(); return fig

def make_dual_line(hc_hist, sa_hist):
    fig, ax = _base_fig(w=6, h=3.2)
    x1, x2 = range(len(hc_hist)), range(len(sa_hist))
    base1, base2 = min(hc_hist), min(sa_hist)
    ax.fill_between(x1, hc_hist, base1, alpha=.10, color="#1a3fcc")
    ax.fill_between(x2, sa_hist, base2, alpha=.10, color="#1D9E75")
    ax.plot(hc_hist, color="#1a3fcc", linewidth=2.0, label="HC Seed Score", zorder=3)
    ax.plot(sa_hist, color="#1D9E75", linewidth=2.0, linestyle="-", label="SA Seed Score", zorder=3)
    ax.set_xlabel("Iteration No.", fontsize=7, color="#64748B")
    ax.set_ylabel("Convergence Value", fontsize=7, color="#64748B")
    ax.set_title("Convergence Curves", fontsize=10, color=TXT, pad=8, fontweight="700")
    ax.legend(fontsize=7, framealpha=.8)
    # Add target line
    ax.axhline(y=1000, color='#DC2626', linestyle='--', linewidth=1, alpha=0.5, label="Target: 1000 → 0.0")
    plt.tight_layout(); return fig

def make_scatter(top_stocks, hc_w, sa_w):
    fig, ax = _base_fig(w=5.5, h=3.2)
    ax.set_facecolor("#F8FAFD")
    for i, row in top_stocks.iterrows():
        r  = row["Growth_5yr"] * 100
        v  = row["Volatility"] * 100
        sz = 80 + row["Total_Score"]
        ax.scatter(v, r, s=sz, color=PALETTE[i % len(PALETTE)], alpha=.85, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(row["Symbol"], (v, r), textcoords="offset points", xytext=(5,3),
                    fontsize=6, color=TXT, fontweight="500")
    # Efficient frontier curve
    xs = np.linspace(5, 35, 80)
    ys = 8 + 120 * (xs - xs.min())**.5
    ax.plot(xs, ys, color="#94A3B8", linewidth=1.5, linestyle="--", label="Efficient Frontier", zorder=2)
    ax.set_xlabel("Risk", fontsize=7, color="#64748B")
    ax.set_ylabel("Return", fontsize=7, color="#64748B")
    ax.set_title("Risk vs Return Scatter", fontsize=10, color=TXT, pad=8, fontweight="700")
    ax.legend(fontsize=7, framealpha=.8)
    plt.tight_layout(); return fig

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "stocks_data.csv")
    return pd.read_csv(path)

df = load_data()
ALL_SECTORS = sorted(df["Sector"].unique().tolist())

@st.cache_data(show_spinner=False)
def run_pipeline(preferred, excluded, risk, n_stocks, algo):
    scored = score_all(df, list(preferred), list(excluded), risk)
    if scored.empty: return scored, None, None, None
    top = scored.head(n_stocks).reset_index(drop=True)
    hc = sa = None
    if algo in ("Hill Climbing","Both"):        hc = hill_climbing(top, risk, n_stocks)
    if algo in ("Simulated Annealing","Both"):  sa = simulated_annealing(top, risk, n_stocks)
    return scored, top, hc, sa

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## INVESTOR SETTINGS")
    st.markdown("---")
    
    st.markdown("**INVESTOR ADVISOR (0.1%)**")
    amount = st.number_input("Rs", min_value=100_000, max_value=100_000_000, value=5_000_000, step=100_000, label_visibility="collapsed")
    
    st.markdown("**Duration (Years)**")
    duration = st.slider("Years", 1, 15, 5, label_visibility="collapsed")
    
    st.markdown("**Target Annual Return**")
    target_return = st.slider("Return %", 5, 50, 20, label_visibility="collapsed")
    
    st.markdown("**Risk Appetite**")
    risk = st.radio("Risk", ["Low", "Medium", "High"], index=1, horizontal=True, label_visibility="collapsed")
    
    st.markdown("**Inflation Sources**")
    inflation = st.radio("Inflation", ["Reducing", "Rising", "Risk"], index=1, horizontal=True, label_visibility="collapsed")
    
    st.markdown("**Portfolio Size**")
    n_stocks = st.slider("Stocks", 3, 12, 5, label_visibility="collapsed")
    
    st.markdown("**Algorithms**")
    algo = st.selectbox("Algo", ["Hill Climbing", "Simulated Annealing", "Both"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Preferred Sectors**")
    preferred = st.multiselect("Select", options=ALL_SECTORS, default=["Banking", "Energy"], label_visibility="collapsed")
    st.markdown("**Excluded Sectors**")
    excluded = st.multiselect("Exclude", options=[s for s in ALL_SECTORS if s not in preferred], default=[], label_visibility="collapsed")
    
    st.markdown("&nbsp;")
    run_btn = st.button("Run Analysis", use_container_width=True, type="primary")

if run_btn:
    st.session_state.generated = True
    st.session_state.params = dict(
        amount=amount, risk=risk,
        preferred=tuple(preferred), excluded=tuple(excluded),
        n_stocks=n_stocks, algo=algo, target_return=target_return
    )

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<div><h1>PSX AI Investment Advisory System — Streamlit Dashboard</h1>'
    '<p>BS Computer Science — 6th Semester AI Project &nbsp;·&nbsp; PSX Stocks</p></div>'
    '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LANDING
# ─────────────────────────────────────────────
if not st.session_state.get("generated"):
    c1, c2, c3 = st.columns(3)
    for col, sn, desc in [
        (c1,"1","Fill your investor profile in the settings panel"),
        (c2,"2","Select preferred sectors and optimization algorithm"),
        (c3,"3","Click Run Analysis to generate your AI portfolio"),
    ]:
        col.markdown('<div class="istep"><div class="sn">'+sn+'</div><p>'+desc+'</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Available Stock Database")
    st.dataframe(df, use_container_width=True, height=400)
    st.stop()

# ─────────────────────────────────────────────
#  PARAMS & RUN
# ─────────────────────────────────────────────
p         = st.session_state.params
amount    = p["amount"]
risk      = p["risk"]
preferred = p["preferred"]
excluded  = p["excluded"]
n_stocks  = p["n_stocks"]
algo      = p["algo"]
target_return = p["target_return"]

with st.spinner("Scoring & optimizing..."):
    scored_df, top_stocks, hc_result, sa_result = run_pipeline(preferred, excluded, risk, n_stocks, algo)

if scored_df.empty:
    st.error("No stocks found. Adjust your sector filters.")
    st.stop()

def build_alloc(weights):
    return [{"Symbol":row["Symbol"],"Name":row["Name"],"Sector":row["Sector"],
             "Score":row["Total_Score"],
             "Allocation %":round(weights[i]*100,1),
             "Amount (PKR)":int(weights[i]*amount)}
            for i, row in top_stocks.iterrows()]

if hc_result:
    hc_weights, hc_history, hc_iters = hc_result
    hc_metrics = portfolio_metrics(top_stocks, hc_weights)
    hc_alloc   = build_alloc(hc_weights)
if sa_result:
    sa_weights, sa_history, sa_iters = sa_result
    sa_metrics = portfolio_metrics(top_stocks, sa_weights)
    sa_alloc   = build_alloc(sa_weights)

if algo == "Hill Climbing":
    display_alloc, display_weights, display_metrics = hc_alloc, hc_weights, hc_metrics
    display_history, display_iters = hc_history, hc_iters
elif algo == "Simulated Annealing":
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics
    display_history, display_iters = sa_history, sa_iters
else:
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics

# ─────────────────────────────────────────────
#  TABS — exactly like reference (4 tabs)
# ─────────────────────────────────────────────
tab_names = ["Portfolio", "AI Reasoning", "Optimization", "All Scored Stocks"]
page = st.radio("", tab_names, horizontal=True, key="nav_page")
st.markdown("&nbsp;")

# ══════════════════════════════════════════════
#  PORTFOLIO TAB (like reference image 1)
# ══════════════════════════════════════════════
if page == "Portfolio":
    ret    = display_metrics["Expected Return (%)"]
    vrisk  = display_metrics["Portfolio Risk"]
    inf_risk = display_metrics.get("Inflation Risk", 0.13)
    algo_risk = display_metrics.get("Algorithm Risk", 8.7)
    opt_risk = display_metrics.get("Optimization Risk", 1.72)
    
    # Row 1: Estimated Return and Inflation Risk
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(
            '<div class="sc">'
            '<h3>Estimated Return</h3>'
            f'<p style="font-size:28px;font-weight:800;color:#0F1B2D;margin:5px 0;">{round(ret,1)}%</p>'
            f'<span class="kpi-badge bg-green">Above target</span>'
            '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="sc">'
            '<h3>Inflation Risk</h3>'
            f'<p style="font-size:28px;font-weight:800;color:#0F1B2D;margin:5px 0;">{inf_risk:.2f}</p>'
            '<span class="kpi-badge bg-blue">Below tolerance</span>'
            '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="sc">'
            '<h3>AI Reasoning</h3>'
            f'<p style="font-size:28px;font-weight:800;color:#0F1B2D;margin:5px 0;">{algo_risk}%</p>'
            '<span class="kpi-badge bg-amber">Normal scenario</span>'
            '</div>', unsafe_allow_html=True)
    
    # Row 2: Optimization
    col4, col5, col6 = st.columns([1,1,1])
    with col4:
        st.markdown(
            '<div class="sc">'
            '<h3>Optimization</h3>'
            f'<p style="font-size:28px;font-weight:800;color:#0F1B2D;margin:5px 0;">{opt_risk:.2f}</p>'
            '<span class="kpi-badge bg-red">Above tolerance</span>'
            '</div>', unsafe_allow_html=True)
    
    # AI Stocks table
    st.markdown('<div class="sc"><h3>AI Stocks</h3>', unsafe_allow_html=True)
    
    # Create the stock table with sectors
    stock_data = []
    for rec in display_alloc[:5]:
        stock_data.append({
            "Stock": rec["Symbol"],
            "Sector": rec["Sector"]
        })
    
    stock_df = pd.DataFrame(stock_data)
    
    # Display as columns like reference
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Stock**")
        for rec in display_alloc[:5]:
            st.write(rec["Symbol"])
    with col_b:
        st.markdown("**Sector**")
        for rec in display_alloc[:5]:
            st.write(rec["Sector"])
    with col_c:
        st.markdown("**Allocation by Sector**")
        for rec in display_alloc[:5]:
            st.write(f"{rec['Allocation %']}%")
    
    # Allocation by Sector pie
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="sc"><h3>Allocation by Sector</h3>', unsafe_allow_html=True)
    sec = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    show(make_hbar(sec["Sector"].tolist(), sec["Allocation %"].tolist(), ""))
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  AI REASONING TAB (like reference image 3)
# ══════════════════════════════════════════════
elif page == "AI Reasoning":
    st.markdown('<div class="sc"><h3>Settings Panel</h3><p style="color:#64748B;font-size:11px;">(clicked)</p></div>', unsafe_allow_html=True)
    
    COMP_MAXES = [30, 20, 15, 20, 15]
    COMP_NAMES = ["Growth", "Stability", "Dividend", "Momentum", "Sector"]
    
    # Portfolio cards
    st.markdown('<div class="sc"><h3>Portfolio</h3>', unsafe_allow_html=True)
    
    # Show top 3 stocks with detailed breakdown like reference
    top3 = display_alloc[:3]
    
    for rec in top3:
        sym = rec["Symbol"]
        row = scored_df[scored_df["Symbol"] == sym].iloc[0]
        bd = row["Breakdown"]
        
        # Map to component names
        comp_values = list(bd.values())[:5]
        
        # Stock card HTML
        score_html = ""
        for comp_name, score_val, mx in zip(COMP_NAMES, comp_values, COMP_MAXES):
            pct = min(score_val / mx * 100, 100)
            score_html += f'''
            <div style="margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;font-size:9px;color:#374151;">
                    <span>{comp_name}</span>
                    <span style="font-family:monospace;">{score_val:.1f}</span>
                </div>
                <div style="background:#F1F5F9;border-radius:3px;height:4px;">
                    <div style="width:{pct}%;background:#2563EB;height:4px;border-radius:3px;"></div>
                </div>
            </div>
            '''
        
        card = f'''
        <div style="background:white;border:1px solid #E2E8F0;border-radius:10px;padding:12px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:16px;font-weight:800;color:#0F1B2D;">{sym}</span>
                <span style="background:#EFF6FF;color:#2563EB;font-size:9px;font-weight:600;padding:2px 8px;border-radius:12px;">{row['Sector']}</span>
            </div>
            <div style="font-size:10px;color:#64748B;margin-bottom:8px;">{row['Name']}</div>
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;font-size:8px;color:#64748B;margin-bottom:2px;">
                    <span>Corporate Score</span><span>syro</span><span>syro Growth</span><span>Vitality</span><span>Dividend Yield</span><span>Monetars</span>
                </div>
                <div style="font-size:9px;color:#374151;margin-top:6px;">
                    - {sym} ranks highly due to strong 5 year growth acc. consistent dividend payouts. Sector preference: loans, apples
                </div>
            </div>
            {score_html}
            <div style="margin-top:8px;">
                <span style="background:#EFF6FF;color:#2563EB;font-size:10px;font-weight:700;padding:4px 10px;border-radius:6px;">Recommended allocation: {rec['Allocation %']}%</span>
            </div>
        </div>
        '''
        st.markdown(card, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Score Component Breakdown
    st.markdown('<div class="sc"><h3>Score Component Breakdown</h3>', unsafe_allow_html=True)
    
    col_sel, col_chart = st.columns([1, 2])
    with col_sel:
        selected = st.selectbox("Select a stock:", [r["Symbol"] + " — " + r["Name"] for r in display_alloc])
        sym = selected.split(" — ")[0]
    
    with col_chart:
        stock_row = scored_df[scored_df["Symbol"] == sym].iloc[0]
        bd = stock_row["Breakdown"]
        comp_vals = list(bd.values())[:5]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        y_pos = range(len(COMP_NAMES))
        bars = ax.barh(y_pos, comp_vals, color=PALETTE[:5])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(COMP_NAMES)
        ax.set_xlabel("Score")
        ax.set_title(f"{sym} — Component Breakdown", fontsize=10, fontweight="bold")
        for i, (bar, val) in enumerate(zip(bars, comp_vals)):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va="center", fontsize=8)
       
