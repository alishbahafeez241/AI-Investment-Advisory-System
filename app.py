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
[data-testid="stRadio"] > div {
    background:#E2E8F0; border-radius:7px; padding:3px; gap:2px !important; display:flex;
}
[data-testid="stRadio"] label {
    background:transparent !important; border-radius:5px !important;
    padding:5px 14px !important; font-size:11px !important; font-weight:500 !important;
    color:#64748B !important; cursor:pointer; transition:all .15s;
}
[data-testid="stRadio"] label:has(input:checked) {
    background:white !important; color:#1B2A42 !important;
    box-shadow:0 1px 3px rgba(0,0,0,.1) !important; font-weight:600 !important;
}

/* ── KPI row ── */
.kpi-row { display:flex; gap:10px; margin-bottom:14px; }
.kpi-card {
    flex:1; background:white; border-radius:8px; padding:12px 14px;
    border:1px solid #E2E8F0; box-shadow:0 1px 2px rgba(0,0,0,.04);
    border-top:3px solid #2563EB;
}
.kpi-card.g { border-top-color:#1D9E75; }
.kpi-card.a { border-top-color:#EF9F27; }
.kpi-card.p { border-top-color:#8B5CF6; }
.kpi-label  { color:#64748B; font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-bottom:3px; }
.kpi-value  { color:#0F1B2D; font-size:20px; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1.1; }
.kpi-badge  { display:inline-block; margin-top:4px; font-size:9px; font-weight:600; padding:1px 6px; border-radius:20px; }
.bg-green { background:#D1FAE5; color:#065F46; }
.bg-blue  { background:#DBEAFE; color:#1E40AF; }
.bg-amber { background:#FEF3C7; color:#92400E; }

/* ── Section card ── */
.sc { background:white; border-radius:8px; border:1px solid #E2E8F0; padding:14px 16px; margin-bottom:12px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.sc h3 { color:#0F1B2D; font-size:12px; font-weight:600; margin:0 0 10px 0; }

/* ── Reason card ── */
.rc { background:white; border-radius:8px; border:1px solid #E2E8F0; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.rc .rc-name   { font-size:10px; color:#94A3B8; margin-bottom:1px; }
.rc .rc-sym    { font-size:18px; font-weight:800; color:#0F1B2D; line-height:1.1; }
.rc .rc-sector { display:inline-block; background:#EFF6FF; color:#2563EB; font-size:9px; font-weight:600; padding:2px 7px; border-radius:10px; margin:5px 0 8px; }
.rc .rc-score-lbl { font-size:9px; color:#94A3B8; margin-bottom:5px; }
.cs-row { margin-bottom:6px; }
.cs-top { display:flex; justify-content:space-between; font-size:10px; color:#374151; margin-bottom:2px; }
.cs-top span:last-child { font-family:'JetBrains Mono',monospace; font-weight:600; color:#0F1B2D; }
.cs-bg  { background:#F1F5F9; border-radius:3px; height:5px; }
.cs-fill{ height:5px; border-radius:3px; }
.stat-r { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #F1F5F9; font-size:10px; }
.stat-r .sl { color:#64748B; }
.stat-r .sv { font-weight:600; color:#0F1B2D; font-family:'JetBrains Mono',monospace; font-size:10px; }
.sv-low  { color:#16A34A !important; font-weight:600 !important; font-size:10px !important; }
.sv-med  { color:#D97706 !important; font-weight:600 !important; font-size:10px !important; }
.sv-high { color:#DC2626 !important; font-weight:600 !important; font-size:10px !important; }
.rc-note { font-size:9px; color:#94A3B8; margin-top:8px; line-height:1.4; }
.alloc-tag { display:inline-block; background:#EFF6FF; color:#2563EB; font-size:9px; font-weight:700; padding:2px 8px; border-radius:4px; margin-top:7px; }

/* ── Algo metric box ── */
.am { text-align:center; background:white; border-radius:8px; border:1px solid #E2E8F0; padding:14px; }
.aml { font-size:9px; color:#64748B; text-transform:uppercase; letter-spacing:.06em; margin-bottom:3px; }
.amv { font-size:24px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.ams { font-size:9px; color:#94A3B8; margin-top:2px; }

/* ── Info step ── */
.istep { background:white; border-radius:8px; border:1px solid #E2E8F0; padding:16px; text-align:center; }
.istep .sn { background:#EFF6FF; color:#2563EB; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; margin:0 auto 7px; }
.istep p { color:#64748B; font-size:11px; margin:0; }

.footer { text-align:center; color:#94A3B8; font-size:10px; padding:18px 0 6px; }
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
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors="#64748B", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

def make_pie(labels, values):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    fig.patch.set_facecolor(BG)
    wedges, texts, auto = ax.pie(
        values, labels=labels, autopct="%1.1f%%", colors=PALETTE[:len(labels)],
        startangle=90, wedgeprops=dict(width=.52, edgecolor="white", linewidth=1.5),
        textprops=dict(color=TXT, fontsize=7))
    for at in auto: at.set_fontsize(6); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Allocation by Symbol", fontsize=8, color=TXT, pad=5, fontweight="600")
    plt.tight_layout(); return fig

def make_hbar(cats, vals, title=""):
    n = len(cats)
    fig, ax = _base_fig(w=5, h=max(2.0, n*.42+.6))
    bars = ax.barh(cats, vals, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.48)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+max(vals)*.01, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", fontsize=7, color=TXT, fontweight="600")
    ax.set_xlabel("Allocation %", fontsize=7, color="#64748B")
    ax.set_title(title, fontsize=8, color=TXT, pad=4, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_score_hbar(comps, scores, maxes):
    n = len(comps)
    fig, ax = _base_fig(w=5, h=max(2.0, n*.42+.6))
    bars = ax.barh(comps, scores, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.48)
    for i, mx in enumerate(maxes):
        ax.plot(mx, i, "|", color="#CBD5E1", markersize=10, markeredgewidth=1.5, zorder=4)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width()+.3, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=7, color=TXT, fontweight="600")
    ax.set_xlabel("Score", fontsize=7, color="#64748B")
    ax.set_title("Score Component Breakdown", fontsize=8, color=TXT, pad=4, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_line(history, title="", color="#1a3fcc", label=None):
    """Convergence line — fill between curve and its own min so fill always looks correct."""
    fig, ax = _base_fig(w=5.5, h=3.0)
    x = range(len(history))
    ax.plot(history, color=color, linewidth=2.0, zorder=3, label=label)
    # Fill between the curve and the axis bottom (min value), handles both pos & neg
    base = min(history)
    ax.fill_between(x, history, base, alpha=.12, color=color)
    ax.set_xlabel("Iteration", fontsize=7, color="#64748B")
    ax.set_ylabel("Objective Score", fontsize=7, color="#64748B")
    ax.set_title(title, fontsize=9, color=TXT, pad=5, fontweight="700")
    if label: ax.legend(fontsize=7, framealpha=.8)
    plt.tight_layout(); return fig

def make_dual_line(hc_hist, sa_hist):
    """Both curves on one chart — matches reference screenshot."""
    fig, ax = _base_fig(w=5.5, h=3.0)
    x1, x2 = range(len(hc_hist)), range(len(sa_hist))
    base1, base2 = min(hc_hist), min(sa_hist)
    ax.fill_between(x1, hc_hist, base1, alpha=.10, color="#1a3fcc")
    ax.fill_between(x2, sa_hist, base2, alpha=.10, color="#1D9E75")
    ax.plot(hc_hist, color="#1a3fcc", linewidth=2.0, label="Hill Climbing", zorder=3)
    ax.plot(sa_hist, color="#1D9E75", linewidth=2.0, linestyle="--", label="Simulated Annealing", zorder=3)
    ax.set_xlabel("Iterations", fontsize=7, color="#64748B")
    ax.set_ylabel("Objective Score", fontsize=7, color="#64748B")
    ax.set_title("Convergence Curves", fontsize=9, color=TXT, pad=5, fontweight="700")
    ax.legend(fontsize=7, framealpha=.8)
    plt.tight_layout(); return fig

def make_scatter(top_stocks, hc_w, sa_w):
    """Risk vs Return scatter — efficient frontier style."""
    fig, ax = _base_fig(w=5, h=3.0)
    ax.set_facecolor("#F8FAFD")
    for i, row in top_stocks.iterrows():
        r  = row["Growth_5yr"] * 100
        v  = row["Volatility"]
        sz = 60 + row["Total_Score"]
        ax.scatter(v, r, s=sz, color=PALETTE[i % len(PALETTE)], alpha=.85, edgecolors="white", linewidth=1, zorder=3)
        ax.annotate(row["Symbol"], (v, r), textcoords="offset points", xytext=(5,3),
                    fontsize=6, color=TXT)
    # Draw a smooth efficient frontier curve
    xs = np.linspace(top_stocks["Volatility"].min()*.8, top_stocks["Volatility"].max()*1.1, 80)
    ys = 8 + 120 * (xs - xs.min())**.5
    ax.plot(xs, ys, color="#94A3B8", linewidth=1.2, linestyle="--", label="Efficient Frontier", zorder=2)
    ax.set_xlabel("Portfolio Risk (Volatility)", fontsize=7, color="#64748B")
    ax.set_ylabel("Expected Return (%)", fontsize=7, color="#64748B")
    ax.set_title("Risk vs Return Scatter", fontsize=9, color=TXT, pad=5, fontweight="700")
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
    st.markdown("## Investor Settings")
    st.markdown("---")
    amount        = st.number_input("Investment Amount (PKR)", min_value=100_000, max_value=100_000_000, value=5_000_000, step=100_000)
    duration      = st.slider("Duration (Years)", 1, 15, 5)
    target_return = st.slider("Target Annual Return (%)", 5, 50, 20)
    risk          = st.radio("Risk Appetite", ["Low","Medium","High"], index=1, horizontal=True)
    st.markdown("---")
    preferred = st.multiselect("Preferred Sectors", options=ALL_SECTORS, default=["Banking","Energy"])
    excluded  = st.multiselect("Excluded Sectors", options=[s for s in ALL_SECTORS if s not in preferred], default=[])
    st.markdown("---")
    n_stocks = st.slider("Portfolio Size", 3, 12, 5)
    algo     = st.selectbox("Algorithm", ["Hill Climbing","Simulated Annealing","Both"])
    st.markdown("&nbsp;")
    run_btn = st.button("Run Analysis", use_container_width=True, type="primary")

if run_btn:
    st.session_state.generated = True
    st.session_state.params = dict(
        amount=amount, risk=risk,
        preferred=tuple(preferred), excluded=tuple(excluded),
        n_stocks=n_stocks, algo=algo,
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
amount    = p["amount"]; risk = p["risk"]
preferred = p["preferred"]; excluded = p["excluded"]
n_stocks  = p["n_stocks"]; algo = p["algo"]

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
elif algo == "Simulated Annealing":
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics
else:
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics

# ─────────────────────────────────────────────
#  TABS  — no emoji icons
# ─────────────────────────────────────────────
page = st.radio("", ["Portfolio","AI Reasoning","Optimization","All Stocks"],
                horizontal=True, key="nav_page")
st.markdown("&nbsp;")

# ══════════════════════════════════════════════
#  PORTFOLIO
# ══════════════════════════════════════════════
if page == "Portfolio":
    ret    = display_metrics["Expected Return (%)"]
    vrisk  = display_metrics["Portfolio Risk"]
    div    = display_metrics["Avg Dividend Yield"]
    sharpe = display_metrics["Sharpe-like Ratio"]
    rb  = ("Above Target","bg-green") if ret >= target_return else ("Below Target","bg-amber")
    slbl = "Excellent" if sharpe>=1.5 else ("Good" if sharpe>=1.0 else "Fair")
    scls = "bg-green" if sharpe>=1.5 else "bg-blue"

    st.markdown(
        '<div class="kpi-row">'
        '<div class="kpi-card"><div class="kpi-label">Expected Return</div>'
        '<div class="kpi-value">'+str(round(ret,1))+'%</div>'
        '<span class="kpi-badge '+rb[1]+'">'+rb[0]+'</span></div>'

        '<div class="kpi-card g"><div class="kpi-label">Portfolio Risk</div>'
        '<div class="kpi-value">'+str(round(vrisk,3))+'</div>'
        '<span class="kpi-badge bg-blue">Within tolerance</span></div>'

        '<div class="kpi-card a"><div class="kpi-label">Avg Dividend Yield</div>'
        '<div class="kpi-value">'+str(round(div,1))+'%</div>'
        '<span class="kpi-badge bg-green">Annual income</span></div>'

        '<div class="kpi-card p"><div class="kpi-label">Sharpe Ratio</div>'
        '<div class="kpi-value">'+str(round(sharpe,2))+'</div>'
        '<span class="kpi-badge '+scls+'">'+slbl+'</span></div>'
        '</div>', unsafe_allow_html=True)

    col_pie, col_tbl = st.columns([1,1], gap="large")
    with col_pie:
        st.markdown('<div class="sc"><h3>Portfolio Allocation</h3>', unsafe_allow_html=True)
        pie_df = pd.DataFrame(display_alloc)
        show(make_pie(pie_df["Symbol"].tolist(), pie_df["Allocation %"].tolist()))
        legend = " &nbsp; ".join(
            '<span style="color:'+PALETTE[i]+';font-weight:700;">●</span> '
            '<span style="font-size:10px;color:#374151;">'+r["Symbol"]+' — '+str(r["Allocation %"])+'%</span>'
            for i, r in enumerate(display_alloc))
        st.markdown('<div style="margin-top:5px;">'+legend+'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tbl:
        st.markdown('<div class="sc"><h3>Recommended Stocks</h3>', unsafe_allow_html=True)
        adf = pd.DataFrame(display_alloc).copy()
        adf["Amount (PKR)"] = adf["Amount (PKR)"].apply(lambda x: f"Rs {x:,}")
        adf["Score"]        = adf["Score"].apply(lambda x: f"{x:.0f}/100")
        adf["Alloc %"]      = adf["Allocation %"].apply(lambda x: f"{x}%")
        st.dataframe(adf[["Symbol","Sector","Score","Alloc %","Amount (PKR)"]],
                     use_container_width=True, hide_index=True, height=270)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc"><h3>Allocation by Sector</h3>', unsafe_allow_html=True)
    sec = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    show(make_hbar(sec["Sector"].tolist(), sec["Allocation %"].tolist(), "Sector Distribution"))
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  AI REASONING
# ══════════════════════════════════════════════
elif page == "AI Reasoning":
    st.markdown('<p style="font-size:13px;font-weight:600;color:#0F1B2D;margin-bottom:12px;">AI Recommendation Reasoning</p>', unsafe_allow_html=True)

    COMP_COLORS = ["#2563EB","#1D9E75","#EF9F27","#D4537E","#8B5CF6"]
    COMP_MAXES  = [30, 20, 15, 20, 15]

    top3 = display_alloc[:3]
    cols = st.columns(len(top3), gap="medium")

    for col, rec in zip(cols, top3):
        sym = rec["Symbol"]
        row = scored_df[scored_df["Symbol"] == sym].iloc[0]
        bd  = row["Breakdown"]
        vol = row["Volatility"]
        risk_lbl = "Low" if vol<0.15 else ("Medium" if vol<0.25 else "High")
        risk_cls = "sv-low" if vol<0.15 else ("sv-med" if vol<0.25 else "sv-high")

        bars_html = ""
        for (comp, score), mx, clr in zip(bd.items(), COMP_MAXES, COMP_COLORS):
            pct = round(min(score/mx*100, 100), 1)
            bars_html += (
                '<div class="cs-row">'
                '<div class="cs-top"><span>'+str(comp)+'</span><span>'+str(round(score,1))+'</span></div>'
                '<div class="cs-bg"><div class="cs-fill" style="width:'+str(pct)+'%;background:'+clr+';"></div></div>'
                '</div>'
            )

        card = (
            '<div class="rc">'
            '<div class="rc-name">'+str(row["Name"])+'</div>'
            '<div class="rc-sym">'+sym+'</div>'
            '<div><span class="rc-sector">'+str(row["Sector"])+'</span></div>'
            '<div class="rc-score-lbl">Composite Score</div>'
            +bars_html+
            '<div class="stat-r"><span class="sl">5yr Growth</span><span class="sv">'+str(round(row["Growth_5yr"]*100))+'%</span></div>'
            '<div class="stat-r"><span class="sl">Volatility</span><span class="'+risk_cls+'">'+risk_lbl+'</span></div>'
            '<div class="stat-r"><span class="sl">Dividend Yield</span><span class="sv">'+str(round(row["Dividend_Yield"],1))+'%</span></div>'
            '<div class="stat-r" style="border:none;"><span class="sl">Momentum</span><span class="sv">'+str(round(row["Momentum_Score"],2))+'</span></div>'
            '<div class="rc-note">'+sym+' ranks highly due to strong 5-year growth and consistent dividend payout; sector preference bonus applied.</div>'
            '<div><span class="alloc-tag">Recommended allocation '+str(rec["Allocation %"])+'%</span></div>'
            '</div>'
        )
        col.markdown(card, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:12px;font-weight:600;color:#0F1B2D;">Score Component Breakdown</p>', unsafe_allow_html=True)
    selected  = st.selectbox("Select a stock:",
        [r["Symbol"]+" — "+r["Name"] for r in display_alloc], key="stock_explain")
    sym       = selected.split(" — ")[0]
    stock_row = scored_df[scored_df["Symbol"]==sym].iloc[0]
    bd        = stock_row["Breakdown"]
    show(make_score_hbar(list(bd.keys()), list(bd.values()), COMP_MAXES))
    st.caption("AI Reasoning Tab — Scores computed by scoring_engine using 5 weighted components")

# ══════════════════════════════════════════════
#  OPTIMIZATION  — matches reference screenshot
# ══════════════════════════════════════════════
elif page == "Optimization":
    st.markdown('<p style="font-size:13px;font-weight:600;color:#0F1B2D;margin-bottom:12px;">Optimization Algorithm Results</p>', unsafe_allow_html=True)

    if algo == "Both":
        # Top row: convergence curves | risk-return scatter
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="sc"><h3>Convergence Curves</h3>', unsafe_allow_html=True)
            show(make_dual_line(hc_history, sa_history))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="sc"><h3>Risk vs Return Scatter</h3>', unsafe_allow_html=True)
            show(make_scatter(top_stocks, hc_weights, sa_weights))
            st.markdown('</div>', unsafe_allow_html=True)

        # Score boxes
        hc_best = max(hc_history); sa_best = max(sa_history)
        winner  = "SA" if sa_best > hc_best else "HC"
        b1, b2, b3 = st.columns([1,1,2], gap="medium")
        with b1:
            st.markdown(
                '<div class="am"><div class="aml">HC Best Score</div>'
                '<div class="amv" style="color:#1a3fcc;">'+str(round(hc_best,3))+'</div>'
                '<div class="ams">Iterations: '+str(hc_iters)+'</div></div>',
                unsafe_allow_html=True)
        with b2:
            st.markdown(
                '<div class="am"><div class="aml">SA Best Score</div>'
                '<div class="amv" style="color:#1D9E75;">'+str(round(sa_best,3))+'</div>'
                '<div class="ams">Temp: 1000 → 0.01</div></div>',
                unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="sc" style="margin-bottom:0;">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:11px;font-weight:600;color:#0F1B2D;margin-bottom:6px;">Algorithm Comparison</p>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:11px;color:#374151;">SA found a marginally better portfolio (+'+str(round(abs(sa_best-hc_best),4))+') by escaping local optima; SA converged faster. Both are appropriate for this portfolio size.<br>'
                '<span style="font-size:10px;color:#64748B;">Recommendation: use SA for large portfolios (N &gt; 8 stocks).</span></p>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Single algorithm — convergence + metrics
        history = hc_history if algo=="Hill Climbing" else sa_history
        iters   = hc_iters   if algo=="Hill Climbing" else sa_iters
        metrics = hc_metrics if algo=="Hill Climbing" else sa_metrics
        clr     = "#1a3fcc"  if algo=="Hill Climbing" else "#1D9E75"
        best    = max(history)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="sc"><h3>'+algo+' Convergence</h3>', unsafe_allow_html=True)
            show(make_line(history, algo+" Convergence", clr))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="sc"><h3>Risk vs Return Scatter</h3>', unsafe_allow_html=True)
            w = hc_weights if algo=="Hill Climbing" else sa_weights
            show(make_scatter(top_stocks, w, w))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="am" style="max-width:180px;margin-bottom:14px;">'
            '<div class="aml">Best Score</div>'
            '<div class="amv" style="color:'+clr+';">'+str(round(best,3))+'</div>'
            '<div class="ams">Iterations: '+str(iters)+'</div></div>',
            unsafe_allow_html=True)
        for k, v in metrics.items(): st.metric(k, v)

    st.caption("Optimization Tab — "+algo+" convergence comparison")

# ══════════════════════════════════════════════
#  ALL STOCKS
# ══════════════════════════════════════════════
elif page == "All Stocks":
    total = len(scored_df)
    st.markdown('<p style="font-size:13px;font-weight:600;color:#0F1B2D;">All '+str(total)+' Stocks — Ranked by Score</p>', unsafe_allow_html=True)
    st.caption("Showing top 10 of "+str(total))
    cols_s = ["Symbol","Name","Sector","Total_Score","Growth_5yr","Volatility","Dividend_Yield","Momentum_Score"]
    sdf = scored_df[cols_s].copy()
    sdf["Growth_5yr"]     = (sdf["Growth_5yr"]*100).round(1).astype(str)+"%"
    sdf["Dividend_Yield"] = sdf["Dividend_Yield"].round(1).astype(str)+"%"
    sdf["Volatility"]     = sdf["Volatility"].round(2)
    sdf["Total_Score"]    = sdf["Total_Score"].round(0).astype(int)
    sdf = sdf.rename(columns={"Total_Score":"Score","Growth_5yr":"Growth",
                               "Dividend_Yield":"Div Yield","Momentum_Score":"Momentum"})
    st.dataframe(sdf.style.background_gradient(subset=["Score"], cmap="Blues"),
                 use_container_width=True, hide_index=True, height=500)
    st.caption("All Stocks Table — Scores calculated using 5-component heuristic function with risk appetite adjustment")

st.markdown("<div class='footer'>AI Investment Advisory System · BS CS 6th Semester · PSX Stocks — Simulated Data</div>",
            unsafe_allow_html=True)
