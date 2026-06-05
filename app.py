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

/* ── Main bg ── */
.stApp { background:#F5F7FA !important; }

/* ══════════════════════════════
   SIDEBAR — Dark Blue Gradient (Matches Header)
══════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F, #1B4F82) !important;
    border-right: 1px solid #1E3A5F !important;
    min-width: 260px !important;
}
/* Force all text inside sidebar to white/light grey for contrast */
[data-testid="stSidebar"] * { color:#E2E8F0 !important; }

/* sidebar section headings */
[data-testid="stSidebar"] h2 {
    color:#FFFFFF !important; font-size:14px !important;
    font-weight:700 !important; margin-bottom:2px !important;
}

/* sidebar labels */
[data-testid="stSidebar"] label {
    color:#93C5FD !important; font-size:11px !important; font-weight:500 !important;
}

/* sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background:#152E4D !important; border:1px solid #2B4C7E !important;
    color:#FFFFFF !important; font-size:12px !important; border-radius:6px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background:#152E4D !important; border:1px solid #2B4C7E !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color:#FFFFFF !important; font-size:12px !important;
}

/* multiselect tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background:#1E40AF !important; border:1px solid #3B82F6 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color:#FFFFFF !important; }

/* sidebar radio — VERTICAL stack, no pill */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    background:transparent !important; border-radius:0 !important;
    padding:0 !important; gap:2px !important; flex-direction:column !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background:#1C3554 !important; border:1px solid #2B4C7E !important;
    border-radius:6px !important; padding:7px 12px !important;
    font-size:12px !important; font-weight:500 !important;
    color:#E2E8F0 !important; cursor:pointer; transition:all .15s;
    margin-bottom:3px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:#2563EB !important; border-color:#60A5FA !important;
    color:#FFFFFF !important; font-weight:600 !important;
}

/* run button */
[data-testid="stSidebar"] .stButton > button {
    background:linear-gradient(135deg,#0EA5E9,#38BDF8) !important;
    color:white !important; border:none !important; border-radius:7px !important;
    font-weight:600 !important; font-size:12px !important; padding:9px 0 !important;
    width:100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:linear-gradient(135deg,#0284C7,#0EA5E9) !important;
    box-shadow:0 4px 12px rgba(14,165,233,.35) !important;
}

/* slider thumb */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background:#38BDF8 !important;
}

/* divider */
[data-testid="stSidebar"] hr { border-color:#2B4C7E !important; }

/* Override sidebar nav buttons to act like flat tabs */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background:#1C3554 !important;
    border:1px solid #2B4C7E !important;
    color:#E2E8F0 !important;
    font-size:12px !important;
    font-weight:500 !important;
    text-align:left !important;
    padding:8px 12px !important;
    border-radius:7px !important;
    box-shadow:none !important;
    margin-bottom:3px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background:#2563EB !important;
    border-color:#60A5FA !important;
    color:#FFFFFF !important;
}
/* Run Analysis Button specific styling overrides */
[data-testid="stSidebar"] [data-testid="stButton"]:last-of-type > button {
    background:linear-gradient(135deg,#2563EB,#0EA5E9) !important;
    color:white !important;
    border:none !important;
    font-weight:600 !important;
}

/* ── Page header ── */
.page-header {
    background:linear-gradient(90deg,#1E3A5F,#1B4F82);
    border-radius:8px; padding:12px 20px; margin-bottom:14px;
    display:flex; align-items:center; gap:10px;
}
.page-header h1 { color:#F8FAFC; font-size:14px; font-weight:700; margin:0; }
.page-header p  { color:#93C5FD; font-size:10px; margin:2px 0 0; }

/* ── KPI row ── */
.kpi-row { display:flex; gap:10px; margin-bottom:14px; }
.kpi-card {
    flex:1; background:white; border-radius:8px; padding:12px 14px;
    border:1px solid #E5E7EB; box-shadow:0 1px 2px rgba(0,0,0,.04);
    border-top:3px solid #2563EB;
}
.kpi-card.g { border-top-color:#059669; }
.kpi-card.a { border-top-color:#D97706; }
.kpi-card.p { border-top-color:#7C3AED; }
.kpi-label  { color:#6B7280; font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-bottom:3px; }
.kpi-value  { color:#111827; font-size:20px; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1.1; }
.kpi-badge  { display:inline-block; margin-top:4px; font-size:9px; font-weight:600; padding:1px 7px; border-radius:20px; }
.bg-green { background:#D1FAE5; color:#065F46; }
.bg-blue  { background:#DBEAFE; color:#1E40AF; }
.bg-amber { background:#FEF3C7; color:#92400E; }

/* ── Section card ── */
.sc { background:white; border-radius:8px; border:1px solid #E5E7EB; padding:14px 16px; margin-bottom:12px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.sc h3 { color:#111827; font-size:12px; font-weight:600; margin:0 0 10px 0; }

/* Reason card */
.rc { background:white; border-radius:8px; border:1px solid #E5E7EB; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.rc .rc-name   { font-size:10px; color:#9CA3AF; margin-bottom:1px; }
.rc .rc-sym    { font-size:18px; font-weight:800; color:#111827; line-height:1.1; }
.rc .rc-sector { display:inline-block; background:#EFF6FF; color:#2563EB; font-size:9px; font-weight:600; padding:2px 7px; border-radius:10px; margin:5px 0 8px; }
.cs-row { margin-bottom:6px; }
.cs-top { display:flex; justify-content:space-between; font-size:10px; color:#374151; margin-bottom:2px; }
.cs-top span:last-child { font-family:'JetBrains Mono',monospace; font-weight:600; color:#111827; }
.cs-bg   { background:#F3F4F6; border-radius:3px; height:5px; }
.cs-fill{ height:5px; border-radius:3px; }
.stat-r { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #F3F4F6; font-size:10px; }
.stat-r .sl { color:#6B7280; }
.stat-r .sv { font-weight:600; color:#111827; font-family:'JetBrains Mono',monospace; }
.sv-low  { color:#059669 !important; font-weight:600 !important; }
.sv-med  { color:#D97706 !important; font-weight:600 !important; }
.sv-high { color:#DC2626 !important; font-weight:600 !important; }
.rc-note { font-size:9px; color:#9CA3AF; margin-top:8px; line-height:1.4; }
.alloc-tag { display:inline-block; background:#EFF6FF; color:#2563EB; font-size:9px; font-weight:700; padding:2px 8px; border-radius:4px; margin-top:7px; }

/* Algo metric box */
.am { text-align:center; background:white; border-radius:8px; border:1px solid #E5E7EB; padding:14px; }
.aml { font-size:9px; color:#6B7280; text-transform:uppercase; letter-spacing:.06em; margin-bottom:3px; }
.amv { font-size:24px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.ams { font-size:9px; color:#9CA3AF; margin-top:2px; }

.greeting { font-size:13px; color:#374151; font-weight:500; margin-bottom:10px; }
.greeting span { color:#2563EB; font-weight:700; }
.istep { background:white; border-radius:8px; border:1px solid #E5E7EB; padding:16px; text-align:center; }
.istep .sn { background:#EFF6FF; color:#2563EB; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; margin:0 auto 7px; }
.istep p { color:#6B7280; font-size:11px; margin:0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB VISUALIZATIONS (Optimized Sizes)
# ─────────────────────────────────────────────
PALETTE = ["#1a3fcc","#059669","#D97706","#DB2777","#7C3AED",
           "#0891B2","#DC2626","#16A34A","#CA8A04","#9333EA"]
BG, GRID, TXT = "#FFFFFF", "#E5E7EB", "#111827"

def _base_fig(w=5, h=2.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors="#6B7280", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

def make_pie(labels, values):
    fig, ax = plt.subplots(figsize=(2.8, 2.4)) # Scaled down nicely
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
    fig, ax = _base_fig(w=4.5, h=max(1.8, n*.35+.5))
    bars = ax.barh(cats, vals, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+max(vals)*.01, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", fontsize=7, color=TXT, fontweight="600")
    ax.set_xlabel("Allocation %", fontsize=7, color="#6B7280")
    ax.set_title(title, fontsize=8, color=TXT, pad=4, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_score_hbar(comps, scores, maxes):
    n = len(comps)
    fig, ax = _base_fig(w=4.5, h=max(1.8, n*.35+.5))
    bars = ax.barh(comps, scores, color=PALETTE[:n], edgecolor="white", linewidth=.6, zorder=3, height=.45)
    for i, mx in enumerate(maxes):
        ax.plot(mx, i, "|", color="#D1D5DB", markersize=10, markeredgewidth=1.5, zorder=4)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width()+.3, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=7, color=TXT, fontweight="600")
    ax.set_xlabel("Score", fontsize=7, color="#6B7280")
    ax.set_title("Score Component Breakdown", fontsize=8, color=TXT, pad=4, fontweight="600")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    plt.tight_layout(); return fig

def make_dual_line(hc_hist, sa_hist):
    fig, ax = _base_fig(w=5.0, h=2.6)
    x1, x2 = range(len(hc_hist)), range(len(sa_hist))
    ax.plot(hc_hist, color="#1a3fcc", linewidth=1.8, label="Hill Climbing", zorder=3)
    ax.plot(sa_hist, color="#059669", linewidth=1.8, linestyle="--", label="Simulated Annealing", zorder=3)
    ax.set_xlabel("Iterations", fontsize=7, color="#6B7280")
    ax.set_ylabel("Objective Score", fontsize=7, color="#6B7280")
    ax.set_title("Convergence Curves", fontsize=8, color=TXT, pad=5, fontweight="700")
    ax.legend(fontsize=7, framealpha=.8)
    plt.tight_layout(); return fig

def make_scatter(top_stocks, w1, w2):
    fig, ax = _base_fig(w=5.0, h=2.6)
    for i, row in top_stocks.iterrows():
        r  = row["Growth_5yr"] * 100
        v  = row["Volatility"]
        sz = 60 + row["Total_Score"]
        ax.scatter(v, r, s=sz, color=PALETTE[i % len(PALETTE)], alpha=.85, edgecolors="white", linewidth=1, zorder=3)
        ax.annotate(row["Symbol"], (v, r), textcoords="offset points", xytext=(5,3), fontsize=6, color=TXT)
    ax.set_xlabel("Portfolio Risk (Volatility)", fontsize=7, color="#6B7280")
    ax.set_ylabel("Expected Return (%)", fontsize=7, color="#6B7280")
    ax.set_title("Risk vs Return Scatter", fontsize=8, color=TXT, pad=5, fontweight="700")
    plt.tight_layout(); return fig

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────
#  DATA LOADING
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
    if algo in ("Hill Climbing","Both"):   hc = hill_climbing(top, risk, n_stocks)
    if algo in ("Simulated Annealing","Both"):  sa = simulated_annealing(top, risk, n_stocks)
    return scored, top, hc, sa

# ─────────────────────────────────────────────
#  SIDEBAR RENDERING
# ─────────────────────────────────────────────
NAV_ITEMS = [
    ("📊", "Portfolio"),
    ("🤖", "AI Reasoning"),
    ("⚡", "Optimization"),
    ("📋", "All Stocks"),
]

with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 10px;">'
        '<span style="font-size:20px;">📈</span>'
        '<span style="font-size:14px;font-weight:700;color:#FFFFFF;margin-left:6px;">InvestAI</span>'
        '<div style="font-size:10px;color:#93C5FD;margin-top:1px;">PSX Advisory System</div>'
        '</div>',
        unsafe_allow_html=True)

    user_name = st.text_input("Your Name", placeholder="Enter your name", key="user_name_input")
    st.markdown('<hr style="margin:4px 0 10px;"/>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;color:#93C5FD;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;">Navigation</div>', unsafe_allow_html=True)

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Portfolio"

    for icon, label in NAV_ITEMS:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.nav_page = label
            st.rerun()
            
    page = st.session_state.nav_page
    st.markdown('<hr style="margin:10px 0;"/>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;color:#93C5FD;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Investor Profile</div>', unsafe_allow_html=True)

    amount        = st.number_input("Investment Amount (PKR)", min_value=100_000, max_value=100_000_000, value=5_000_000, step=100_000)
    duration      = st.slider("Duration (Years)", 1, 15, 5)
    target_return = st.slider("Target Annual Return (%)", 5, 50, 20)

    st.markdown('<div style="font-size:11px;color:#93C5FD;font-weight:500;margin-bottom:4px;">Risk Appetite</div>', unsafe_allow_html=True)
    risk = st.radio("", ["Low", "Medium", "High"], index=1, key="risk_radio")

    st.markdown('<hr style="margin:10px 0;"/>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;color:#93C5FD;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Sector Preferences</div>', unsafe_allow_html=True)
    preferred = st.multiselect("Preferred Sectors", options=ALL_SECTORS, default=["Banking","Energy"])
    excluded  = st.multiselect("Excluded Sectors", options=[s for s in ALL_SECTORS if s not in preferred], default=[])

    st.markdown('<hr style="margin:10px 0;"/>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;color:#93C5FD;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Portfolio Settings</div>', unsafe_allow_html=True)
    n_stocks = st.slider("Portfolio Size", 3, 12, 5)
    algo     = st.selectbox("Algorithm", ["Hill Climbing","Simulated Annealing","Both"])

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  STATE PROCESSING & RENDERING
# ─────────────────────────────────────────────
if run_btn:
    st.session_state.generated = True
    st.session_state.params = dict(
        amount=amount, risk=risk, preferred=tuple(preferred), excluded=tuple(excluded),
        n_stocks=n_stocks, algo=algo, user_name=user_name,
    )

name_display = user_name.strip() if user_name and user_name.strip() else None
greeting_html = f'<div class="greeting">Welcome back, <span>{name_display}</span> 👋</div>' if name_display else ""

st.markdown(
    '<div class="page-header">'
    '<div>'
    '<h1>PSX AI Investment Advisory System</h1>'
    '<p>BS Computer Science — 6th Semester AI Project &nbsp;·&nbsp; PSX Stocks</p>'
    '</div>'
    '</div>', unsafe_allow_html=True)

if greeting_html:
    st.markdown(greeting_html, unsafe_allow_html=True)

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

p = st.session_state.params
amount = p["amount"]; risk = p["risk"]
preferred = p["preferred"]; excluded = p["excluded"]
n_stocks = p["n_stocks"]; algo = p["algo"]

with st.spinner("Scoring & optimizing..."):
    scored_df, top_stocks, hc_result, sa_result = run_pipeline(preferred, excluded, risk, n_stocks, algo)

if scored_df.empty:
    st.error("No stocks found. Adjust your sector filters.")
    st.stop()

def build_alloc(weights):
    return [{"Symbol":row["Symbol"],"Name":row["Name"],"Sector":row["Sector"],
             "Score":row["Total_Score"], "Allocation %":round(weights[i]*100,1),
             "Amount (PKR)":int(weights[i]*amount)} for i, row in top_stocks.iterrows()]

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
else:
    display_alloc, display_weights, display_metrics = sa_alloc, sa_weights, sa_metrics

# ══════════════════════════════════════════════
#  PAGES CONTENT ROUTER
# ══════════════════════════════════════════════
if page == "Portfolio":
    ret    = display_metrics["Expected Return (%)"]
    vrisk  = display_metrics["Portfolio Risk"]
    div    = display_metrics["Avg Dividend Yield"]
    sharpe = display_metrics["Sharpe-like Ratio"]
    rb   = ("Above Target","bg-green") if ret >= target_return else ("Below Target","bg-amber")
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

    col_pie, col_tbl = st.columns([4, 6], gap="medium")
    with col_pie:
        st.markdown('<div class="sc"><h3>Portfolio Allocation</h3>', unsafe_allow_html=True)
        pie_df = pd.DataFrame(display_alloc)
        show(make_pie(pie_df["Symbol"].tolist(), pie_df["Allocation %"].tolist()))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tbl:
        st.markdown('<div class="sc"><h3>Recommended Stocks</h3>', unsafe_allow_html=True)
        adf = pd.DataFrame(display_alloc).copy()
        adf["Amount (PKR)"] = adf["Amount (PKR)"].apply(lambda x: f"Rs {x:,}")
        adf["Score"]        = adf["Score"].apply(lambda x: f"{x:.0f}/100")
        adf["Alloc %"]      = adf["Allocation %"].apply(lambda x: f"{x}%")
        st.dataframe(adf[["Symbol","Sector","Score","Alloc %","Amount (PKR)"]], use_container_width=True, hide_index=True, height=230)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc"><h3>Allocation by Sector</h3>', unsafe_allow_html=True)
    sec = pd.DataFrame(display_alloc).groupby("Sector")["Allocation %"].sum().reset_index()
    show(make_hbar(sec["Sector"].tolist(), sec["Allocation %"].tolist(), "Sector Distribution"))
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "AI Reasoning":
    st.markdown('<p style="font-size:13px;font-weight:600;color:#111827;margin-bottom:12px;">AI Recommendation Reasoning</p>', unsafe_allow_html=True)
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
        for (comp, score), mx, clr in zip(bd.items(), COMP_MAXES, PALETTE[:5]):
            pct = round(min(score/mx*100, 100), 1)
            bars_html += f'<div class="cs-row"><div class="cs-top"><span>{comp}</span><span>{round(score,1)}</span></div><div class="cs-bg"><div class="cs-fill" style="width:{pct}%;background:{clr};"></div></div></div>'

        card = f'<div class="rc"><div class="rc-name">{row["Name"]}</div><div class="rc-sym">{sym}</div><div><span class="rc-sector">{row["Sector"]}</span></div><div class="rc-score-lbl">Composite Score</div>{bars_html}<div class="stat-r"><span class="sl">5yr Growth</span><span class="sv">{round(row["Growth_5yr"]*100)}%</span></div><div class="stat-r"><span class="sl">Volatility</span><span class="{risk_cls}">{risk_lbl}</span></div><div class="stat-r"><span class="sl">Dividend Yield</span><span class="sv">{round(row["Dividend_Yield"],1)}%</span></div><div class="rc-note">{sym} ranks highly due to solid performance markers.</div><div><span class="alloc-tag">Recommended allocation {rec["Allocation %"]}%</span></div></div>'
        col.markdown(card, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:12px;font-weight:600;color:#111827;">Score Component Breakdown</p>', unsafe_allow_html=True)
    selected = st.selectbox("Select a stock:", [r["Symbol"]+" — "+r["Name"] for r in display_alloc], key="stock_explain")
    sym = selected.split(" — ")[0]
    stock_row = scored_df[scored_df["Symbol"]==sym].iloc[0]
    bd = stock_row["Breakdown"]
    show(make_score_hbar(list(bd.keys()), list(bd.values()), COMP_MAXES))

elif page == "Optimization":
    st.markdown('<p style="font-size:13px;font-weight:600;color:#111827;margin-bottom:12px;">Optimization Algorithm Results</p>', unsafe_allow_html=True)
    if algo == "Both" and hc_result and sa_result:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<p class="chart-title">Convergence Curves</p>', unsafe_allow_html=True)
            show(make_dual_line(hc_history, sa_history))
        with c2:
            st.markdown('<p class="chart-title">Risk vs Return Scatter</p>', unsafe_allow_html=True)
            show(make_scatter(top_stocks, hc_weights, sa_weights))
    else:
        st.info("Select algorithm 'Both' in the profile dashboard settings sidebar to see algorithmic convergence comparisons.")

elif page == "All Stocks":
    st.markdown('<p style="font-size:13px;font-weight:600;color:#111827;margin-bottom:12px;">Complete Filtered Stock Evaluation Matrix</p>', unsafe_allow_html=True)
    st.dataframe(scored_df, use_container_width=True, height=450)
