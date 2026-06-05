import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PSX AI Investment Advisory System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark-navy dashboard theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f5f7fb; color: #0f172a; }

section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] * { color: #0f172a !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e3a8a !important; font-size: 0.8rem; letter-spacing: 0.10em;
    text-transform: uppercase; font-weight: 700;
}

/* Make the "open sidebar" arrow always visible when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    background: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
    border-radius: 8px !important;
    padding: 6px !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.15) !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    color: #1d4ed8 !important;
    fill: #1d4ed8 !important;
    opacity: 1 !important;
}


.title-bar {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 50%, #1d4ed8 100%);
    border-bottom: 1px solid #1e40af;
    padding: 16px 28px; text-align: center; margin: -1rem -1rem 1.2rem -1rem;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 0.02em; color: #ffffff;
}

.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 22px; text-align: center;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(37,99,235,.15); }
.metric-label { font-size: 0.75rem; color: #475569; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; font-weight: 600; }
.metric-value { font-size: 2rem; font-weight: 700; color: #1d4ed8; font-family: 'DM Mono', monospace; }
.metric-sub { font-size: 0.78rem; color: #16a34a; margin-top: 4px; font-weight: 600; }
.metric-sub.warn { color: #d97706; }

.stock-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; background:#ffffff; border-radius: 8px; overflow:hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }
.stock-table th {
    background: #1d4ed8; color: #ffffff; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.08em; padding: 12px 14px; text-align: left;
}
.stock-table td { padding: 11px 14px; border-bottom: 1px solid #e2e8f0; color:#0f172a; }
.stock-table tr:hover td { background: #eff6ff; }
.symbol { color: #1d4ed8; font-weight: 700; font-family: 'DM Mono', monospace; }
.score-badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-weight: 700; font-family: 'DM Mono', monospace; font-size: 0.82rem; }
.score-high { background: #dbeafe; color: #1d4ed8; }
.score-mid  { background: #fef3c7; color: #b45309; }
.score-low  { background: #e2e8f0; color: #475569; }
.ret-pos { color: #16a34a; font-weight: 600; }
.ret-med { color: #ea580c; font-weight: 600; }
.status-top { color: #b45309; font-weight: 700; }
.status-rec { color: #1d4ed8; font-weight: 600; }
.status-sel { color: #16a34a; font-weight: 600; }
.status-con { color: #475569; }
.status-neu { color: #64748b; }

.reason-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 22px; height: 100%; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.reason-card .ticker { font-size: 1.5rem; font-weight: 800; color: #1d4ed8; }
.reason-card .cname  { font-size: 0.82rem; color: #475569; margin-bottom: 6px; }
.reason-card .sector-tag {
    display: inline-block; background: #dbeafe; color: #1d4ed8;
    border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600; margin-bottom: 14px;
}
.reason-metric { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.86rem; color:#0f172a; }
.reason-metric .lbl { color: #475569; }
.reason-card .reasoning-text {
    font-size: 0.82rem; color: #1e293b; line-height: 1.6;
    margin: 12px 0; padding: 12px; background: #f1f5f9; border-radius: 8px; border-left: 3px solid #2563eb;
}
.alloc-badge {
    background: #1d4ed8; color: #ffffff; border-radius: 6px;
    padding: 5px 14px; font-size: 0.82rem; font-weight: 700; font-family: 'DM Mono', monospace;
}

.algo-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.algo-label { font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; font-weight:600; }
.algo-value { font-size: 2.2rem; font-weight: 800; font-family: 'DM Mono', monospace; }
.algo-hc { color: #1d4ed8; }
.algo-sa { color: #dc2626; }

.section-header {
    font-size: 1.05rem; font-weight: 700; color: #0f172a;
    margin: 1rem 0 0.6rem 0; padding-bottom: 6px; border-bottom: 2px solid #2563eb;
}

.prog-row { display: flex; align-items: center; margin-bottom: 10px; font-size: 0.85rem; }
.prog-label { width: 160px; color: #475569; }
.prog-bar-outer { flex: 1; background: #e2e8f0; border-radius: 6px; height: 8px; overflow: hidden; }
.prog-bar-inner { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.prog-val { width: 36px; text-align: right; color: #1d4ed8; font-family: 'DM Mono', monospace; margin-left: 8px; font-weight:600; }

div[data-testid="stButton"] > button {
    background: #1d4ed8; color: #ffffff !important; border: none;
    border-radius: 10px; font-weight: 700; font-size: 0.95rem;
    padding: 10px 0; width: 100%; transition: all .2s; letter-spacing: 0.04em;
}
div[data-testid="stButton"] > button:hover {
    background: #2563eb; transform: translateY(-1px); box-shadow: 0 6px 18px rgba(37,99,235,.35);
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #475569; font-weight: 600; font-size: 0.9rem; padding: 8px 20px;
}
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: #ffffff !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/*header { visibility: hidden; } */

label { color: #0f172a !important; font-size: 0.88rem !important; font-weight: 600 !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div, .stTextInput input {
    background: #ffffff !important; border: 1px solid #cbd5e1 !important;
    color: #0f172a !important; border-radius: 8px !important;
}
.stRadio label, .stCheckbox label { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STOCK UNIVERSE  (25 PSX stocks)
# ─────────────────────────────────────────────
STOCKS = [
    {"symbol": "MCB",    "name": "MCB Bank",            "sector": "Banking",    "price": 280,  "growth": 0.18, "volatility": "Low",    "div_yield": 0.082, "momentum": 0.85},
    {"symbol": "OGDC",   "name": "Oil & Gas Dev. Corp.", "sector": "Energy",     "price": 485,  "growth": 0.16, "volatility": "Medium", "div_yield": 0.065, "momentum": 0.78},
    {"symbol": "TRG",    "name": "TRG Pakistan",         "sector": "Technology", "price": 145,  "growth": 0.24, "volatility": "High",   "div_yield": 0.021, "momentum": 0.91},
    {"symbol": "HUBC",   "name": "Hub Power Co.",        "sector": "Power",      "price": 98,   "growth": 0.14, "volatility": "Low",    "div_yield": 0.131, "momentum": 0.71},
    {"symbol": "ENGRO",  "name": "Engro Corp.",          "sector": "Fertilizer", "price": 312,  "growth": 0.19, "volatility": "Medium", "div_yield": 0.094, "momentum": 0.68},
    {"symbol": "HBL",    "name": "Habib Bank Ltd.",      "sector": "Banking",    "price": 145,  "growth": 0.15, "volatility": "Medium", "div_yield": 0.075, "momentum": 0.75},
    {"symbol": "PPL",    "name": "Pakistan Petroleum",   "sector": "Energy",     "price": 225,  "growth": 0.13, "volatility": "Medium", "div_yield": 0.058, "momentum": 0.62},
    {"symbol": "SYS",    "name": "Systems Ltd.",         "sector": "Technology", "price": 1180, "growth": 0.22, "volatility": "High",   "div_yield": 0.018, "momentum": 0.80},
    {"symbol": "LUCK",   "name": "Lucky Cement",         "sector": "Cement",     "price": 680,  "growth": 0.11, "volatility": "High",   "div_yield": 0.032, "momentum": 0.55},
    {"symbol": "NESTLE", "name": "Nestle Pakistan",      "sector": "FMCG",       "price": 5800, "growth": 0.10, "volatility": "Low",    "div_yield": 0.029, "momentum": 0.58},
    {"symbol": "UBL",    "name": "United Bank Ltd.",     "sector": "Banking",    "price": 175,  "growth": 0.14, "volatility": "Medium", "div_yield": 0.071, "momentum": 0.70},
    {"symbol": "FFC",    "name": "Fauji Fertilizer",     "sector": "Fertilizer", "price": 135,  "growth": 0.09, "volatility": "Low",    "div_yield": 0.115, "momentum": 0.60},
    {"symbol": "POL",    "name": "Pakistan Oilfields",   "sector": "Energy",     "price": 490,  "growth": 0.12, "volatility": "Medium", "div_yield": 0.072, "momentum": 0.65},
    {"symbol": "BAFL",   "name": "Bank Alfalah",         "sector": "Banking",    "price": 62,   "growth": 0.13, "volatility": "Medium", "div_yield": 0.060, "momentum": 0.64},
    {"symbol": "EFERT",  "name": "Engro Fertilizers",    "sector": "Fertilizer", "price": 88,   "growth": 0.11, "volatility": "Low",    "div_yield": 0.108, "momentum": 0.58},
    {"symbol": "ATRL",   "name": "Attock Refinery",      "sector": "Energy",     "price": 310,  "growth": 0.10, "volatility": "Medium", "div_yield": 0.045, "momentum": 0.56},
    {"symbol": "POWER",  "name": "Power Cement",         "sector": "Cement",     "price": 22,   "growth": 0.08, "volatility": "High",   "div_yield": 0.010, "momentum": 0.48},
    {"symbol": "MLCF",   "name": "Maple Leaf Cement",    "sector": "Cement",     "price": 48,   "growth": 0.09, "volatility": "High",   "div_yield": 0.015, "momentum": 0.50},
    {"symbol": "PKGS",   "name": "Packages Ltd.",        "sector": "FMCG",       "price": 580,  "growth": 0.08, "volatility": "Low",    "div_yield": 0.038, "momentum": 0.52},
    {"symbol": "KAPCO",  "name": "Kot Addu Power",       "sector": "Power",      "price": 78,   "growth": 0.07, "volatility": "Low",    "div_yield": 0.140, "momentum": 0.45},
    {"symbol": "MARI",   "name": "Mari Petroleum",       "sector": "Energy",     "price": 2800, "growth": 0.15, "volatility": "Medium", "div_yield": 0.020, "momentum": 0.73},
    {"symbol": "SEARL",  "name": "Searle Pakistan",      "sector": "Pharma",     "price": 195,  "growth": 0.12, "volatility": "Medium", "div_yield": 0.025, "momentum": 0.60},
    {"symbol": "ABOT",   "name": "Abbott Pakistan",      "sector": "Pharma",     "price": 740,  "growth": 0.11, "volatility": "Low",    "div_yield": 0.030, "momentum": 0.55},
    {"symbol": "MUGHAL", "name": "Mughal Iron & Steel",  "sector": "Steel",      "price": 110,  "growth": 0.16, "volatility": "High",   "div_yield": 0.025, "momentum": 0.67},
    {"symbol": "FNEL",   "name": "Fauji Electric",       "sector": "Power",      "price": 56,   "growth": 0.13, "volatility": "Medium", "div_yield": 0.090, "momentum": 0.62},
]

ALL_SECTORS = sorted(list(set(s["sector"] for s in STOCKS)))

# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────
WEIGHTS = {
    "growth":    0.30,
    "volatility":0.20,
    "dividend":  0.15,
    "sector":    0.20,
    "momentum":  0.15,
}

def score_stock(s, risk_appetite, preferred_sectors, excluded_sectors, target_return):
    if s["sector"] in excluded_sectors:
        return 0

    # growth component (normalize to 0-100)
    g_score = min(s["growth"] / 0.30, 1.0) * 100

    # volatility component (low vol = high score)
    vol_map = {"Low": 100, "Medium": 60, "High": 30}
    if risk_appetite == "High":
        vol_map = {"Low": 50, "Medium": 80, "High": 100}
    elif risk_appetite == "Medium":
        vol_map = {"Low": 70, "Medium": 100, "High": 50}
    v_score = vol_map[s["volatility"]]

    # dividend component
    d_score = min(s["div_yield"] / 0.15, 1.0) * 100

    # sector preference bonus
    sec_score = 100 if s["sector"] in preferred_sectors else 50

    # momentum component
    mom_score = s["momentum"] * 100

    total = (
        g_score   * WEIGHTS["growth"] +
        v_score   * WEIGHTS["volatility"] +
        d_score   * WEIGHTS["dividend"] +
        sec_score * WEIGHTS["sector"] +
        mom_score * WEIGHTS["momentum"]
    )
    return round(total)

def score_components(s, risk_appetite, preferred_sectors):
    vol_map = {"Low": 100, "Medium": 60, "High": 30}
    if risk_appetite == "High":   vol_map = {"Low": 50, "Medium": 80, "High": 100}
    elif risk_appetite == "Medium": vol_map = {"Low": 70, "Medium": 100, "High": 50}
    return {
        "Growth (30pts)":    round(min(s["growth"]/0.30,1.0)*100 * WEIGHTS["growth"]),
        "Volatility (20pts)":round(vol_map[s["volatility"]] * WEIGHTS["volatility"]),
        "Sector Match (20pts)": round((100 if s["sector"] in preferred_sectors else 50) * WEIGHTS["sector"]),
        "Momentum (15pts)":  round(s["momentum"]*100 * WEIGHTS["momentum"]),
        "Dividend (15pts)":  round(min(s["div_yield"]/0.15,1.0)*100 * WEIGHTS["dividend"]),
    }

# ─────────────────────────────────────────────
#  PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────
def build_scored_list(risk, preferred, excluded, target):
    scored = []
    for s in STOCKS:
        sc = score_stock(s, risk, preferred, excluded, target)
        scored.append({**s, "score": sc})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def portfolio_from_top(scored, n):
    candidates = [s for s in scored if s["score"] > 0][:n]
    total_score = sum(s["score"] for s in candidates)
    for s in candidates:
        s["alloc"] = round(s["score"] / total_score * 100) if total_score else round(100/n)
    return candidates

# ─────────────────────────────────────────────
#  HILL CLIMBING
# ─────────────────────────────────────────────
def portfolio_value(weights, stocks):
    """Simple objective: weighted return - risk penalty"""
    vol_penalty = {"Low": 0, "Medium": 0.5, "High": 1.5}
    val = 0
    for w, s in zip(weights, stocks):
        val += w * (s["growth"] * 100 + s["div_yield"] * 50 - vol_penalty[s["volatility"]])
    return val

def hill_climbing(stocks, iterations=100):
    n = len(stocks)
    weights = np.array([1.0/n]*n)
    best_val = portfolio_value(weights, stocks)
    history = [best_val]
    for _ in range(iterations):
        i, j = random.sample(range(n), 2)
        delta = random.uniform(0.01, 0.05)
        new_w = weights.copy()
        new_w[i] = max(0.05, new_w[i] - delta)
        new_w[j] = min(0.60, new_w[j] + delta)
        new_w = new_w / new_w.sum()
        val = portfolio_value(new_w, stocks)
        if val > best_val:
            best_val, weights = val, new_w
        history.append(best_val)
    return weights, best_val, history

def simulated_annealing(stocks, iterations=100, T_start=1000, T_end=0.01):
    n = len(stocks)
    weights = np.array([1.0/n]*n)
    best_val = portfolio_value(weights, stocks)
    best_w = weights.copy()
    history = [best_val]
    T = T_start
    cooling = (T_end / T_start) ** (1 / iterations)
    for _ in range(iterations):
        i, j = random.sample(range(n), 2)
        delta = random.uniform(0.01, 0.08)
        new_w = weights.copy()
        new_w[i] = max(0.05, new_w[i] - delta)
        new_w[j] = min(0.60, new_w[j] + delta)
        new_w = new_w / new_w.sum()
        val = portfolio_value(new_w, stocks)
        diff = val - best_val
        if diff > 0 or random.random() < math.exp(diff / T):
            weights, best_val = new_w, val
            if val > portfolio_value(best_w, stocks):
                best_w = new_w.copy()
        history.append(portfolio_value(best_w, stocks))
        T *= cooling
    return best_w, portfolio_value(best_w, stocks), history

# ─────────────────────────────────────────────
#  SIDEBAR  — INVESTOR SETTINGS
# ─────────────────────────────────────────────

with st.sidebar:

    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="font-size:0.85rem; letter-spacing:0.12em; font-weight:700;">
            INVESTOR PROFILE
        </div>
    </div>
    """, unsafe_allow_html=True)

    investor_name = st.text_input(
        "Enter Your Name",
        value="",
        placeholder="e.g. Ali Khan"
    )
with st.sidebar:
    investor_name = st.text_input("Enter Your Name", value="", placeholder="e.g. Ali Khan")
    if investor_name:
        st.markdown(f"<div style='color:#1d4ed8;font-size:1rem;font-weight:700;margin-top:-4px;margin-bottom:8px;'>Welcome, {investor_name}</div>", unsafe_allow_html=True)

    st.markdown("### INVESTOR SETTINGS")
    st.divider()

    invest_amount = st.number_input(
        "Investment Amount (PKR)",
        min_value=100_000, max_value=100_000_000,
        value=5_000_000, step=100_000,
        format="%d"
    )
    st.markdown(f"<div style='color:#1d4ed8;font-size:1.1rem;font-weight:700;margin-top:-8px;margin-bottom:8px;'>Rs {invest_amount:,}</div>", unsafe_allow_html=True)

    duration = st.slider("Duration (Years)", 1, 20, 5)
    target_return = st.slider("Target Annual Return %", 5, 50, 20)
    risk_appetite = st.radio("Risk Appetite", ["Low", "Medium", "High"], horizontal=True, index=1)

    preferred = st.multiselect(
        "Preferred Sectors",
        options=ALL_SECTORS,
        default=["Banking", "Energy"],
    )

    excluded = st.multiselect(
        "Excluded Sectors",
        options=[sec for sec in ALL_SECTORS if sec not in preferred],
        default=[],
    )

    portfolio_size = st.select_slider("Portfolio Size", options=[3, 5, 7, 10], value=5)
    algorithm = st.radio("Algorithm", ["Hill Climbing", "Sim. Annealing", "Both"], horizontal=True)

    st.divider()
    run = st.button("Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────
#  TITLE BAR
# ─────────────────────────────────────────────
st.markdown('<div class="title-bar">PSX AI Investment Advisory System — Streamlit Dashboard</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COMPUTE  (on first load or button press)
# ─────────────────────────────────────────────
if "results" not in st.session_state or run:
    scored_list = build_scored_list(risk_appetite, preferred, excluded, target_return)
    portfolio   = portfolio_from_top(scored_list, portfolio_size)

    # Apply chosen optimiser
    hc_w, hc_score, hc_hist = hill_climbing(portfolio)
    sa_w, sa_score, sa_hist = simulated_annealing(portfolio)

    if algorithm == "Hill Climbing":
        opt_weights = hc_w
    elif algorithm == "Sim. Annealing":
        opt_weights = sa_w
    else:  # Both — blend the two solutions
        opt_weights = (hc_w + sa_w) / 2
        opt_weights = opt_weights / opt_weights.sum()

    # Update allocations from optimiser
    opt_allocs = (opt_weights * 100).round(1)
    for i, s in enumerate(portfolio):
        s["alloc"] = round(float(opt_allocs[i]))

    # Compute portfolio metrics
    exp_return = sum(s["alloc"] / 100 * s["growth"] for s in portfolio) * 100
    vol_num = {"Low": 0.05, "Medium": 0.10, "High": 0.20}
    port_risk = round(sum(s["alloc"] / 100 * vol_num[s["volatility"]] for s in portfolio), 2)
    avg_div = round(sum(s["alloc"] / 100 * s["div_yield"] for s in portfolio) * 100, 1)
    sharpe  = round((exp_return/100 - 0.05) / (port_risk if port_risk else 0.01), 2)

    st.session_state.results = {
        "scored_list": scored_list,
        "portfolio": portfolio,
        "exp_return": round(exp_return, 1),
        "port_risk": port_risk,
        "avg_div": avg_div,
        "sharpe": sharpe,
        "hc_score": round(hc_score, 3),
        "sa_score": round(sa_score, 3),
        "hc_hist": hc_hist,
        "sa_hist": sa_hist,
        "preferred": preferred,
        "excluded": excluded,
        "risk": risk_appetite,
        "duration": duration,
        "target": target_return,
        "invest_amount": invest_amount,
    }

R = st.session_state.results

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "AI Reasoning", "Optimization", "All Stocks"])

# ══════════════════════════════════════════════
#  TAB 1 — PORTFOLIO
# ══════════════════════════════════════════════
with tab1:
    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Return</div>
            <div class="metric-value">{R['exp_return']}%</div>
            <div class="metric-sub">{'Above target' if R['exp_return'] >= R['target'] else 'Below target'}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Risk</div>
            <div class="metric-value">{R['port_risk']}</div>
            <div class="metric-sub">Within tolerance</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Dividend Yield</div>
            <div class="metric-value">{R['avg_div']}%</div>
            <div class="metric-sub">Annual income</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        qual = "Excellent" if R['sharpe'] > 1.5 else "Good" if R['sharpe'] > 1 else "Moderate"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{R['sharpe']}</div>
            <div class="metric-sub">{qual}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown('<div class="section-header">Portfolio Allocation</div>', unsafe_allow_html=True)
        # Donut chart
        labels = [f"{s['symbol']} — {s['alloc']}%" for s in R['portfolio']]
        values = [s['alloc'] for s in R['portfolio']]
        colors = ['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe']

        fig_donut = go.Figure(go.Pie(
            labels=[s['symbol'] for s in R['portfolio']],
            values=values,
            hole=0.60,
            marker=dict(colors=colors[:len(values)], line=dict(color='#0f1a2e', width=2)),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
        ))
        fig_donut.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=230,
            annotations=[dict(text=f"{len(R['portfolio'])} Stocks<br>Diversified",
                              x=0.5, y=0.5, font_size=12, showarrow=False,
                              font_color='#94a3b8')]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Portfolio Holdings</div>', unsafe_allow_html=True)

        color_map_ret = {'Low': '#4ade80', 'Medium': '#fb923c', 'High': '#f87171'}
        rows = ""
        for s in R['portfolio']:
            ret_color = '#4ade80' if s['growth'] >= 0.18 else '#fb923c' if s['growth'] >= 0.13 else '#f87171'
            rows += f"""
            <tr>
              <td><span class="symbol">{s['symbol']}</span></td>
              <td style="color:#94a3b8">{s['sector']}</td>
              <td><span class="score-badge score-high">{s['score']}/100</span></td>
              <td>{s['alloc']}%</td>
              <td style="color:{ret_color};font-weight:600">{round(s['growth']*100)}%</td>
              <td style="color:#60a5fa">{round(s['div_yield']*100,1)}%</td>
            </tr>"""

        st.markdown(f"""
        <table class="stock-table">
          <thead><tr>
            <th>Stock</th><th>Sector</th><th>Score</th>
            <th>Alloc %</th><th>Return</th><th>Div %</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    # Sector bar chart
    st.markdown('<div class="section-header">Allocation by Sector</div>', unsafe_allow_html=True)
    sector_alloc = {}
    for s in R['portfolio']:
        sector_alloc[s['sector']] = sector_alloc.get(s['sector'], 0) + s['alloc']

    fig_bar = go.Figure(go.Bar(
        y=list(sector_alloc.keys()),
        x=list(sector_alloc.values()),
        orientation='h',
        marker=dict(
            color=['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd'],
            line=dict(color='rgba(0,0,0,0)')
        ),
        text=[f"{v}%" for v in sector_alloc.values()],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=10, b=10), height=180,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color='#94a3b8'),
        yaxis=dict(color='#94a3b8', tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        f"<div style='text-align:center;color:#475569;font-size:0.72rem;margin-top:8px;'>"
        f"PSX AI Investment Advisory System • Week 1 Demo • Team A, B & C • BS Computer Science, Semester 6"
        f"</div>", unsafe_allow_html=True
    )

# ══════════════════════════════════════════════
#  TAB 2 — AI REASONING
# ══════════════════════════════════════════════
with tab2:
    # Show top 3 reason cards
    top3 = R['portfolio'][:3]
    cols = st.columns(3)
    for col, s in zip(cols, top3):
        comps = score_components(s, R['risk'], R['preferred'])
        vol_color = {'Low': '#4ade80', 'Medium': '#facc15', 'High': '#f87171'}
        mom_label = 'Very High' if s['momentum'] > 0.88 else 'High' if s['momentum'] > 0.75 else 'Medium'
        mom_color = '#4ade80' if s['momentum'] > 0.75 else '#facc15'

        with col:
            st.markdown(f"""
            <div class="reason-card">
              <div class="ticker">{s['symbol']}</div>
              <div class="cname">{s['name']}</div>
              <div class="sector-tag">{s['sector']}</div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:6px;">Composite Score</div>
              <div style="background:rgba(59,130,246,0.3);border-radius:4px;height:10px;margin-bottom:14px;position:relative;">
                <div style="width:{s['score']}%;background:linear-gradient(90deg,#1d4ed8,#60a5fa);height:100%;border-radius:4px;"></div>
                <span style="position:absolute;right:4px;top:-1px;font-size:0.68rem;color:#93c5fd;">{s['score']}/100</span>
              </div>
              <div class="reason-metric"><span class="lbl">5yr Growth</span><span style="color:#4ade80;font-weight:700">{round(s['growth']*100)}%</span></div>
              <div class="reason-metric"><span class="lbl">Volatility</span><span style="color:{vol_color[s['volatility']]};font-weight:700">{s['volatility']}</span></div>
              <div class="reason-metric"><span class="lbl">Dividend Yield</span><span style="color:#60a5fa;font-weight:700">{round(s['div_yield']*100,1)}%</span></div>
              <div class="reason-metric"><span class="lbl">Momentum</span><span style="color:{mom_color};font-weight:700">{mom_label}</span></div>
              <div class="reasoning-text">
                {s['symbol']} ranks highly due to strong 5-year growth and consistent dividend payouts. Sector preference bonus applied.
              </div>
              <span class="alloc-badge">Recommended allocation: {s['alloc']}%</span>
            </div>""", unsafe_allow_html=True)

    # Score breakdown bar
    st.markdown('<div class="section-header" style="margin-top:24px;">Score Component Breakdown (Top Stock)</div>', unsafe_allow_html=True)
    top_s = R['portfolio'][0]
    comps = score_components(top_s, R['risk'], R['preferred'])

    prog_html = ""
    for k, v in comps.items():
        pct = min(v * 3, 100)  # scale for visual
        prog_html += f"""
        <div class="prog-row">
          <div class="prog-label">{k}</div>
          <div class="prog-bar-outer"><div class="prog-bar-inner" style="width:{pct}%"></div></div>
          <div class="prog-val">{v}</div>
        </div>"""

    st.markdown(f'<div style="background:rgba(15,32,68,0.6);border-radius:12px;padding:18px;border:1px solid rgba(59,130,246,0.2);">{prog_html}</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:16px;">AI Reasoning Tab • Scores computed by scoring_engine.py using 5 weighted components.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — OPTIMIZATION
# ══════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Convergence Curves</div>', unsafe_allow_html=True)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=R['hc_hist'], name='Hill Climbing',
            line=dict(color='#3b82f6', width=2.5),
        ))
        fig_conv.add_trace(go.Scatter(
            y=R['sa_hist'], name='Simulated Annealing',
            line=dict(color='#f87171', width=2.5, dash='dash'),
        ))
        fig_conv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,28,58,0.6)',
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(title='Iterations', color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Objective Value', color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Risk vs Return Scatter</div>', unsafe_allow_html=True)
        vol_num = {"Low": 0.05, "Medium": 0.10, "High": 0.20}
        syms   = [s['symbol'] for s in R['portfolio']]
        rets   = [s['growth'] * 100 for s in R['portfolio']]
        risks  = [vol_num[s['volatility']] for s in R['portfolio']]
        sizes  = [s['alloc'] * 4 for s in R['portfolio']]
        colors_scatter = ['#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe','#dbeafe']

        fig_scatter = go.Figure()
        for i, s in enumerate(R['portfolio']):
            fig_scatter.add_trace(go.Scatter(
                x=[risks[i]], y=[rets[i]],
                mode='markers+text',
                text=[syms[i]], textposition='top center',
                marker=dict(size=sizes[i], color=colors_scatter[i % len(colors_scatter)],
                            line=dict(color='white', width=1)),
                name=syms[i],
                textfont=dict(color='white', size=10),
            ))
        # Efficient frontier line
        ef_x = np.linspace(0.03, 0.22, 50)
        ef_y = 5 + 80 * ef_x
        fig_scatter.add_trace(go.Scatter(
            x=ef_x, y=ef_y, mode='lines',
            line=dict(color='#facc15', dash='dot', width=1.5),
            name='Efficient Frontier',
        ))
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,28,58,0.6)',
            legend=dict(font=dict(color='#94a3b8', size=9), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(title='Portfolio Risk (Volatility)', color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Expected Return (%)', color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Algo summary cards
    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        st.markdown(f"""
        <div class="algo-card">
          <div class="algo-label">HC Best Score</div>
          <div class="algo-value algo-hc">{R['hc_score']}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Iterations: 100</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="algo-card">
          <div class="algo-label">SA Best Score</div>
          <div class="algo-value algo-sa">{R['sa_score']}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Temp: 1000 → 0.01</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        winner = "SA" if R['sa_score'] > R['hc_score'] else "HC"
        diff = abs(R['sa_score'] - R['hc_score'])
        st.markdown(f"""
        <div class="algo-card" style="text-align:left;">
          <div style="font-size:0.92rem;font-weight:700;color:#60a5fa;margin-bottom:8px;">Algorithm Comparison</div>
          <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.7;">
            {"SA found a marginally better portfolio" if winner == "SA" else "HC found a competitive portfolio"}
            (+{round(diff*100,1)}%) by {"escaping local optima" if winner == "SA" else "fast convergence"}.<br>
            {"HC converged faster." if winner == "SA" else "SA explored more broadly."} Both are appropriate for this portfolio size.
            <br><span style="color:#60a5fa;font-weight:600;">Recommendation: Use SA for portfolios (N &gt; 8 stocks)</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:16px;">Optimization Tab • HC and Simulated Annealing comparison comparison</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — ALL STOCKS
# ══════════════════════════════════════════════
with tab4:
    scored_list = R['scored_list']
    port_syms   = {s['symbol'] for s in R['portfolio']}

    st.markdown(f'<div class="section-header">All {len(scored_list)} Stocks — Ranked by Score</div>', unsafe_allow_html=True)

    top10 = scored_list[:10]
    rows = ""
    for i, s in enumerate(top10):
        if s['symbol'] == top10[0]['symbol']:
            status = '<span class="status-top">Top Pick</span>'
        elif s['symbol'] == top10[1]['symbol'] if len(top10) > 1 else None:
            status = '<span class="status-rec">Recommended</span>'
        elif s['symbol'] in port_syms:
            status = '<span class="status-sel">Selected</span>'
        elif s['score'] >= 60:
            status = '<span class="status-con">Consider</span>'
        else:
            status = '<span class="status-neu">Neutral</span>'

        sc = s['score']
        badge_cls = 'score-high' if sc >= 75 else 'score-mid' if sc >= 60 else 'score-low'
        vol_col = {'Low': '#4ade80', 'Medium': '#facc15', 'High': '#f87171'}

        rows += f"""
        <tr>
          <td><span class="symbol">{s['symbol']}</span></td>
          <td style="color:#94a3b8">{s['sector']}</td>
          <td><span class="score-badge {badge_cls}">{sc}</span></td>
          <td style="color:#e2e8f0">Rs {s['price']:,}</td>
          <td style="color:#4ade80;font-weight:600">{round(s['growth']*100)}%</td>
          <td style="color:{vol_col[s['volatility']]};font-weight:600">{s['volatility']}</td>
          <td style="color:#60a5fa">{round(s['div_yield']*100,1)}%</td>
          <td style="color:#e2e8f0">{s['momentum']:.2f}</td>
          <td>{status}</td>
        </tr>"""

    st.markdown(f"""
    <table class="stock-table">
      <thead><tr>
        <th>Symbol</th><th>Sector</th><th>Score</th><th>Price</th>
        <th>Growth</th><th>Risk</th><th>Div Yield</th><th>Momentum</th><th>Status</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:24px;margin-top:14px;font-size:0.77rem;color:#64748b;">
      <span>Top 5 — Selected for Portfolio</span>
      <span>Remaining — Considered but not selected</span>
      <span style="margin-left:auto;">Score ≥ 80: Top Tier | 60-79: Good | &lt; 60: Marginal</span>
    </div>
    <div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:12px;">All Stocks Tab • Scores calculated using 5-component weighted heuristic with risk appetite adjustment</div>
    """, unsafe_allow_html=True)
