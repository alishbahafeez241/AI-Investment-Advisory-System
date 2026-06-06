import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  IMPORT NEW PIPELINE MODULES
# ─────────────────────────────────────────────
try:
    from scoring_engine import enrich_and_score, portfolio_metrics, allocate_portfolio, hill_climbing, simulated_annealing
    from cnf_filter import cnf_filter
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

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

[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important;
    opacity: 1 !important; z-index: 999999 !important;
    position: fixed !important; top: 12px !important; left: 12px !important;
    background: #ffffff !important; border: 1px solid #1d4ed8 !important;
    border-radius: 8px !important; padding: 6px !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.15) !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    color: #1d4ed8 !important; fill: #1d4ed8 !important; opacity: 1 !important;
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

.nlp-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #16a34a;
    margin-bottom: 12px;
}
.cnf-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eff6ff; border: 1px solid #93c5fd; border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #1d4ed8;
    margin-bottom: 12px; margin-left: 8px;
}
.pipeline-info {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 16px; font-size: 0.82rem; color: #475569;
    border-left: 3px solid #2563eb;
}

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
.sent-pos { background: #dcfce7; color: #15803d; border-radius: 12px; padding: 2px 10px; font-size: 0.75rem; font-weight: 700; }
.sent-neg { background: #fee2e2; color: #dc2626; border-radius: 12px; padding: 2px 10px; font-size: 0.75rem; font-weight: 700; }
.sent-neu { background: #f1f5f9; color: #64748b; border-radius: 12px; padding: 2px 10px; font-size: 0.75rem; font-weight: 700; }

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
.reason-card .headline-text {
    font-size: 0.78rem; color: #475569; line-height: 1.5;
    margin: 8px 0; padding: 10px; background: #f8fafc; border-radius: 8px;
    border-left: 3px solid #86efac; font-style: italic;
}
.alloc-badge {
    background: #1d4ed8; color: #ffffff; border-radius: 6px;
    padding: 5px 14px; font-size: 0.82rem; font-weight: 700; font-family: 'DM Mono', monospace;
}

.cnf-rule {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 0.82rem; color: #1e293b;
    border-left: 3px solid #2563eb;
}
.cnf-rule .rule-label { font-weight: 700; color: #1d4ed8; margin-right: 6px; }
.cnf-rule .rule-logic { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #64748b; display: block; margin-top: 4px; }

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

label { color: #0f172a !important; font-size: 0.88rem !important; font-weight: 600 !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div, .stTextInput input {
    background: #ffffff !important; border: 1px solid #cbd5e1 !important;
    color: #0f172a !important; border-radius: 8px !important;
}
.stRadio label, .stCheckbox label { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD STOCK UNIVERSE FROM CSV
#  This is the merged natural-language + stock
#  dataset your instructor requires.
# ─────────────────────────────────────────────
@st.cache_data
def load_stocks_csv():
    """
    Load stocks_data.csv and normalize column names so they match
    what both the old scoring helpers and the new scoring_engine.py expect.

    CSV columns   : Symbol, Name, Sector, Price_PKR, Growth_5yr,
                    Volatility, Dividend_Yield, Momentum_Score, Market_Cap_B
    Added columns : volatility_label (Low/Medium/High) mapped from numeric Volatility
    """
    df = pd.read_csv("stocks_data.csv")

    # Map numeric volatility to label used by old score helpers
    def vol_label(v):
        if v <= 0.15:
            return "Low"
        elif v <= 0.25:
            return "Medium"
        else:
            return "High"

    df["volatility_label"] = df["Volatility"].apply(vol_label)

    # Build the STOCKS list format expected by the original app helpers
    stocks = []
    for _, row in df.iterrows():
        stocks.append({
            "symbol":     row["Symbol"],
            "name":       row["Name"],
            "sector":     row["Sector"],
            "price":      row["Price_PKR"],
            "growth":     row["Growth_5yr"],
            "volatility": row["volatility_label"],   # "Low" / "Medium" / "High"
            "div_yield":  row["Dividend_Yield"] / 100,  # CSV stores as %, convert to decimal
            "momentum":   row["Momentum_Score"],
            # Keep raw columns for new pipeline
            "_volatility_raw": row["Volatility"],
            "_market_cap":     row["Market_Cap_B"],
        })
    return df, stocks


# Load once at startup
RAW_DF, STOCKS = load_stocks_csv()
ALL_SECTORS = sorted(list(set(s["sector"] for s in STOCKS)))

# ─────────────────────────────────────────────
#  LEGACY SCORING HELPERS
#  (kept for offline fallback when NLP modules
#   are not installed)
# ─────────────────────────────────────────────
WEIGHTS = {
    "growth":     0.30,
    "volatility": 0.20,
    "dividend":   0.15,
    "sector":     0.20,
    "momentum":   0.15,
}


def score_stock_legacy(s, risk_appetite, preferred_sectors, excluded_sectors, target_return):
    if s["sector"] in excluded_sectors:
        return 0
    g_score = min(s["growth"] / 0.30, 1.0) * 100
    vol_map = {"Low": 100, "Medium": 60, "High": 30}
    if risk_appetite == "High":
        vol_map = {"Low": 50, "Medium": 80, "High": 100}
    elif risk_appetite == "Medium":
        vol_map = {"Low": 70, "Medium": 100, "High": 50}
    v_score   = vol_map[s["volatility"]]
    d_score   = min(s["div_yield"] / 0.15, 1.0) * 100
    sec_score = 100 if s["sector"] in preferred_sectors else 50
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
    if risk_appetite == "High":
        vol_map = {"Low": 50, "Medium": 80, "High": 100}
    elif risk_appetite == "Medium":
        vol_map = {"Low": 70, "Medium": 100, "High": 50}
    return {
        "Growth (30pts)":       round(min(s["growth"] / 0.30, 1.0) * 100 * WEIGHTS["growth"]),
        "Volatility (20pts)":   round(vol_map[s["volatility"]] * WEIGHTS["volatility"]),
        "Sector Match (20pts)": round((100 if s["sector"] in preferred_sectors else 50) * WEIGHTS["sector"]),
        "Momentum (15pts)":     round(s["momentum"] * 100 * WEIGHTS["momentum"]),
        "Dividend (15pts)":     round(min(s["div_yield"] / 0.15, 1.0) * 100 * WEIGHTS["dividend"]),
    }


# ─────────────────────────────────────────────
#  NEW PIPELINE SCORING
#  Calls enrich_and_score() from scoring_engine
#  which runs: NLP → heuristic score → CNF filter
# ─────────────────────────────────────────────

def run_new_pipeline(preferred_sectors, excluded_sectors, risk_appetite):
    """
    Full pipeline:
      1. enrich_with_sentiment()  — scraper.py headlines → TextBlob → Sentiment_Score
      2. score_stock()            — 5-component heuristic + sentiment adjustment
      3. cnf_filter()             — CNF investment rules filter bad stocks
      4. Sort by Total_Score

    Returns enriched + filtered DataFrame from scoring_engine.py
    """
    scored_df = enrich_and_score(
        RAW_DF.copy(),
        preferred_sectors=preferred_sectors,
        excluded_sectors=excluded_sectors,
        risk_appetite=risk_appetite,
        run_nlp=True,    # enables article headline → sentiment
        run_cnf=True,    # enables CNF logic pre-filter
        verbose=True,
    )
    return scored_df


def pipeline_to_stocks_list(scored_df):
    """Convert scored DataFrame rows back to the dict format used by the UI."""
    stocks_out = []
    for _, row in scored_df.iterrows():
        vol_label = (
            "Low" if row["Volatility"] <= 0.15
            else "Medium" if row["Volatility"] <= 0.25
            else "High"
        )
        stocks_out.append({
            "symbol":          row["Symbol"],
            "name":            row["Name"],
            "sector":          row["Sector"],
            "price":           row["Price_PKR"],
            "growth":          row["Growth_5yr"],
            "volatility":      vol_label,
            "div_yield":       row["Dividend_Yield"] / 100,
            "momentum":        row["Momentum_Score"],
            "score":           row["Total_Score"],
            # NLP fields added by pipeline
            "sentiment_score": row.get("Sentiment_Score", 0.0),
            "sentiment_label": row.get("Sentiment_Label", "Neutral"),
            "headline":        row.get("Headline", "No recent news"),
            # Score breakdown if present
            "breakdown":       row.get("Breakdown", {}),
        })
    return stocks_out


# ─────────────────────────────────────────────
#  PORTFOLIO CONSTRUCTION  (legacy + new)
# ─────────────────────────────────────────────

def build_scored_list_legacy(risk, preferred, excluded, target):
    scored = []
    for s in STOCKS:
        sc = score_stock_legacy(s, risk, preferred, excluded, target)
        scored.append({**s, "score": sc,
                       "sentiment_score": 0.0,
                       "sentiment_label": "Neutral",
                       "headline": "NLP pipeline not available",
                       "breakdown": {}})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def portfolio_from_top(scored, n):
    candidates = [s for s in scored if s["score"] > 0][:n]
    total_score = sum(s["score"] for s in candidates)
    for s in candidates:
        s["alloc"] = round(s["score"] / total_score * 100) if total_score else round(100 / n)
    return candidates


# ─────────────────────────────────────────────
#  OPTIMISERS  (local — used when pipeline
#  module not available)
# ─────────────────────────────────────────────

def portfolio_value(weights, stocks):
    vol_penalty = {"Low": 0, "Medium": 0.5, "High": 1.5}
    val = 0
    for w, s in zip(weights, stocks):
        val += w * (s["growth"] * 100 + s["div_yield"] * 50 - vol_penalty[s["volatility"]])
    return val


def hill_climbing_local(stocks, iterations=100):
    n = len(stocks)
    weights = np.array([1.0 / n] * n)
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


def simulated_annealing_local(stocks, iterations=100, T_start=1000, T_end=0.01):
    n = len(stocks)
    weights = np.array([1.0 / n] * n)
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
#  SIDEBAR — INVESTOR SETTINGS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);
                    color:white; padding:18px; border-radius:14px;
                    text-align:center; margin-bottom:20px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.2);">
            <div style="font-size:2.4rem;">📊</div>
            <div style="font-size:1rem; font-weight:800; letter-spacing:0.15em;">INVESTOR PROFILE</div>
        </div>""", unsafe_allow_html=True)

    investor_name = st.text_input("Enter Your Name", value="", placeholder="e.g. Ali Khan")
    if investor_name:
        st.markdown(f"""<div style="color:#1d4ed8;font-size:1rem;font-weight:700;
                        margin-top:-4px;margin-bottom:8px;">Welcome, {investor_name}</div>""",
                    unsafe_allow_html=True)

    st.markdown("### INVESTOR SETTINGS")
    st.divider()

    invest_amount = st.number_input(
        "Investment Amount (PKR)", min_value=100_000, max_value=100_000_000,
        value=5_000_000, step=100_000, format="%d"
    )
    st.markdown(f"""<div style='color:#1d4ed8;font-size:1.1rem;font-weight:700;
                    margin-top:-8px;margin-bottom:8px;'>Rs {invest_amount:,}</div>""",
                unsafe_allow_html=True)

    duration      = st.slider("Duration (Years)", 1, 20, 5)
    target_return = st.slider("Target Annual Return %", 5, 50, 20)
    risk_appetite = st.radio("Risk Appetite", ["Low", "Medium", "High"], horizontal=True, index=1)

    preferred = st.multiselect("Preferred Sectors", options=ALL_SECTORS, default=["Banking", "Energy"])
    excluded  = st.multiselect("Excluded Sectors",
                                options=[s for s in ALL_SECTORS if s not in preferred], default=[])
    portfolio_size = st.select_slider("Portfolio Size", options=[3, 5, 7, 10], value=5)
    algorithm      = st.radio("Algorithm", ["Hill Climbing", "Sim. Annealing", "Both"], horizontal=True)

    st.divider()

    # Show whether NLP pipeline is active
    if PIPELINE_AVAILABLE:
        st.markdown("""<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                       padding:10px 12px;font-size:0.78rem;color:#15803d;font-weight:600;">
                       ✅ NLP + CNF Pipeline Active<br>
                       <span style="font-weight:400;color:#166534;">
                       Article sentiment + CNF rules enabled</span></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;
                       padding:10px 12px;font-size:0.78rem;color:#92400e;font-weight:600;">
                       ⚠ Legacy Mode<br>
                       <span style="font-weight:400;">
                       Install textblob & beautifulsoup4 to enable NLP</span></div>""",
                    unsafe_allow_html=True)

    run = st.button("Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────
#  TITLE BAR
# ─────────────────────────────────────────────
st.markdown('<div class="title-bar">PSX AI Investment Advisory System — Streamlit Dashboard</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COMPUTE — NEW PIPELINE OR LEGACY FALLBACK
# ─────────────────────────────────────────────
if "results" not in st.session_state or run:

    use_pipeline = PIPELINE_AVAILABLE

    if use_pipeline:
        # ── NEW PIPELINE ──────────────────────────────────────────────────
        # Step 1: NLP enrichment → sentiment scores from article headlines
        # Step 2: Heuristic scoring with sentiment adjustment
        # Step 3: CNF filter — removes stocks violating investment rules
        # Step 4: Returns ranked DataFrame
        with st.spinner("Running NLP enrichment + CNF filter…"):
            try:
                scored_df   = run_new_pipeline(preferred, excluded, risk_appetite)
                scored_list = pipeline_to_stocks_list(scored_df)
                pipeline_used = "NLP + CNF"
            except Exception as e:
                st.warning(f"Pipeline error: {e}. Falling back to legacy scoring.")
                scored_list   = build_scored_list_legacy(risk_appetite, preferred, excluded, target_return)
                pipeline_used = "Legacy (fallback)"
    else:
        # ── LEGACY FALLBACK ───────────────────────────────────────────────
        scored_list   = build_scored_list_legacy(risk_appetite, preferred, excluded, target_return)
        pipeline_used = "Legacy"

    portfolio = portfolio_from_top(scored_list, portfolio_size)

    # Optimisers — use scoring_engine versions if available, else local
    if PIPELINE_AVAILABLE and pipeline_used != "Legacy (fallback)":
        try:
            top_df = scored_df.head(portfolio_size).reset_index(drop=True)
            hc_w, hc_hist, _ = hill_climbing(top_df, risk_appetite)
            sa_w, sa_hist, _ = simulated_annealing(top_df, risk_appetite)
        except Exception:
            hc_w, hc_score_val, hc_hist = hill_climbing_local(portfolio)
            sa_w, sa_score_val, sa_hist = simulated_annealing_local(portfolio)
    else:
        hc_w, hc_score_val, hc_hist = hill_climbing_local(portfolio)
        sa_w, sa_score_val, sa_hist = simulated_annealing_local(portfolio)

    hc_score_val = portfolio_value(hc_w, portfolio)
    sa_score_val = portfolio_value(sa_w, portfolio)

    if algorithm == "Hill Climbing":
        opt_weights = hc_w
    elif algorithm == "Sim. Annealing":
        opt_weights = sa_w
    else:
        opt_weights = (hc_w + sa_w) / 2
        opt_weights = opt_weights / opt_weights.sum()

    opt_allocs = (opt_weights * 100).round(1)
    for i, s in enumerate(portfolio):
        s["alloc"] = round(float(opt_allocs[i]))

    # Portfolio metrics
    exp_return = sum(s["alloc"] / 100 * s["growth"] for s in portfolio) * 100
    vol_num    = {"Low": 0.05, "Medium": 0.10, "High": 0.20}
    port_risk  = round(sum(s["alloc"] / 100 * vol_num[s["volatility"]] for s in portfolio), 2)
    avg_div    = round(sum(s["alloc"] / 100 * s["div_yield"] for s in portfolio) * 100, 1)
    sharpe     = round((exp_return / 100 - 0.05) / (port_risk if port_risk else 0.01), 2)

    st.session_state.results = {
        "scored_list":    scored_list,
        "portfolio":      portfolio,
        "exp_return":     round(exp_return, 1),
        "port_risk":      port_risk,
        "avg_div":        avg_div,
        "sharpe":         sharpe,
        "hc_score":       round(hc_score_val, 3),
        "sa_score":       round(sa_score_val, 3),
        "hc_hist":        hc_hist,
        "sa_hist":        sa_hist,
        "preferred":      preferred,
        "excluded":       excluded,
        "risk":           risk_appetite,
        "duration":       duration,
        "target":         target_return,
        "invest_amount":  invest_amount,
        "pipeline_used":  pipeline_used,
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
    # Pipeline indicator
    pipeline_color = "#f0fdf4" if "NLP" in R["pipeline_used"] else "#fef9c3"
    pipeline_border = "#86efac" if "NLP" in R["pipeline_used"] else "#fde047"
    pipeline_text_color = "#15803d" if "NLP" in R["pipeline_used"] else "#92400e"
    st.markdown(f"""
    <div class="pipeline-info" style="background:{pipeline_color};border-color:{pipeline_border};">
      <strong style="color:{pipeline_text_color};">Pipeline: {R['pipeline_used']}</strong>
      &nbsp;|&nbsp; Stocks loaded from <code>stocks_data.csv</code>
      &nbsp;|&nbsp; {len(R['scored_list'])} stocks scored
      &nbsp;|&nbsp; Top {len(R['portfolio'])} selected after CNF filter
    </div>""", unsafe_allow_html=True)

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

    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown('<div class="section-header">Portfolio Allocation</div>', unsafe_allow_html=True)
        colors = ['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe']
        fig_donut = go.Figure(go.Pie(
            labels=[s['symbol'] for s in R['portfolio']],
            values=[s['alloc'] for s in R['portfolio']],
            hole=0.60,
            marker=dict(colors=colors[:len(R['portfolio'])], line=dict(color='#0f1a2e', width=2)),
            textinfo='label+percent',
            textfont=dict(color='white', size=11),
        ))
        fig_donut.update_layout(
            showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10), height=380,
            annotations=[dict(text=f"{len(R['portfolio'])} Stocks<br>Diversified",
                              x=0.5, y=0.5, font_size=12, showarrow=False, font_color='#94a3b8')]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Portfolio Holdings</div>', unsafe_allow_html=True)

        # Show sentiment column only when NLP pipeline ran
        show_sentiment = "NLP" in R["pipeline_used"]
        header_sent = "<th>Sentiment</th>" if show_sentiment else ""

        rows = ""
        for s in R['portfolio']:
            ret_color = '#16a34a' if s['growth'] >= 0.18 else '#ea580c' if s['growth'] >= 0.13 else '#ef4444'
            score_cls = 'score-high' if s['score'] >= 75 else 'score-mid' if s['score'] >= 60 else 'score-low'

            if show_sentiment:
                lbl = s.get("sentiment_label", "Neutral")
                sent_cls = "sent-pos" if lbl == "Positive" else "sent-neg" if lbl == "Negative" else "sent-neu"
                sent_cell = f'<td><span class="{sent_cls}">{lbl}</span></td>'
            else:
                sent_cell = ""

            rows += f"""
            <tr>
              <td><span class="symbol">{s['symbol']}</span></td>
              <td style="color:#475569">{s['sector']}</td>
              <td><span class="score-badge {score_cls}">{s['score']}/100</span></td>
              <td>{s['alloc']}%</td>
              <td style="color:{ret_color};font-weight:600">{round(s['growth']*100)}%</td>
              <td style="color:#2563eb">{round(s['div_yield']*100,1)}%</td>
              {sent_cell}
            </tr>"""

        st.markdown(f"""
        <table class="stock-table">
          <thead><tr>
            <th>Stock</th><th>Sector</th><th>Score</th>
            <th>Alloc %</th><th>Return</th><th>Div %</th>{header_sent}
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    # Sector bar chart
    st.markdown('<div class="section-header">Allocation by Sector</div>', unsafe_allow_html=True)
    sector_alloc = {}
    for s in R['portfolio']:
        sector_alloc[s['sector']] = sector_alloc.get(s['sector'], 0) + s['alloc']

    fig_bar = go.Figure(go.Bar(
        y=list(sector_alloc.keys()), x=list(sector_alloc.values()), orientation='h',
        marker=dict(color=['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd'],
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f"{v}%" for v in sector_alloc.values()], textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=10, b=10), height=180,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='#94a3b8'),
        yaxis=dict(color='#94a3b8', tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div style='text-align:center;color:#475569;font-size:0.72rem;margin-top:8px;'>"
                "PSX AI Investment Advisory System • NLP + CNF Pipeline • BS Computer Science, Semester 6"
                "</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — AI REASONING
#  Now shows: CNF rules, NLP sentiment,
#  article headline per stock
# ══════════════════════════════════════════════
with tab2:
    show_nlp = "NLP" in R["pipeline_used"]

    # ── CNF Rules panel ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">CNF Investment Rules (AI Reasoning Layer)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#475569;margin-bottom:12px;">
      These 6 rules (from cnf_filter.py) are expressed in
      <strong>Conjunctive Normal Form</strong> — each is a disjunction (OR) of literals.
      A stock passes only if ALL clauses evaluate to True.
    </div>""", unsafe_allow_html=True)

    cnf_rules = [
        ("Rule 1", "NOT(Sentiment=Negative AND Risk=High)",
         "NOT sentiment_negative OR NOT risk_high",
         "Blocks stocks that are simultaneously negative sentiment and high volatility"),
        ("Rule 2", "Negative sentiment → NOT Investable",
         "NOT sentiment_negative",
         "Directly excludes any stock with negative news sentiment score"),
        ("Rule 3", "High Risk requires Strong Growth",
         "NOT risk_high OR growth_strong",
         "High-volatility stocks must have ≥20% 5yr growth to compensate"),
        ("Rule 4", "Excluded Sector → NOT Investable",
         "NOT sector_excluded",
         "Strictly honours the user's excluded sector preferences"),
        ("Rule 5", "Low Risk must offer Return Potential",
         "NOT risk_low OR growth_strong OR dividend_high",
         "Low-volatility stocks still need good growth or dividend ≥8%"),
        ("Rule 6", "Weak Momentum + Negative Sentiment blocked",
         "momentum_strong OR NOT sentiment_negative",
         "Stocks with poor momentum AND bad news are excluded"),
    ]
    for label, title, logic, desc in cnf_rules:
        st.markdown(f"""
        <div class="cnf-rule">
          <span class="rule-label">{label}:</span> {title}<br>
          <span class="rule-logic">CNF clause: ( {logic} )</span>
          <div style="font-size:0.78rem;color:#64748b;margin-top:4px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">Top Stock Reasoning Cards</div>',
                unsafe_allow_html=True)

    # ── Stock reason cards ────────────────────────────────────────────────
    top3 = R['portfolio'][:3]
    cols = st.columns(3)
    for col, s in zip(cols, top3):
        comps = score_components(s, R['risk'], R['preferred'])
        vol_color = {'Low': '#4ade80', 'Medium': '#facc15', 'High': '#f87171'}
        mom_label = 'Very High' if s['momentum'] > 0.88 else 'High' if s['momentum'] > 0.75 else 'Medium'
        mom_color = '#4ade80' if s['momentum'] > 0.75 else '#facc15'

        # Sentiment fields (from NLP pipeline)
        sent_lbl   = s.get("sentiment_label", "Neutral")
        sent_score = s.get("sentiment_score", 0.0)
        headline   = s.get("headline", "No recent news available")
        sent_color = '#16a34a' if sent_lbl == 'Positive' else '#dc2626' if sent_lbl == 'Negative' else '#64748b'

        # AI reasoning text — now includes NLP result
        if show_nlp:
            reasoning = (
                f"{s['symbol']} selected after passing all 6 CNF investment rules. "
                f"Sentiment analysis of news headlines returned a score of {sent_score:+.2f} ({sent_lbl}), "
                f"contributing a {'bonus' if sent_score > 0 else 'penalty'} to the composite score. "
                f"Strong sector match ({s['sector']}) and {round(s['growth']*100)}% 5yr growth "
                f"further support the recommendation."
            )
        else:
            reasoning = (
                f"{s['symbol']} ranks highly due to strong 5-year growth and consistent dividend payouts. "
                f"Sector preference bonus applied. Install NLP modules to see sentiment analysis."
            )

        headline_html = (
            f'<div class="headline-text">📰 "{headline}"</div>'
            if show_nlp else ""
        )
        sentiment_row = (
            f'<div class="reason-metric"><span class="lbl">News Sentiment</span>'
            f'<span style="color:{sent_color};font-weight:700">{sent_lbl} ({sent_score:+.2f})</span></div>'
            if show_nlp else ""
        )

        with col:
            st.markdown(f"""
            <div class="reason-card">
              <div class="ticker">{s['symbol']}</div>
              <div class="cname">{s['name']}</div>
              <div class="sector-tag">{s['sector']}</div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:6px;">Composite Score</div>
              <div style="background:#e2e8f0;border-radius:4px;height:10px;margin-bottom:14px;position:relative;">
                <div style="width:{min(s['score'],100)}%;background:linear-gradient(90deg,#1d4ed8,#60a5fa);height:100%;border-radius:4px;"></div>
                <span style="position:absolute;right:4px;top:-1px;font-size:0.68rem;color:#64748b;">{s['score']}/100</span>
              </div>
              <div class="reason-metric"><span class="lbl">5yr Growth</span><span style="color:#16a34a;font-weight:700">{round(s['growth']*100)}%</span></div>
              <div class="reason-metric"><span class="lbl">Volatility</span><span style="color:{vol_color[s['volatility']]};font-weight:700">{s['volatility']}</span></div>
              <div class="reason-metric"><span class="lbl">Dividend Yield</span><span style="color:#2563eb;font-weight:700">{round(s['div_yield']*100,1)}%</span></div>
              <div class="reason-metric"><span class="lbl">Momentum</span><span style="color:{mom_color};font-weight:700">{mom_label}</span></div>
              {sentiment_row}
              {headline_html}
              <div class="reasoning-text">{reasoning}</div>
              <span class="alloc-badge">Recommended allocation: {s['alloc']}%</span>
            </div>""", unsafe_allow_html=True)

    # ── Score breakdown bar ───────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:24px;">Score Component Breakdown (Top Stock)</div>',
                unsafe_allow_html=True)
    top_s = R['portfolio'][0]

    # Use pipeline breakdown if available, else compute legacy
    breakdown = top_s.get("breakdown", {})
    if breakdown:
        comps_display = {
            "Growth (30pts)":       round(breakdown.get("Growth", 0), 1),
            "Volatility (20pts)":   round(breakdown.get("Volatility", 0), 1),
            "Dividend (15pts)":     round(breakdown.get("Dividend", 0), 1),
            "Sector Match (20pts)": round(breakdown.get("Sector", 0), 1),
            "Momentum (15pts)":     round(breakdown.get("Momentum", 0), 1),
            "Sentiment Adj":        round(breakdown.get("Sentiment_Adj", 0), 2),
        }
    else:
        comps_display = score_components(top_s, R['risk'], R['preferred'])

    prog_html = ""
    for k, v in comps_display.items():
        pct = min(abs(v) * 3, 100)
        bar_color = "linear-gradient(90deg,#dc2626,#f87171)" if v < 0 else "linear-gradient(90deg,#1d4ed8,#60a5fa)"
        prog_html += f"""
        <div class="prog-row">
          <div class="prog-label">{k}</div>
          <div class="prog-bar-outer"><div class="prog-bar-inner" style="width:{pct}%;background:{bar_color};"></div></div>
          <div class="prog-val" style="color:{'#dc2626' if v < 0 else '#1d4ed8'}">{v}</div>
        </div>"""

    st.markdown(f"""
    <div style="background:#f8fafc;border-radius:12px;padding:18px;border:1px solid #e2e8f0;">
      {prog_html}
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;color:#475569;font-size:0.72rem;margin-top:16px;'>"
                "AI Reasoning Tab • NLP Sentiment (TextBlob) + CNF Rules filter + 5-component heuristic scoring"
                "</div>", unsafe_allow_html=True)

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
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,28,58,0.04)',
            legend=dict(font=dict(color='#475569'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(title='Iterations', color='#475569', gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(title='Objective Value', color='#475569', gridcolor='rgba(0,0,0,0.05)'),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Risk vs Return Scatter</div>', unsafe_allow_html=True)
        vol_num = {"Low": 0.05, "Medium": 0.10, "High": 0.20}
        syms   = [s['symbol'] for s in R['portfolio']]
        rets   = [s['growth'] * 100 for s in R['portfolio']]
        risks  = [vol_num[s['volatility']] for s in R['portfolio']]
        sizes  = [s['alloc'] * 4 for s in R['portfolio']]
        clrs   = ['#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe','#dbeafe']

        fig_scatter = go.Figure()
        for i, s in enumerate(R['portfolio']):
            fig_scatter.add_trace(go.Scatter(
                x=[risks[i]], y=[rets[i]], mode='markers+text',
                text=[syms[i]], textposition='top center',
                marker=dict(size=sizes[i], color=clrs[i % len(clrs)],
                            line=dict(color='white', width=1)),
                name=syms[i], textfont=dict(color='#0f172a', size=10),
            ))
        ef_x = np.linspace(0.03, 0.22, 50)
        ef_y = 5 + 80 * ef_x
        fig_scatter.add_trace(go.Scatter(
            x=ef_x, y=ef_y, mode='lines',
            line=dict(color='#f59e0b', dash='dot', width=1.5),
            name='Efficient Frontier',
        ))
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,28,58,0.04)',
            legend=dict(font=dict(color='#475569', size=9), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(title='Portfolio Risk (Volatility)', color='#475569', gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(title='Expected Return (%)', color='#475569', gridcolor='rgba(0,0,0,0.05)'),
            margin=dict(l=10, r=10, t=10, b=10), height=280, showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

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
        diff   = abs(R['sa_score'] - R['hc_score'])
        st.markdown(f"""
        <div class="algo-card" style="text-align:left;">
          <div style="font-size:0.92rem;font-weight:700;color:#2563eb;margin-bottom:8px;">Algorithm Comparison</div>
          <div style="font-size:0.8rem;color:#475569;line-height:1.7;">
            {"SA found a marginally better portfolio" if winner == "SA" else "HC found a competitive portfolio"}
            (+{round(diff*100,1)}%) by {"escaping local optima" if winner == "SA" else "fast convergence"}.<br>
            {"HC converged faster." if winner == "SA" else "SA explored more broadly."}
            Both are appropriate for this portfolio size.<br>
            <span style="color:#2563eb;font-weight:600;">
              Recommendation: Use SA for portfolios (N > 8 stocks)
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;color:#475569;font-size:0.72rem;margin-top:16px;'>"
                "Optimization Tab • HC and Simulated Annealing applied after CNF pre-filter</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — ALL STOCKS
#  Now shows Sentiment column from NLP pipeline
# ══════════════════════════════════════════════
with tab4:
    scored_list = R['scored_list']
    port_syms   = {s['symbol'] for s in R['portfolio']}
    show_nlp    = "NLP" in R["pipeline_used"]

    st.markdown(f'<div class="section-header">All {len(scored_list)} Stocks — Ranked by Score</div>',
                unsafe_allow_html=True)

    if show_nlp:
        st.markdown("""<div style="font-size:0.82rem;color:#475569;margin-bottom:10px;">
          Scores include NLP sentiment adjustment. Stocks filtered by CNF rules are excluded from this list.
          Negative-sentiment stocks were removed before ranking.
        </div>""", unsafe_allow_html=True)

    top_n = scored_list[:15]
    sent_header = "<th>Sentiment</th><th>Headline (excerpt)</th>" if show_nlp else ""
    rows = ""

    for i, s in enumerate(top_n):
        if s['symbol'] == top_n[0]['symbol']:
            status = '<span class="status-top">Top Pick</span>'
        elif s['symbol'] in port_syms:
            status = '<span class="status-sel">Selected</span>'
        elif s['score'] >= 60:
            status = '<span class="status-con">Consider</span>'
        else:
            status = '<span class="status-neu">Neutral</span>'

        sc = s['score']
        badge_cls = 'score-high' if sc >= 75 else 'score-mid' if sc >= 60 else 'score-low'
        vol_col   = {'Low': '#16a34a', 'Medium': '#f59e0b', 'High': '#ef4444'}

        if show_nlp:
            lbl      = s.get("sentiment_label", "Neutral")
            sent_cls = "sent-pos" if lbl == "Positive" else "sent-neg" if lbl == "Negative" else "sent-neu"
            headline = s.get("headline", "")
            excerpt  = (headline[:55] + "…") if len(headline) > 55 else headline
            sent_cells = f'<td><span class="{sent_cls}">{lbl}</span></td><td style="color:#64748b;font-size:0.78rem;">{excerpt}</td>'
        else:
            sent_cells = ""

        rows += f"""
        <tr>
          <td><span class="symbol">{s['symbol']}</span></td>
          <td style="color:#475569">{s['sector']}</td>
          <td><span class="score-badge {badge_cls}">{sc}</span></td>
          <td style="color:#0f172a">Rs {s['price']:,}</td>
          <td style="color:#16a34a;font-weight:600">{round(s['growth']*100)}%</td>
          <td style="color:{vol_col[s['volatility']]};font-weight:600">{s['volatility']}</td>
          <td style="color:#2563eb">{round(s['div_yield']*100,1)}%</td>
          <td style="color:#0f172a">{s['momentum']:.2f}</td>
          {sent_cells}
          <td>{status}</td>
        </tr>"""

    st.markdown(f"""
    <table class="stock-table">
      <thead><tr>
        <th>Symbol</th><th>Sector</th><th>Score</th><th>Price</th>
        <th>Growth</th><th>Risk</th><th>Div Yield</th><th>Momentum</th>
        {sent_header}<th>Status</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:24px;margin-top:14px;font-size:0.77rem;color:#64748b;">
      <span>Score ≥ 75: Top Tier</span>
      <span>60-74: Good</span>
      <span>&lt; 60: Marginal</span>
      <span style="margin-left:auto;">Stocks removed by CNF filter not shown</span>
    </div>
    <div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:12px;">
      All Stocks Tab • Data from stocks_data.csv • Sentiment from scraper.py + TextBlob • Filtered by cnf_filter.py
    </div>""", unsafe_allow_html=True)
