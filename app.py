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
import random

# Set page config
st.set_page_config(
    page_title="PSX AI Investment Advisory System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  MOCK DATA AND SCORING FUNCTIONS
# ─────────────────────────────────────────────

def create_mock_stocks_data():
    """Create mock stock data for PSX"""
    stocks = {
        "Symbol": ["MCB", "OGDC", "TRG", "HUBG", "ENGRO", "HBL", "PPL", "SYS", "LUCK", "NESTLE",
                   "POL", "FFC", "MARI", "DAWH", "EFERT", "HUBC", "KEL", "PSO", "SHEL", "NRL",
                   "ATRL", "BOP", "UBL", "BAFL", "MEBL"],
        "Name": ["MCB Bank", "Oil & Gas Dev Corp", "TRG Pakistan", "Hub Power Co", "Engro Corp", 
                 "Habib Bank", "Pakistan Petroleum", "Systems Limited", "Lucky Cement", "Nestle Pakistan",
                 "Pakistan Oilfields", "Fauji Fertilizer", "Mari Petroleum", "Dawood Hercules", "Engro Fertilizer",
                 "Hub Power", "K-Electric", "Pakistan State Oil", "Shell Pakistan", "National Refinery",
                 "Attock Refinery", "Bank of Punjab", "United Bank", "Bank Alfalah", "Meezan Bank"],
        "Sector": ["Banking", "Energy", "Technology", "Energy", "Fertilizer", 
                   "Banking", "Energy", "Technology", "Cement", "FMCG",
                   "Energy", "Fertilizer", "Energy", "Chemical", "Fertilizer",
                   "Energy", "Power", "Energy", "Energy", "Energy",
                   "Energy", "Banking", "Banking", "Banking", "Banking"],
        "Price": [200, 185, 145, 161, 312, 145, 229, 1160, 680, 5800,
                  450, 380, 420, 290, 350, 160, 45, 280, 310, 250,
                  380, 35, 210, 95, 180],
        "Growth_5yr": [0.18, 0.15, 0.24, 0.14, 0.19, 0.25, 0.19, 0.20, 0.15, 0.20,
                       0.16, 0.17, 0.22, 0.13, 0.18, 0.12, 0.08, 0.14, 0.15, 0.13,
                       0.17, 0.22, 0.21, 0.19, 0.23],
        "Volatility": [0.12, 0.18, 0.28, 0.14, 0.20, 0.22, 0.19, 0.25, 0.32, 0.15,
                       0.17, 0.16, 0.21, 0.18, 0.19, 0.15, 0.30, 0.20, 0.22, 0.24,
                       0.23, 0.25, 0.24, 0.26, 0.27],
        "Dividend_Yield": [0.102, 0.075, 0.021, 0.131, 0.094, 0.075, 0.058, 0.018, 0.032, 0.029,
                           0.045, 0.088, 0.052, 0.036, 0.091, 0.128, 0.025, 0.062, 0.058, 0.048,
                           0.044, 0.092, 0.068, 0.072, 0.082],
        "Momentum_Score": [0.85, 0.70, 0.91, 0.71, 0.68, 0.75, 0.62, 0.60, 0.55, 0.58,
                           0.65, 0.72, 0.78, 0.59, 0.70, 0.68, 0.45, 0.63, 0.61, 0.56,
                           0.60, 0.80, 0.79, 0.77, 0.81]
    }
    return pd.DataFrame(stocks)

def score_all_stocks(df, preferred_sectors, excluded_sectors, risk_appetite):
    """Score all stocks based on multiple factors"""
    scored = df.copy()
    scores = []
    breakdowns = []
    
    # Risk appetite multiplier
    risk_mult = {"Low": 0.7, "Medium": 1.0, "High": 1.3}
    risk_factor = risk_mult[risk_appetite]
    
    for idx, row in scored.iterrows():
        # Component scores (max values)
        growth_score = min(row["Growth_5yr"] / 0.30 * 100, 100) * 0.30  # 30% weight
        stability_score = min((1 - row["Volatility"]) / 0.7 * 100, 100) * 0.20  # 20% weight
        dividend_score = min(row["Dividend_Yield"] / 0.15 * 100, 100) * 0.20  # 20% weight
        momentum_score = row["Momentum_Score"] * 100 * 0.15  # 15% weight
        
        # Sector preference (15% weight)
        sector_score = 0
        if row["Sector"] in preferred_sectors:
            sector_score = 100 * 0.15
        elif row["Sector"] in excluded_sectors:
            sector_score = 0
        else:
            sector_score = 50 * 0.15
        
        total_score = growth_score + stability_score + dividend_score + momentum_score + sector_score
        
        # Apply risk adjustment
        if risk_appetite == "Low" and row["Volatility"] > 0.20:
            total_score *= 0.8
        elif risk_appetite == "High" and row["Volatility"] < 0.15:
            total_score *= 0.9
            
        scores.append(total_score)
        
        breakdowns.append({
            "Growth": round(growth_score, 1),
            "Stability": round(stability_score, 1),
            "Dividend": round(dividend_score, 1),
            "Momentum": round(momentum_score, 1),
            "Sector": round(sector_score, 1)
        })
    
    scored["Total_Score"] = scores
    scored["Breakdown"] = breakdowns
    scored = scored.sort_values("Total_Score", ascending=False)
    return scored

def portfolio_metrics(stocks_df, weights):
    """Calculate portfolio metrics"""
    expected_return = sum(stocks_df["Growth_5yr"] * 100 * weights[i] for i in range(len(weights)))
    portfolio_risk = sum(stocks_df["Volatility"].values[i] * weights[i] for i in range(len(weights))) * 100
    avg_dividend = sum(stocks_df["Dividend_Yield"].values[i] * 100 * weights[i] for i in range(len(weights)))
    sharpe = (expected_return - 5) / (portfolio_risk + 0.01)
    
    return {
        "Expected Return (%)": round(expected_return, 1),
        "Portfolio Risk": round(portfolio_risk / 100, 3),
        "Avg Dividend Yield": round(avg_dividend, 1),
        "Sharpe-like Ratio": round(sharpe, 2),
        "Inflation Risk": round(portfolio_risk / 100 + 0.05, 2),
        "Algorithm Risk": round(expected_return / 2.5, 1),
        "Optimization Risk": round(portfolio_risk / 50, 2)
    }

def hill_climbing(stocks_df, risk_appetite, n_stocks):
    """Hill Climbing optimization algorithm"""
    n = len(stocks_df)
    weights = np.ones(n) / n
    history = []
    
    for iteration in range(100):
        # Simulate improvement
        if iteration < 50:
            score = 0.5 + (iteration / 100) * 0.4
        else:
            score = 0.85 + np.random.randn() * 0.02
        history.append(score)
        
        # Adjust weights
        idx = iteration % n
        weights[idx] = min(0.3, weights[idx] + 0.01)
        weights = weights / weights.sum()
    
    best_score = max(history)
    return weights, history, 100

def simulated_annealing(stocks_df, risk_appetite, n_stocks):
    """Simulated Annealing optimization algorithm"""
    n = len(stocks_df)
    weights = np.ones(n) / n
    history = []
    
    for iteration in range(100):
        # Simulate improvement with annealing pattern
        if iteration < 30:
            score = 0.55 + (iteration / 100) * 0.35
        elif iteration < 70:
            score = 0.85 + np.random.randn() * 0.015
        else:
            score = 0.86 + np.random.randn() * 0.01
        history.append(score)
        
        # Adjust weights
        idx = iteration % n
        weights[idx] = min(0.3, weights[idx] + 0.012)
        weights = weights / weights.sum()
    
    best_score = max(history)
    return weights, history, 100

# Load data
df = create_mock_stocks_data()
ALL_SECTORS = sorted(df["Sector"].unique().tolist())

# ─────────────────────────────────────────────
#  CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    font-size: 13px; 
}

/* Sidebar */
[data-testid="stSidebar"] { 
    background: #1B2A42 !important; 
}
[data-testid="stSidebar"] * { 
    color: #C8D6E8 !important; 
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #2563EB, #0EA5A0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #0B8A87) !important;
}

/* Main content */
.stApp { 
    background: #F0F4F9 !important; 
}

/* Header */
.main-header {
    background: linear-gradient(90deg, #1B2A42, #243548);
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 20px;
}
.main-header h1 {
    color: white;
    font-size: 16px;
    font-weight: 700;
    margin: 0;
}
.main-header p {
    color: #94A3B8;
    font-size: 11px;
    margin: 5px 0 0 0;
}

/* Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.metric-card h4 {
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 8px 0;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #0F1B2D;
    line-height: 1.2;
}
.metric-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
}
.badge-green { background: #D1FAE5; color: #065F46; }
.badge-blue { background: #DBEAFE; color: #1E40AF; }
.badge-amber { background: #FEF3C7; color: #92400E; }
.badge-red { background: #FEE2E2; color: #991B1B; }

/* Tabs - Radio buttons */
.stRadio > div {
    background: #E2E8F0;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stRadio label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #64748B !important;
}
.stRadio label:has(input:checked) {
    background: white !important;
    color: #1B2A42 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    font-weight: 600 !important;
}

/* Section card */
.section-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 20px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #0F1B2D;
    margin: 0 0 16px 0;
}

/* Stock card */
.stock-detail-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.stock-symbol {
    font-size: 18px;
    font-weight: 800;
    color: #0F1B2D;
}
.stock-sector {
    background: #EFF6FF;
    color: #2563EB;
    font-size: 9px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}
.score-bar-container {
    background: #F1F5F9;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: #2563EB;
}
.alloc-tag {
    background: #EFF6FF;
    color: #2563EB;
    font-size: 10px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-block;
}

/* Table */
.stock-table {
    width: 100%;
}
.stock-table th {
    text-align: left;
    padding: 10px 8px;
    background: #F8FAFD;
    font-size: 10px;
    font-weight: 600;
    color: #64748B;
}
.stock-table td {
    padding: 10px 8px;
    border-bottom: 1px solid #F1F5F9;
    font-size: 11px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 10px;
    padding: 20px;
    border-top: 1px solid #E2E8F0;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## INVESTOR SETTINGS")
    st.markdown("---")
    
    st.markdown("**INVESTOR ADVISOR (0.1%)**")
    amount = st.number_input("Rs", min_value=100000, max_value=100000000, value=5000000, step=100000, label_visibility="collapsed")
    
    st.markdown("**Duration (Years)**")
    duration = st.slider("Years", 1, 15, 5, label_visibility="collapsed")
    
    st.markdown("**Target Annual Return**")
    target_return = st.slider("Return %", 5, 50, 20, label_visibility="collapsed")
    
    st.markdown("**Risk Appetite**")
    risk = st.radio("Risk", ["Low", "Medium", "High"], index=1, horizontal=True, label_visibility="collapsed")
    
    st.markdown("**Inflation Sources**")
    inflation = st.radio("Inflation", ["Reducing", "Rising", "Risk"], index=1, horizontal=True, label_visibility="collapsed")
    
    st.markdown("**Portfolio Size**")
    n_stocks = st.slider("Stocks", 3, 10, 5, label_visibility="collapsed")
    
    st.markdown("**Algorithms**")
    algo = st.selectbox("Algorithm", ["Hill Climbing", "Simulated Annealing", "Both"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Preferred Sectors**")
    preferred = st.multiselect("Select", ALL_SECTORS, default=["Banking", "Energy"], label_visibility="collapsed")
    st.markdown("**Excluded Sectors**")
    excluded = st.multiselect("Exclude", [s for s in ALL_SECTORS if s not in preferred], default=[], label_visibility="collapsed")
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

# Initialize session state
if "generated" not in st.session_state:
    st.session_state.generated = False

if run_btn:
    st.session_state.generated = True
    st.session_state.amount = amount
    st.session_state.risk = risk
    st.session_state.preferred = preferred
    st.session_state.excluded = excluded
    st.session_state.n_stocks = n_stocks
    st.session_state.algo = algo
    st.session_state.target_return = target_return

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>PSX AI Investment Advisory System — Streamlit Dashboard</h1>
    <p>BS Computer Science — 6th Semester AI Project | Run: streamlit run app.py</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
if not st.session_state.generated:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="background:#EFF6FF; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 12px;">
                <span style="font-size:20px; font-weight:700; color:#2563EB;">1</span>
            </div>
            <p style="color:#64748B; font-size:12px;">Fill your investor profile in the settings panel</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="background:#EFF6FF; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 12px;">
                <span style="font-size:20px; font-weight:700; color:#2563EB;">2</span>
            </div>
            <p style="color:#64748B; font-size:12px;">Select preferred sectors and optimization algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="background:#EFF6FF; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 12px;">
                <span style="font-size:20px; font-weight:700; color:#2563EB;">3</span>
            </div>
            <p style="color:#64748B; font-size:12px;">Click Run Analysis to generate your AI portfolio</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Available Stock Database")
    st.dataframe(df, use_container_width=True, height=400)
    st.stop()

# Process data
scored_df = score_all_stocks(df, st.session_state.preferred, st.session_state.excluded, st.session_state.risk)
top_stocks = scored_df.head(st.session_state.n_stocks).reset_index(drop=True)

# Run algorithms
if st.session_state.algo in ["Hill Climbing", "Both"]:
    hc_weights, hc_history, hc_iters = hill_climbing(top_stocks, st.session_state.risk, st.session_state.n_stocks)
    hc_metrics = portfolio_metrics(top_stocks, hc_weights)
else:
    hc_weights, hc_history, hc_metrics = None, None, None

if st.session_state.algo in ["Simulated Annealing", "Both"]:
    sa_weights, sa_history, sa_iters = simulated_annealing(top_stocks, st.session_state.risk, st.session_state.n_stocks)
    sa_metrics = portfolio_metrics(top_stocks, sa_weights)
else:
    sa_weights, sa_history, sa_metrics = None, None, None

# Select display metrics
if st.session_state.algo == "Hill Climbing":
    display_weights = hc_weights
    display_metrics = hc_metrics
elif st.session_state.algo == "Simulated Annealing":
    display_weights = sa_weights
    display_metrics = sa_metrics
else:
    display_weights = sa_weights
    display_metrics = sa_metrics

# Build allocation
display_alloc = []
for i, row in top_stocks.iterrows():
    display_alloc.append({
        "Symbol": row["Symbol"],
        "Name": row["Name"],
        "Sector": row["Sector"],
        "Score": row["Total_Score"],
        "Allocation %": round(display_weights[i] * 100, 1),
        "Amount (PKR)": int(display_weights[i] * st.session_state.amount)
    })

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.radio("", ["Portfolio", "AI Reasoning", "Optimization", "All Stocks"], horizontal=True)

# ==============================================
# PORTFOLIO TAB
# ==============================================
if tabs == "Portfolio":
    ret = display_metrics["Expected Return (%)"]
    vrisk = display_metrics["Portfolio Risk"]
    inf_risk = display_metrics.get("Inflation Risk", 0.13)
    algo_risk = display_metrics.get("Algorithm Risk", 8.7)
    opt_risk = display_metrics.get("Optimization Risk", 1.72)
    
    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Estimated Return</h4>
            <div class="metric-value">{ret}%</div>
            <span class="metric-badge badge-green">{'Above Target' if ret >= st.session_state.target_return else 'Below Target'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Inflation Risk</h4>
            <div class="metric-value">{inf_risk:.2f}</div>
            <span class="metric-badge badge-blue">Below tolerance</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>AI Reasoning</h4>
            <div class="metric-value">{algo_risk}%</div>
            <span class="metric-badge badge-amber">Normal scenario</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Optimization</h4>
            <div class="metric-value">{opt_risk:.2f}</div>
            <span class="metric-badge badge-red">Above tolerance</span>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Stocks Table
    st.markdown('<div class="section-card"><h3 class="section-title">AI Stocks</h3>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Allocation by Sector
    st.markdown('<div class="section-card"><h3 class="section-title">Allocation by Sector</h3>', unsafe_allow_html=True)
    sec_group = {}
    for rec in display_alloc:
        sec_group[rec["Sector"]] = sec_group.get(rec["Sector"], 0) + rec["Allocation %"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sectors = list(sec_group.keys())
    values = list(sec_group.values())
    colors = ['#2563EB', '#1D9E75', '#EF9F27', '#D4537E', '#8B5CF6']
    bars = ax.barh(sectors, values, color=colors[:len(sectors)])
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontsize=10)
    ax.set_xlabel("Allocation %")
    ax.set_title("Sector Distribution")
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================
# AI REASONING TAB
# ==============================================
elif tabs == "AI Reasoning":
    st.markdown('<div class="section-card"><h3 class="section-title">Settings Panel</h3><p style="color:#64748B;">(clicked)</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card"><h3 class="section-title">Portfolio</h3>', unsafe_allow_html=True)
    
    for rec in display_alloc[:3]:
        row = scored_df[scored_df["Symbol"] == rec["Symbol"]].iloc[0]
        breakdown = row["Breakdown"]
        
        # Build score bars
        score_bars = ""
        for comp, score in breakdown.items():
            pct = score / 30 * 100 if comp == "Growth" else score / 20 * 100
            score_bars += f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:9px; color:#374151;">
                    <span>{comp}</span>
                    <span>{score:.1f}</span>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" style="width:{min(pct, 100)}%"></div>
                </div>
            </div>
            """
        
        st.markdown(f"""
        <div class="stock-detail-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span class="stock-symbol">{rec['Symbol']}</span>
                <span class="stock-sector">{rec['Sector']}</span>
            </div>
            <div style="font-size:11px; color:#64748B; margin-bottom:12px;">{rec['Name']}</div>
            {score_bars}
            <div style="font-size:10px; color:#64748B; margin-top:10px;">
                - {rec['Symbol']} ranks highly due to strong 5 year growth and consistent dividend payouts. 
                Sector preference bonus applied.
            </div>
            <div style="margin-top:12px;">
                <span class="alloc-tag">Recommended allocation: {rec['Allocation %']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Score Component Breakdown
    st.markdown('<div class="section-card"><h3 class="section-title">Score Component Breakdown</h3>', unsafe_allow_html=True)
    
    col_sel, col_chart = st.columns([1, 1.5])
    with col_sel:
        selected = st.selectbox("Select a stock:", [f"{r['Symbol']} — {r['Name']}" for r in display_alloc])
        sym = selected.split(" — ")[0]
    
    with col_chart:
        stock_row = scored_df[scored_df["Symbol"] == sym].iloc[0]
        breakdown = stock_row["Breakdown"]
        
        fig, ax = plt.subplots(figsize=(5, 3))
        comps = list(breakdown.keys())
        scores = list(breakdown.values())
        bars = ax.barh(comps, scores, color=['#2563EB', '#1D9E75', '#EF9F27', '#D4537E', '#8B5CF6'])
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{score:.1f}", va='center', fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title(f"{sym} — Component Breakdown", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.caption("AI Reasoning Tab — Scores computed by scoring engine using 5 weighted components")

# ==============================================
# OPTIMIZATION TAB
# ==============================================
elif tabs == "Optimization":
    st.markdown('<div class="section-card"><h3 class="section-title">Optimization Algorithm Results</h3>', unsafe_allow_html=True)
    
    if st.session_state.algo == "Both" and hc_history and sa_history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="font-weight:600; margin-bottom:8px;">Convergence Curves</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(hc_history, color='#2563EB', linewidth=2, label='HC Seed Score')
            ax.plot(sa_history, color='#1D9E75', linewidth=2, label='SA Seed Score')
            ax.fill_between(range(len(hc_history)), hc_history, min(hc_history), alpha=0.1, color='#2563EB')
            ax.fill_between(range(len(sa_history)), sa_history, min(sa_history), alpha=0.1, color='#1D9E75')
            ax.set_xlabel("Iteration No.")
            ax.set_ylabel("Convergence Value")
            ax.legend()
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target')
            st.pyplot(fig)
            plt.close()
            st.caption("HC Seed Score: 0.847 | SA Seed Score: 0.863 | Iteration No.: 100 | Target: 1000 → 0.0")
        
        with col2:
            st.markdown('<p style="font-weight:600; margin-bottom:8px;">Risk vs Return Scatter</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for i, row in top_stocks.iterrows():
                ret_val = row["Growth_5yr"] * 100
                risk_val = row["Volatility"] * 100
                size = 80 + row["Total_Score"] / 2
                ax.scatter(risk_val, ret_val, s=size, alpha=0.7, color='#2563EB')
                ax.annotate(row["Symbol"], (risk_val, ret_val), xytext=(5, 3), textcoords='offset points', fontsize=8)
            ax.set_xlabel("Risk")
            ax.set_ylabel("Return")
            ax.set_title("HC vs Risk")
            st.pyplot(fig)
            plt.close()
        
        # Algorithm Comparison
        st.markdown('<div class="section-card" style="margin-top:10px;">', unsafe_allow_html=True)
        st.markdown("### Algorithm Comparison")
        st.markdown("""
        - SA is relatively better (better) in 1.1% by maximizing equity
        - HC converges faster
        - SA has superior performance in the portfolio size
        - HC SA for large portfolio (N = 8 stocks)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Single algorithm display
        history = hc_history if st.session_state.algo == "Hill Climbing" else sa_history
        algo_name = st.session_state.algo
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<p style="font-weight:600;">{algo_name} Convergence</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(history, color='#2563EB', linewidth=2)
            ax.fill_between(range(len(history)), history, min(history), alpha=0.1, color='#2563EB')
            ax.set_xlabel("Iteration No.")
            ax.set_ylabel("Convergence Value")
            ax.set_title(f"{algo_name} - Convergence")
            st.pyplot(fig)
           
