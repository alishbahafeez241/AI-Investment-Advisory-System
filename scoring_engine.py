import pandas as pd
import numpy as np
import random
import math


# ─────────────────────────────────────────────
#  HEURISTIC SCORING
# ─────────────────────────────────────────────

WEIGHTS = {
    "growth":    0.30,   # 30 pts max
    "volatility":0.20,   # 20 pts  (inverse — lower is better)
    "dividend":  0.15,   # 15 pts
    "sector":    0.20,   # 20 pts  (sector match bonus)
    "momentum":  0.15,   # 15 pts
}

def score_stock(row, preferred_sectors, risk_appetite):
    """
    Score a single stock (0–100).
    risk_appetite: 'Low', 'Medium', 'High'
    """
    risk_multiplier = {"Low": 1.5, "Medium": 1.0, "High": 0.6}[risk_appetite]

    # Growth score (0–30)
    growth_score = min(row["Growth_5yr"] / 0.40, 1.0) * 30

    # Volatility score (0–20) — lower volatility → higher score
    vol_penalty = row["Volatility"] * risk_multiplier
    volatility_score = max(0, (1 - vol_penalty / 0.5)) * 20

    # Dividend score (0–15)
    dividend_score = min(row["Dividend_Yield"] / 15.0, 1.0) * 15

    # Sector match (0–20)
    sector_score = 20 if row["Sector"] in preferred_sectors else 8

    # Momentum score (0–15)
    momentum_score = row["Momentum_Score"] * 15

    total = growth_score + volatility_score + dividend_score + sector_score + momentum_score

    return round(total, 1), {
        "Growth":    round(growth_score, 1),
        "Volatility":round(volatility_score, 1),
        "Dividend":  round(dividend_score, 1),
        "Sector":    round(sector_score, 1),
        "Momentum":  round(momentum_score, 1),
    }


def score_all(df, preferred_sectors, excluded_sectors, risk_appetite):
    results = []
    for _, row in df.iterrows():
        if row["Sector"] in excluded_sectors:
            continue
        total, breakdown = score_stock(row, preferred_sectors, risk_appetite)
        results.append({**row.to_dict(), "Total_Score": total, "Breakdown": breakdown})
    scored = pd.DataFrame(results)
    return scored.sort_values("Total_Score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  PORTFOLIO ALLOCATION HELPER
# ─────────────────────────────────────────────

def allocate_portfolio(top_stocks, weights_arr):
    """Given stocks and weights array (sums to 1), return allocation %"""
    alloc = []
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        alloc.append({
            "Symbol":     row["Symbol"],
            "Name":       row["Name"],
            "Sector":     row["Sector"],
            "Score":      row["Total_Score"],
            "Allocation": round(weights_arr[i] * 100, 1),
        })
    return alloc


def portfolio_metrics(top_stocks, weights_arr):
    returns = np.array(top_stocks["Growth_5yr"].values)
    vols    = np.array(top_stocks["Volatility"].values)
    divs    = np.array(top_stocks["Dividend_Yield"].values)

    exp_return  = float(np.dot(weights_arr, returns) * 100)
    avg_risk    = float(np.dot(weights_arr, vols))
    avg_div     = float(np.dot(weights_arr, divs))
    sharpe      = exp_return / (avg_risk * 100 + 1e-9)

    return {
        "Expected Return (%)": round(exp_return, 2),
        "Portfolio Risk":       round(avg_risk, 3),
        "Avg Dividend Yield":   round(avg_div, 2),
        "Sharpe-like Ratio":    round(sharpe, 2),
    }


# ─────────────────────────────────────────────
#  HILL CLIMBING
# ─────────────────────────────────────────────

def objective(weights, returns, vols, risk_appetite):
    """Maximize return, penalize risk based on appetite."""
    penalty = {"Low": 3.0, "Medium": 1.5, "High": 0.5}[risk_appetite]
    exp_ret = np.dot(weights, returns)
    exp_vol = np.dot(weights, vols)
    return exp_ret - penalty * exp_vol


def normalize(w):
    w = np.clip(w, 0.02, 0.60)
    return w / w.sum()


def hill_climbing(top_stocks, risk_appetite, n_stocks=7, iterations=500, step=0.03):
    stocks = top_stocks.head(n_stocks)
    returns = stocks["Growth_5yr"].values
    vols    = stocks["Volatility"].values

    # Start: score-proportional weights
    scores = stocks["Total_Score"].values
    current = normalize(scores / scores.sum())
    best_score = objective(current, returns, vols, risk_appetite)

    history = [best_score]
    for _ in range(iterations):
        i, j = random.sample(range(n_stocks), 2)
        delta = random.uniform(0, step)
        candidate = current.copy()
        candidate[i] += delta
        candidate[j] -= delta
        candidate = normalize(candidate)
        score = objective(candidate, returns, vols, risk_appetite)
        if score > best_score:
            current, best_score = candidate, score
        history.append(best_score)

    return current, history, len(history)


# ─────────────────────────────────────────────
#  SIMULATED ANNEALING
# ─────────────────────────────────────────────

def simulated_annealing(top_stocks, risk_appetite, n_stocks=7,
                         T=1.0, T_min=0.001, alpha=0.995, step=0.05):
    stocks  = top_stocks.head(n_stocks)
    returns = stocks["Growth_5yr"].values
    vols    = stocks["Volatility"].values

    scores  = stocks["Total_Score"].values
    current = normalize(scores / scores.sum())
    best    = current.copy()
    best_score = objective(current, returns, vols, risk_appetite)

    history = [best_score]
    iters = 0
    while T > T_min:
        i, j = random.sample(range(n_stocks), 2)
        delta = random.uniform(0, step)
        candidate = current.copy()
        candidate[i] += delta
        candidate[j] -= delta
        candidate = normalize(candidate)

        delta_score = objective(candidate, returns, vols, risk_appetite) - \
                      objective(current,   returns, vols, risk_appetite)

        if delta_score > 0 or random.random() < math.exp(delta_score / T):
            current = candidate
            if objective(current, returns, vols, risk_appetite) > best_score:
                best = current.copy()
                best_score = objective(current, returns, vols, risk_appetite)

        T *= alpha
        iters += 1
        history.append(best_score)

    return best, history, iters
