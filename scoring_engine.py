
import pandas as pd
import numpy as np
import random
import math

# ── NEW IMPORTS ───────────────────────────────────────────────────────────────
from nlp_sentiment import enrich_with_sentiment
from cnf_filter    import cnf_filter
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  HEURISTIC SCORING
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "growth":     0.30,   # 30 pts max
    "volatility": 0.20,   # 20 pts  (inverse — lower is better)
    "dividend":   0.15,   # 15 pts
    "sector":     0.20,   # 20 pts  (sector match bonus)
    "momentum":   0.15,   # 15 pts
}


def score_stock(row, preferred_sectors, risk_appetite):
    """
    Score a single stock (0-100).
    risk_appetite: 'Low', 'Medium', 'High'
    """
    risk_multiplier = {"Low": 1.5, "Medium": 1.0, "High": 0.6}[risk_appetite]

    # Growth score (0-30)
    growth_score = min(row["Growth_5yr"] / 0.40, 1.0) * 30

    # Volatility score (0-20) — lower volatility -> higher score
    vol_penalty     = row["Volatility"] * risk_multiplier
    volatility_score = max(0, (1 - vol_penalty / 0.5)) * 20

    # Dividend score (0-15)
    dividend_score = min(row["Dividend_Yield"] / 15.0, 1.0) * 15

    # Sector match (0-20)
    sector_score = 20 if row["Sector"] in preferred_sectors else 8

    # Momentum score (0-15)
    momentum_score = row["Momentum_Score"] * 15

    # ── Sentiment bonus / penalty ─────────────────────────────────────────
    # Positive sentiment adds up to +5 pts; negative subtracts up to 5 pts.
    sentiment_adj = row.get("Sentiment_Score", 0.0) * 5.0
    # ─────────────────────────────────────────────────────────────────────

    total = (
        growth_score + volatility_score + dividend_score
        + sector_score + momentum_score + sentiment_adj
    )
    total = round(max(0.0, min(total, 105.0)), 1)   # cap at 105 to allow small bonus

    return total, {
        "Growth":         round(growth_score, 1),
        "Volatility":     round(volatility_score, 1),
        "Dividend":       round(dividend_score, 1),
        "Sector":         round(sector_score, 1),
        "Momentum":       round(momentum_score, 1),
        "Sentiment_Adj":  round(sentiment_adj, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ENRICHMENT + SCORING + CNF FILTER  (new combined pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_and_score(
    df: pd.DataFrame,
    preferred_sectors: list,
    excluded_sectors:  list,
    risk_appetite:     str,
    run_nlp:           bool = True,
    run_cnf:           bool = True,
    verbose:           bool = True,
) -> pd.DataFrame:
    """
    Full pipeline:
      1. NLP enrichment  — add Sentiment_Score / Sentiment_Label / Headline
      2. Heuristic score — compute Total_Score per stock
      3. CNF pre-filter  — remove stocks violating investment rules
      4. Sort            — return ranked, investable candidates

    Parameters
    ----------
    df                : raw stocks DataFrame (from stocks_data.csv)
    preferred_sectors : sectors the user selected
    excluded_sectors  : sectors the user wants excluded
    risk_appetite     : 'Low' | 'Medium' | 'High'
    run_nlp           : set False to skip NLP (e.g., offline tests)
    run_cnf           : set False to skip CNF filter (baseline comparison)
    verbose           : print CNF filter details to console

    Returns
    -------
    Scored + filtered DataFrame, sorted by Total_Score descending.
    """

    # ── Step 1: NLP enrichment ────────────────────────────────────────────
    if run_nlp:
        df = enrich_with_sentiment(df)
    else:
        if "Sentiment_Score" not in df.columns:
            df = df.copy()
            df["Sentiment_Score"] = 0.0
            df["Sentiment_Label"] = "Neutral"
            df["Headline"]        = "N/A"

    # ── Step 2: Heuristic scoring ─────────────────────────────────────────
    results = []
    for _, row in df.iterrows():
        total, breakdown = score_stock(row, preferred_sectors, risk_appetite)
        results.append({**row.to_dict(), "Total_Score": total, "Breakdown": breakdown})

    scored = pd.DataFrame(results)

    # ── Step 3: CNF pre-filter ────────────────────────────────────────────
    if run_cnf:
        scored = cnf_filter(
            scored,
            risk_appetite=risk_appetite,
            preferred_sectors=preferred_sectors,
            excluded_sectors=excluded_sectors,
            verbose=verbose,
        )
    else:
        # Still respect excluded sectors even without full CNF logic
        scored = scored[~scored["Sector"].isin(excluded_sectors)]

    # ── Step 4: Sort by Total_Score ───────────────────────────────────────
    return scored.sort_values("Total_Score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY score_all  (kept for backward compatibility with existing UI code)
# ─────────────────────────────────────────────────────────────────────────────

def score_all(df, preferred_sectors, excluded_sectors, risk_appetite):
    """
    Original interface retained for backward compatibility.
    Internally calls enrich_and_score with run_nlp=True, run_cnf=True.
    """
    return enrich_and_score(
        df,
        preferred_sectors=preferred_sectors,
        excluded_sectors=excluded_sectors,
        risk_appetite=risk_appetite,
        run_nlp=True,
        run_cnf=True,
        verbose=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO ALLOCATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def allocate_portfolio(top_stocks, weights_arr):
    """Given stocks and weights array (sums to 1), return allocation %."""
    alloc = []
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        alloc.append({
            "Symbol":     row["Symbol"],
            "Name":       row["Name"],
            "Sector":     row["Sector"],
            "Score":      row["Total_Score"],
            "Sentiment":  row.get("Sentiment_Label", "N/A"),
            "Allocation": round(weights_arr[i] * 100, 1),
        })
    return alloc


def portfolio_metrics(top_stocks, weights_arr):
    returns = np.array(top_stocks["Growth_5yr"].values)
    vols    = np.array(top_stocks["Volatility"].values)
    divs    = np.array(top_stocks["Dividend_Yield"].values)

    exp_return = float(np.dot(weights_arr, returns) * 100)
    avg_risk   = float(np.dot(weights_arr, vols))
    avg_div    = float(np.dot(weights_arr, divs))
    sharpe     = exp_return / (avg_risk * 100 + 1e-9)

    return {
        "Expected Return (%)": round(exp_return, 2),
        "Portfolio Risk":       round(avg_risk, 3),
        "Avg Dividend Yield":   round(avg_div, 2),
        "Sharpe-like Ratio":    round(sharpe, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HILL CLIMBING
# ─────────────────────────────────────────────────────────────────────────────

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
    stocks  = top_stocks.head(n_stocks)
    returns = stocks["Growth_5yr"].values
    vols    = stocks["Volatility"].values

    scores  = stocks["Total_Score"].values
    current = normalize(scores / scores.sum())
    best_score = objective(current, returns, vols, risk_appetite)

    history = [best_score]
    for _ in range(iterations):
        i, j   = random.sample(range(n_stocks), 2)
        delta  = random.uniform(0, step)
        candidate = current.copy()
        candidate[i] += delta
        candidate[j] -= delta
        candidate = normalize(candidate)
        score = objective(candidate, returns, vols, risk_appetite)
        if score > best_score:
            current, best_score = candidate, score
        history.append(best_score)

    return current, history, len(history)


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────────────────────

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
    iters   = 0
    while T > T_min:
        i, j   = random.sample(range(n_stocks), 2)
        delta  = random.uniform(0, step)
        candidate = current.copy()
        candidate[i] += delta
        candidate[j] -= delta
        candidate = normalize(candidate)

        delta_score = (objective(candidate, returns, vols, risk_appetite)
                       - objective(current,   returns, vols, risk_appetite))

        if delta_score > 0 or random.random() < math.exp(delta_score / T):
            current = candidate
            if objective(current, returns, vols, risk_appetite) > best_score:
                best       = current.copy()
                best_score = objective(current, returns, vols, risk_appetite)

        T    *= alpha
        iters += 1
        history.append(best_score)

    return best, history, iters
