"""
cnf_filter.py
-------------
Converts natural-language investment rules into Conjunctive Normal Form (CNF)
and applies them as a pre-filter before Hill Climbing / Simulated Annealing.

=============================================================================
CNF THEORY (brief)
=============================================================================
A propositional formula in CNF is a conjunction (AND) of clauses,
where each clause is a disjunction (OR) of literals (atoms or their negations).

Example rule:
  "If Sentiment is Positive AND Risk is Low  →  Investable"

As an implication:
  (Sentiment_Positive AND Risk_Low) → Investable

Equivalently (contrapositive form, then distributed to CNF clauses):
  NOT(Sentiment_Positive AND Risk_Low) OR Investable
  = NOT Sentiment_Positive OR NOT Risk_Low OR Investable

That single disjunction IS a CNF clause.  Combining all rules yields
a conjunction of such clauses — i.e., the full CNF formula.

Runtime evaluation:
  A stock passes the CNF filter only if ALL clauses evaluate to True for it.
=============================================================================

Public API
----------
  cnf_filter(df, risk_appetite, user_prefs) -> pd.DataFrame
      Returns the sub-DataFrame of stocks that satisfy every CNF clause.
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  ATOM DEFINITIONS
#  Each atom is a predicate: (row, params) -> bool
# ─────────────────────────────────────────────────────────────────────────────

def atom_sentiment_positive(row, _):
    return row.get("Sentiment_Score", 0.0) > 0.05

def atom_sentiment_negative(row, _):
    return row.get("Sentiment_Score", 0.0) < -0.05

def atom_sentiment_neutral(row, _):
    score = row.get("Sentiment_Score", 0.0)
    return -0.05 <= score <= 0.05

def atom_risk_low(row, _):
    return row["Volatility"] <= 0.15

def atom_risk_high(row, _):
    return row["Volatility"] > 0.25

def atom_growth_strong(row, _):
    return row["Growth_5yr"] >= 0.20

def atom_dividend_high(row, _):
    return row["Dividend_Yield"] >= 8.0

def atom_momentum_strong(row, _):
    return row["Momentum_Score"] >= 0.75

def atom_sector_preferred(row, params):
    preferred = params.get("preferred_sectors", [])
    return row["Sector"] in preferred

def atom_sector_excluded(row, params):
    excluded = params.get("excluded_sectors", [])
    return row["Sector"] in excluded

def atom_score_above_threshold(row, params):
    threshold = params.get("score_threshold", 40.0)
    return row.get("Total_Score", 0.0) >= threshold


# Atom registry: name -> function
ATOMS = {
    "sentiment_positive":   atom_sentiment_positive,
    "sentiment_negative":   atom_sentiment_negative,
    "sentiment_neutral":    atom_sentiment_neutral,
    "risk_low":             atom_risk_low,
    "risk_high":            atom_risk_high,
    "growth_strong":        atom_growth_strong,
    "dividend_high":        atom_dividend_high,
    "momentum_strong":      atom_momentum_strong,
    "sector_preferred":     atom_sector_preferred,
    "sector_excluded":      atom_sector_excluded,
    "score_above_threshold":atom_score_above_threshold,
}


# ─────────────────────────────────────────────────────────────────────────────
#  CLAUSE REPRESENTATION
#  A clause is a list of signed atoms: +atom (positive literal) / -atom (negated)
#  Clause evaluates to True if ANY literal is True.
# ─────────────────────────────────────────────────────────────────────────────

class Literal:
    def __init__(self, atom_name: str, negated: bool = False):
        assert atom_name in ATOMS, f"Unknown atom: {atom_name}"
        self.atom_name = atom_name
        self.negated   = negated

    def evaluate(self, row, params) -> bool:
        result = ATOMS[self.atom_name](row, params)
        return (not result) if self.negated else result

    def __repr__(self):
        prefix = "NOT " if self.negated else ""
        return f"{prefix}{self.atom_name}"


def NOT(atom_name): return Literal(atom_name, negated=True)
def POS(atom_name): return Literal(atom_name, negated=False)


class Clause:
    """A disjunction (OR) of literals.  True if at least one literal is True."""
    def __init__(self, *literals, description: str = ""):
        self.literals    = list(literals)
        self.description = description

    def evaluate(self, row, params) -> bool:
        return any(lit.evaluate(row, params) for lit in self.literals)

    def __repr__(self):
        body = " OR ".join(str(l) for l in self.literals)
        return f"({body})"


# ─────────────────────────────────────────────────────────────────────────────
#  CNF FORMULA
#  Built from the following investment rules (converted to CNF clauses)
#
#  Rule 1: IF Sentiment=Positive AND Risk=Low  THEN Investable
#           → clause: (NOT sentiment_negative OR NOT risk_high)
#             [block stocks that are simultaneously negative + high-risk]
#
#  Rule 2: IF Sentiment=Negative  THEN NOT Investable
#           → clause: (NOT sentiment_negative)
#             [directly exclude negative-sentiment stocks]
#
#  Rule 3: IF Risk=High AND Growth NOT Strong  THEN NOT Investable
#           → clause: (NOT risk_high OR growth_strong)
#
#  Rule 4: IF Sector=Excluded  THEN NOT Investable
#           → clause: (NOT sector_excluded)
#
#  Rule 5: IF Risk=Low  THEN Growth OR Dividend must be good
#           → clause: (NOT risk_low OR growth_strong OR dividend_high)
#             [low-risk stocks must still have return potential]
#
#  Rule 6: IF Momentum=Strong AND Sentiment=Positive  THEN preferred (relax threshold)
#           Contrapositive: block weak momentum + negative sentiment together
#           → clause: (momentum_strong OR NOT sentiment_negative)
#
#  Risk-appetite override clauses (dynamically added):
#  Rule 7 (Low appetite):   block all risk_high stocks
#           → clause: (NOT risk_high)
#  Rule 7 (High appetite):  allow risk_high; no extra restriction
# ─────────────────────────────────────────────────────────────────────────────

BASE_CLAUSES = [
    # Rule 1 — block (negative sentiment AND high risk) combination
    Clause(
        NOT("sentiment_negative"), NOT("risk_high"),
        description="Rule 1: NOT(Sentiment=Negative AND Risk=High)"
    ),
    # Rule 2 — exclude negative-sentiment stocks outright
    Clause(
        NOT("sentiment_negative"),
        description="Rule 2: Negative sentiment blocks investment"
    ),
    # Rule 3 — high-risk stocks must have strong growth to compensate
    Clause(
        NOT("risk_high"), POS("growth_strong"),
        description="Rule 3: High-risk requires strong growth"
    ),
    # Rule 4 — honour user-excluded sectors
    Clause(
        NOT("sector_excluded"),
        description="Rule 4: Excluded sector blocks investment"
    ),
    # Rule 5 — low-risk stocks still need return potential
    Clause(
        NOT("risk_low"), POS("growth_strong"), POS("dividend_high"),
        description="Rule 5: Low-risk must offer growth or dividend"
    ),
    # Rule 6 — weak momentum blocks if also negative sentiment
    Clause(
        POS("momentum_strong"), NOT("sentiment_negative"),
        description="Rule 6: Weak momentum + negative sentiment blocked"
    ),
]


def build_cnf_formula(risk_appetite: str) -> list[Clause]:
    """Return the full list of CNF clauses for the given risk appetite."""
    clauses = list(BASE_CLAUSES)

    if risk_appetite == "Low":
        # Extra hard constraint: no high-volatility stocks at all
        clauses.append(
            Clause(
                NOT("risk_high"),
                description="Risk=Low override: ALL high-risk stocks excluded"
            )
        )
    elif risk_appetite == "Medium":
        # Score threshold filter for medium risk
        clauses.append(
            Clause(
                POS("score_above_threshold"),
                description="Risk=Medium: Total_Score must be >= threshold"
            )
        )
    # High appetite: no extra clause — optimiser takes over

    return clauses


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC FILTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def cnf_filter(
    df: pd.DataFrame,
    risk_appetite: str,
    preferred_sectors: list[str] | None = None,
    excluded_sectors:  list[str] | None = None,
    score_threshold:   float = 40.0,
    verbose:           bool  = True,
) -> pd.DataFrame:
    """
    Apply the CNF investment-rule formula to df.

    Parameters
    ----------
    df                : scored stocks DataFrame (must have Total_Score,
                        Sentiment_Score, Volatility, Growth_5yr, etc.)
    risk_appetite     : 'Low' | 'Medium' | 'High'
    preferred_sectors : sectors user prefers (from UI)
    excluded_sectors  : sectors user excluded (from UI)
    score_threshold   : minimum Total_Score for Medium-risk mode
    verbose           : print which stocks are filtered and why

    Returns
    -------
    DataFrame of stocks that pass ALL CNF clauses.
    """
    preferred_sectors = preferred_sectors or []
    excluded_sectors  = excluded_sectors  or []

    params = {
        "preferred_sectors": preferred_sectors,
        "excluded_sectors":  excluded_sectors,
        "score_threshold":   score_threshold,
    }

    clauses = build_cnf_formula(risk_appetite)

    if verbose:
        print(f"\n{'='*60}")
        print(f"CNF PRE-FILTER  |  Risk Appetite: {risk_appetite}")
        print(f"{'='*60}")
        print(f"Active clauses  : {len(clauses)}")
        for c in clauses:
            print(f"  {c}  [{c.description}]")
        print(f"{'-'*60}")

    passed_rows = []
    for _, row in df.iterrows():
        row_pass = True
        fail_reason = None

        for clause in clauses:
            if not clause.evaluate(row, params):
                row_pass   = False
                fail_reason = clause.description
                break

        if row_pass:
            passed_rows.append(row)
        elif verbose:
            print(f"  FILTERED OUT: {row['Symbol']:8s} | {fail_reason}")

    result = pd.DataFrame(passed_rows).reset_index(drop=True)

    if verbose:
        print(f"{'-'*60}")
        print(f"Stocks before CNF filter : {len(df)}")
        print(f"Stocks after  CNF filter : {len(result)}")
        print(f"{'='*60}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Minimal mock data for quick demo
    mock_data = [
        {"Symbol": "SYS",   "Sector": "Technology", "Volatility": 0.28, "Growth_5yr": 0.40,
         "Dividend_Yield": 2.2, "Momentum_Score": 0.95, "Total_Score": 72.0,
         "Sentiment_Score": 0.30, "Sentiment_Label": "Positive"},
        {"Symbol": "KAPCO", "Sector": "Power",      "Volatility": 0.12, "Growth_5yr": 0.08,
         "Dividend_Yield": 14.2,"Momentum_Score": 0.58, "Total_Score": 45.0,
         "Sentiment_Score": -0.20,"Sentiment_Label":"Negative"},
        {"Symbol": "MCB",   "Sector": "Banking",    "Volatility": 0.12, "Growth_5yr": 0.18,
         "Dividend_Yield": 8.2, "Momentum_Score": 0.85, "Total_Score": 62.0,
         "Sentiment_Score": 0.15, "Sentiment_Label": "Positive"},
        {"Symbol": "TRG",   "Sector": "Technology", "Volatility": 0.30, "Growth_5yr": 0.35,
         "Dividend_Yield": 1.5, "Momentum_Score": 0.92, "Total_Score": 58.0,
         "Sentiment_Score": -0.10,"Sentiment_Label":"Negative"},
    ]

    df_mock = pd.DataFrame(mock_data)
    df_filtered = cnf_filter(
        df_mock,
        risk_appetite="Low",
        preferred_sectors=["Banking", "Technology"],
        excluded_sectors=[],
    )
    print("Passed stocks:", df_filtered["Symbol"].tolist())
