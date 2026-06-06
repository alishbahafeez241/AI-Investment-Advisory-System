"""
nlp_sentiment.py
----------------
Performs TextBlob-based sentiment analysis on financial news headlines
and maps a Sentiment_Score to each stock in stocks_data.csv.

Key output columns added:
  Headline         - most recent matched headline text
  Sentiment_Score  - float in [-1.0, +1.0]  (TextBlob polarity)
  Sentiment_Label  - 'Positive' | 'Neutral' | 'Negative'

Usage:
  from nlp_sentiment import enrich_with_sentiment
  df_enriched = enrich_with_sentiment(df_stocks)

  OR run standalone:
  python nlp_sentiment.py
"""

import os
import pandas as pd
from textblob import TextBlob

from scraper import get_headlines


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SENTIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_headline(text: str) -> dict:
    """
    Run TextBlob sentiment on one headline.
    Returns polarity (sentiment score) and subjectivity.
    """
    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity,     4)   # -1.0 to +1.0
    subjectivity = round(blob.sentiment.subjectivity, 4)   #  0.0 to +1.0

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "Headline":        text,
        "Sentiment_Score": polarity,
        "Subjectivity":    subjectivity,
        "Sentiment_Label": label,
    }


def aggregate_sentiment(records: list[dict]) -> dict[str, dict]:
    """
    Given a list of {Symbol, Headline} records, compute per-symbol
    average sentiment across all matching headlines.
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for r in records:
        result = analyse_headline(r["Headline"])
        buckets[r["Symbol"]].append(result)

    aggregated = {}
    for symbol, results in buckets.items():
        scores = [r["Sentiment_Score"] for r in results]
        avg_score = round(sum(scores) / len(scores), 4)

        # Pick the headline with the highest absolute polarity as representative
        rep = max(results, key=lambda x: abs(x["Sentiment_Score"]))

        if avg_score > 0.05:
            label = "Positive"
        elif avg_score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        aggregated[symbol] = {
            "Headline":        rep["Headline"],
            "Sentiment_Score": avg_score,
            "Subjectivity":    round(sum(r["Subjectivity"] for r in results) / len(results), 4),
            "Sentiment_Label": label,
            "Headline_Count":  len(results),
        }

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the stocks DataFrame and returns it with three new columns:
      Headline | Sentiment_Score | Sentiment_Label

    Stocks with no matched headline receive Sentiment_Score = 0.0 (Neutral).
    """
    print("Fetching headlines and running TextBlob sentiment analysis...")
    records     = get_headlines()
    sent_map    = aggregate_sentiment(records)

    df = df.copy()
    df["Headline"]        = df["Symbol"].map(lambda s: sent_map.get(s, {}).get("Headline",        "No recent news"))
    df["Sentiment_Score"] = df["Symbol"].map(lambda s: sent_map.get(s, {}).get("Sentiment_Score", 0.0))
    df["Sentiment_Label"] = df["Symbol"].map(lambda s: sent_map.get(s, {}).get("Sentiment_Label", "Neutral"))

    print(f"Sentiment enrichment complete. {len(sent_map)} stocks matched with headlines.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE: update stocks_data.csv
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_IN  = "stocks_data.csv"
    CSV_OUT = "stocks_data.csv"   # overwrite in-place

    if not os.path.exists(CSV_IN):
        print(f"ERROR: {CSV_IN} not found. Run from project root directory.")
        raise SystemExit(1)

    df = pd.read_csv(CSV_IN)
    df_enriched = enrich_with_sentiment(df)
    df_enriched.to_csv(CSV_OUT, index=False)

    print(f"\nUpdated {CSV_OUT}")
    print(df_enriched[["Symbol", "Sentiment_Score", "Sentiment_Label", "Headline"]].to_string(index=False))
