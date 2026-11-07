"""
Tracks Kalshi fair-value predictions against actual outcomes, so accuracy
and calibration can be measured empirically instead of judged from one
result. One outcome at 82% isn't evidence the model is wrong -- an event
with an implied 18% chance is *expected* to happen roughly 1 in 6 times.
What actually answers "is this model any good" is: across many resolved
contracts, do "82% predictions" resolve YES about 82% of the time?

Two pieces:
  1. log_prediction() -- called every time a fair-value snapshot is
     computed (each dashboard refresh, while a Kalshi market is being
     auto-tracked), appending a row to KALSHI_PREDICTIONS_LOG_PATH. Off
     by default (config.LOG_KALSHI_PREDICTIONS) since it's a
     continuously-growing file.
  2. resolve_and_report() -- for logged predictions whose contract has
     since closed, queries Kalshi's API for the real settlement result
     (cached locally so already-resolved contracts aren't re-queried),
     and computes a calibration table: for each probability bucket
     (0-20%, 20-40%, ..., 80-100% by default), what fraction of
     predictions in that bucket actually resolved YES? Also reports the
     Brier score (mean squared error between predicted probability and
     the 0/1 outcome -- a standard proper scoring rule for probabilistic
     forecasts, lower is better) for both the model's own predictions AND
     Kalshi's own quoted price at the same moments, so there's a genuine
     baseline to compare against -- prediction markets are usually
     reasonably well-calibrated, so "does the model beat the market's own
     price" is a meaningful bar, not a strawman.

Each contract can have many logged rows (one per refresh while it was
open); the calibration report uses only the LAST (closest-to-close) row
per ticker, since that's the most-informed snapshot and using every row
would pseudo-replicate a single contract's outcome many times over.

Run standalone: python kalshi_tracking.py
"""
import csv
import json
import os
from datetime import datetime, timezone

import pandas as pd

import kalshi_api
from csv_utils import needs_header


PREDICTION_LOG_HEADER = [
    "timestamp", "ticker", "strike", "current_price", "minutes_remaining",
    "p_yes", "annual_vol", "kalshi_yes_bid", "kalshi_yes_ask", "kalshi_last_price",
]


def log_prediction(cfg, ticker, strike, current_price, minutes_remaining, p_yes, annual_vol,
                    kalshi_yes_bid=None, kalshi_yes_ask=None, kalshi_last_price=None):
    """
    Appends one fair-value snapshot to the predictions log. Only active
    when cfg.LOG_KALSHI_PREDICTIONS is True. `ticker` should be the real
    Kalshi ticker when known (via kalshi_api auto-tracking) -- rows with
    no ticker can't later be matched to a resolved outcome.
    """
    if not cfg.LOG_KALSHI_PREDICTIONS:
        return
    # Rotates an older-layout file aside rather than appending wider rows to it.
    write_header = needs_header(cfg.KALSHI_PREDICTIONS_LOG_PATH, PREDICTION_LOG_HEADER)
    with open(cfg.KALSHI_PREDICTIONS_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(PREDICTION_LOG_HEADER)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            ticker or "",
            strike,
            current_price,
            minutes_remaining,
            p_yes,
            annual_vol,
            kalshi_yes_bid if kalshi_yes_bid is not None else "",
            kalshi_yes_ask if kalshi_yes_ask is not None else "",
            kalshi_last_price if kalshi_last_price is not None else "",
        ])


def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_outcomes_cache(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_outcomes_cache(path: str, cache: dict):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def resolve_outcomes(tickers: list, cache: dict) -> dict:
    """
    For each ticker not already in `cache`, queries Kalshi for its
    current status. Contracts that have settled ("finalized", with a
    "yes"/"no" result) get added to the cache; still-open or otherwise
    unresolved ones are left out (so they get checked again next run).
    Returns the updated cache -- does not save it to disk, call
    save_outcomes_cache() separately once you have it.
    """
    updated = dict(cache)
    for ticker in tickers:
        if not ticker or ticker in updated:
            continue
        try:
            market = kalshi_api.get_market(ticker)
        except Exception:
            continue  # network hiccup on this one -- try again next run
        result = market.get("result")
        if result in ("yes", "no"):
            updated[ticker] = result
    return updated


def _bucket_label(lower: float, upper: float) -> str:
    return f"{lower * 100:.0f}-{upper * 100:.0f}%"


def compute_calibration(df: pd.DataFrame, outcomes: dict, bucket_size: float, min_sample: int) -> dict:
    """
    Returns {"model": DataFrame, "kalshi_market": DataFrame or None,
    "model_brier": float, "kalshi_brier": float or None, "n_resolved": int}.

    Deduplicates to one row per ticker first (the row with the smallest
    minutes_remaining -- the closest-to-close, most-informed snapshot),
    then keeps only tickers with a known outcome.
    """
    if df.empty or not outcomes:
        return {"model": pd.DataFrame(), "kalshi_market": None,
                "model_brier": None, "kalshi_brier": None, "n_resolved": 0}

    df = df[df["ticker"].astype(str).str.len() > 0].copy()
    # drop_duplicates keeps one whole row per ticker. (groupby().first() would
    # take the first non-null value per *column*, so a closest-to-close row with
    # no Kalshi quote would silently borrow an older row's quote.)
    df = df.sort_values("minutes_remaining", kind="stable").drop_duplicates("ticker", keep="first")
    df["outcome"] = df["ticker"].map(outcomes)
    df = df.dropna(subset=["outcome"])

    if df.empty:
        return {"model": pd.DataFrame(), "kalshi_market": None,
                "model_brier": None, "kalshi_brier": None, "n_resolved": 0}

    df["actual"] = (df["outcome"] == "yes").astype(int)

    n_buckets = int(round(1.0 / bucket_size))
    # Rounded: 3 * 0.2 is 0.6000000000000001, which would push a prediction of
    # exactly 0.6 down into the 40-60% bucket.
    bounds = [round(i * bucket_size, 10) for i in range(n_buckets + 1)]

    def _bucketize(series):
        rows = []
        for i in range(n_buckets):
            lo, hi = bounds[i], bounds[i + 1]
            is_last = i == n_buckets - 1
            mask = (series >= lo) & ((series <= hi) if is_last else (series < hi))
            sub = df.loc[series[mask].index]   # `series` may be a subset of df (rows with quotes)
            n = len(sub)
            observed = sub["actual"].mean() if n > 0 else None
            rows.append({
                "bucket": _bucket_label(lo, hi),
                "n": n,
                "observed_yes_rate": observed,
                "low_sample": n < min_sample,
            })
        return pd.DataFrame(rows)

    model_table = _bucketize(df["p_yes"])
    model_brier = ((df["p_yes"] - df["actual"]) ** 2).mean()

    kalshi_table = None
    kalshi_brier = None
    has_kalshi_quotes = df["kalshi_yes_bid"].notna().any() and df["kalshi_yes_ask"].notna().any()
    if has_kalshi_quotes:
        df["kalshi_mid"] = (df["kalshi_yes_bid"] + df["kalshi_yes_ask"]) / 2
        with_quote = df.dropna(subset=["kalshi_mid"])
        if not with_quote.empty:
            kalshi_table = _bucketize(with_quote["kalshi_mid"])
            kalshi_brier = ((with_quote["kalshi_mid"] - with_quote["actual"]) ** 2).mean()

    return {
        "model": model_table,
        "kalshi_market": kalshi_table,
        "model_brier": model_brier,
        "kalshi_brier": kalshi_brier,
        "n_resolved": len(df),
    }


def _print_table(table: pd.DataFrame):
    if table.empty:
        print("  (no resolved predictions yet)")
        return
    for _, row in table.iterrows():
        if row["n"] == 0:
            print(f"  {row['bucket']:>10}   n=0")
            continue
        flag = "  (low sample)" if row["low_sample"] else ""
        print(f"  {row['bucket']:>10}   observed YES rate: {row['observed_yes_rate'] * 100:5.1f}%   (n={row['n']}){flag}")


def resolve_and_report(cfg):
    if not os.path.isfile(cfg.KALSHI_PREDICTIONS_LOG_PATH):
        print(f"No predictions log found at {cfg.KALSHI_PREDICTIONS_LOG_PATH}.")
        print("Set LOG_KALSHI_PREDICTIONS = True in config.py and use the dashboard's Kalshi")
        print("section for a while to start building a track record, then run this again.")
        return

    df = load_predictions(cfg.KALSHI_PREDICTIONS_LOG_PATH)
    # dropna first: a blank ticker cell reads back as NaN, and str(NaN) is "nan".
    tickers = [t for t in df["ticker"].dropna().astype(str).unique() if t]
    print(f"Loaded {len(df)} logged predictions across {len(tickers)} unique contracts.")

    cache = load_outcomes_cache(cfg.KALSHI_OUTCOMES_CACHE_PATH)
    print(f"Checking Kalshi for outcomes on {sum(1 for t in tickers if t not in cache)} unresolved contract(s)...")
    cache = resolve_outcomes(tickers, cache)
    save_outcomes_cache(cfg.KALSHI_OUTCOMES_CACHE_PATH, cache)

    result = compute_calibration(df, cache, cfg.KALSHI_CALIBRATION_BUCKET_SIZE, cfg.KALSHI_CALIBRATION_MIN_SAMPLE)

    print(f"\n{result['n_resolved']} contracts resolved so far (out of {len(tickers)} logged).\n")

    print("Model calibration (does an 'X%' prediction actually resolve YES ~X% of the time?):")
    _print_table(result["model"])
    if result["model_brier"] is not None:
        print(f"  Model Brier score: {result['model_brier']:.4f}  (lower is better; 0 = perfect, 0.25 = always guessing 50%)")

    if result["kalshi_market"] is not None:
        print("\nKalshi's own market price calibration, for comparison:")
        _print_table(result["kalshi_market"])
        print(f"  Kalshi market Brier score: {result['kalshi_brier']:.4f}")
        if result["model_brier"] is not None:
            if result["model_brier"] < result["kalshi_brier"]:
                print("\n  Model's Brier score is LOWER than Kalshi's own market -- on this data, the model")
                print("  has been a better-calibrated forecaster than the market price itself.")
            else:
                print("\n  Model's Brier score is HIGHER than (or equal to) Kalshi's own market -- on this")
                print("  data, just trusting the market's own price would have done as well or better.")

    print(
        "\nTreat any of this with real caution until n_resolved is well into the dozens per "
        "bucket -- a handful of contracts is not enough to distinguish a real calibration issue "
        "from ordinary variance."
    )


if __name__ == "__main__":
    import config as cfg
    resolve_and_report(cfg)
