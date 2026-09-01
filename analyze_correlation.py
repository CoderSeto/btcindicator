"""
Analyzes how much the aggregated exchange delta signal (agg_delta) overlaps
with the other 7 indicators, using the signal log accumulated by main.py or
app.py (only recorded when config.LOG_ALL_SIGNALS = True).

This answers the "how much is agg_delta double-counting another indicator"
question empirically, from real logged data, rather than by reasoning
about what each indicator measures and over what timescale.

Two numbers per indicator, both restricted to cycles where agg_delta was
actually present (i.e. the multi-exchange feeds had warmed up):

  - Agreement rate: of the cycles where BOTH agg_delta and this indicator
    had a directional (non-neutral) vote, how often did they land on the
    same side?
  - Correlation: Pearson correlation between the two indicators' votes,
    encoded bullish=+1 / neutral=0 / bearish=-1, across every logged cycle
    where agg_delta was present (whether or not each individual indicator
    was neutral that cycle).

Neither number claims one indicator "causes" the other -- both are
ultimately downstream of Bitcoin's actual trading activity to varying
degrees. High overlap means the two indicators often carry similar
information; it doesn't by itself mean one is safe to drop -- see the
notes printed at the end of the report.

Setup:
  1. Set LOG_ALL_SIGNALS = True in config.py
  2. Run main.py (or the dashboard) for a while with the multi-exchange
     feeds connected, so enough cycles accumulate with agg_delta present
  3. Run this script:  python analyze_correlation.py
"""
import sys

import pandas as pd

import config as cfg

_VOTE_MAP = {"bullish": 1, "neutral": 0, "bearish": -1}

_OTHER_INDICATORS = ["sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "orderbook"]


def load_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def analyze(df: pd.DataFrame, min_sample: int) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per indicator (agreement rate,
    correlation, and sample sizes for each), restricted to rows where
    agg_delta was actually recorded. Returns an empty DataFrame if no such
    rows exist yet.
    """
    if "agg_delta" not in df.columns:
        return pd.DataFrame()

    has_agg = df["agg_delta"].notna() & (df["agg_delta"] != "")
    df = df[has_agg].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for name in _OTHER_INDICATORS:
        if name not in df.columns:
            continue

        sub = df[[name, "agg_delta"]].copy()
        sub[name] = sub[name].replace("", pd.NA)
        sub = sub.dropna(subset=[name])

        # Agreement rate: both directional (non-neutral) -- how often do they match?
        directional = sub[(sub[name] != "neutral") & (sub["agg_delta"] != "neutral")]
        agree_rate = (directional[name] == directional["agg_delta"]).mean() if len(directional) > 0 else None

        # Correlation across all rows with agg_delta present, neutral encoded as 0
        name_num = sub[name].map(_VOTE_MAP)
        agg_num = sub["agg_delta"].map(_VOTE_MAP)
        valid = name_num.notna() & agg_num.notna()
        n_valid = int(valid.sum())
        if n_valid >= 2 and name_num[valid].std() > 0 and agg_num[valid].std() > 0:
            corr = name_num[valid].corr(agg_num[valid])
        else:
            corr = None

        rows.append({
            "indicator": name,
            "agreement_rate": agree_rate,
            "agreement_n": len(directional),
            "correlation": corr,
            "correlation_n": n_valid,
            "low_sample": len(directional) < min_sample,
        })

    return pd.DataFrame(rows)


def _print_report(result: pd.DataFrame, log_path: str):
    if result.empty:
        print(f"No rows with agg_delta recorded yet in {log_path}.")
        print("Make sure LOG_ALL_SIGNALS = True in config.py, and let the bot run for a")
        print("while with ENABLE_EXCHANGE_FEEDS = True so the multi-exchange feeds have")
        print("time to warm up and start contributing agg_delta readings.")
        return

    print(f"{'Indicator':<12} {'Agreement':>10} {'n':>6}    {'Correlation':>12} {'n':>6}")
    print("-" * 56)
    for _, row in result.sort_values("correlation", ascending=False, na_position="last").iterrows():
        agree_str = f"{row['agreement_rate'] * 100:.0f}%" if pd.notna(row["agreement_rate"]) else "n/a"
        corr_str = f"{row['correlation']:+.2f}" if pd.notna(row["correlation"]) else "n/a"
        flag = "  (low sample)" if row["low_sample"] else ""
        print(f"{row['indicator']:<12} {agree_str:>10} {row['agreement_n']:>6}    {corr_str:>12} {row['correlation_n']:>6}{flag}")

    print(
        "\nAgreement rate: of the cycles where both agg_delta and this indicator had a "
        "\ndirectional (non-neutral) vote, how often they matched."
        "\nCorrelation: Pearson correlation across every logged cycle where agg_delta was "
        "\npresent, bullish=+1 / neutral=0 / bearish=-1."
        "\n\nHigh overlap means the two indicators often carry similar information right "
        "\nnow -- it doesn't automatically mean one is safe to remove. A short collection "
        "\nwindow, one market regime, or coincidental timing can all inflate agreement "
        "\nthat wouldn't hold up over a longer period. Treat this as a starting point for "
        "\nfurther judgment, not a final verdict -- and the more cycles logged, the more "
        "\nthis is worth trusting."
    )


if __name__ == "__main__":
    try:
        log_df = load_log(cfg.SIGNAL_LOG_PATH)
    except FileNotFoundError:
        print(f"No signal log found at {cfg.SIGNAL_LOG_PATH}.")
        print("Set LOG_ALL_SIGNALS = True in config.py, then run main.py or the dashboard")
        print("for a while to accumulate data, then run this again.")
        sys.exit(1)

    result = analyze(log_df, cfg.BACKTEST_MIN_SAMPLE)
    _print_report(result, cfg.SIGNAL_LOG_PATH)
