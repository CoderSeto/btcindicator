"""
Backtests the bot's price-based signal patterns against real historical
BTC data to answer: "when this same signal combination has shown up
before, how often did price actually turn within the next few candles?"

Important limitation: Coinbase's public API doesn't expose historical
order-book snapshots, so this backtest only covers the 6 candle-based
indicators (SMA/EMA cross, RSI, MACD, Bollinger, volume spike) -- not the
order-book imbalance signal, which only exists live. Order-book imbalance
has no path to historical backtesting at all. The aggregated exchange
delta signal (agg_delta) is in the same boat here, but CAN be backtested
separately -- see backtest_agg_delta.py, which reconstructs historical
agg_delta readings from Binance.US's trade history and merges the result
into this same indicator_buckets structure.

This produces empirical hit rates grounded in real historical frequency,
along with the sample size (n) for each one -- not a formally rigorous
statistical confidence interval, and not a guarantee about future price
behavior. Treat low-n results with real skepticism.

Besides the composite (direction, strength) buckets, this also tracks a
hit rate for each individual indicator's bullish/bearish calls on their
own -- see weighting.py, which turns those into weights for a separate
"weighted conviction" score. Note that those weights come from this same
historical window, so they're a real historical edge, not an independent
out-of-sample validation that weighting actually beats the raw count.

Run standalone with:  python backtest.py
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config as cfg
import data_fetch
import divergence
import indicators
import signals


def fetch_history(product_id: str, granularity: int, target_candles: int, pause: float) -> pd.DataFrame:
    """
    Paginates backwards from now using start/end windows (Coinbase caps
    each request at 300 candles) until roughly `target_candles` candles
    have been collected, or history runs out.
    """
    chunks = []
    end = pd.Timestamp.now(tz="UTC")
    collected = 0

    while collected < target_candles:
        start = end - pd.Timedelta(seconds=granularity * 300)
        try:
            chunk = data_fetch.get_candles_range(product_id, granularity, start, end)
        except Exception as e:
            print(f"[backtest] stopping pagination early after a request error: {e}")
            break

        if chunk.empty:
            break

        chunks.append(chunk)
        collected += len(chunk)
        end = chunk["time"].min()
        time.sleep(pause)

        if len(chunk) < 2:  # ran out of history
            break

    if not chunks:
        return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    if len(df) > target_candles:
        df = df.tail(target_candles).reset_index(drop=True)
    return df


def _warmup_candles(cfg) -> int:
    return max(
        cfg.SMA_SLOW, cfg.BOLLINGER_PERIOD, cfg.MACD_SLOW + cfg.MACD_SIGNAL,
        cfg.RSI_PERIOD, cfg.VOLUME_SPIKE_WINDOW, cfg.DIVERGENCE_LOOKBACK,
    ) + 5


def run_backtest(cfg) -> dict:
    """
    Fetches history, replays the signal + divergence logic at every
    historical point, and checks the actual outcome `BACKTEST_LOOKAHEAD_CANDLES`
    later. Saves results to `BACKTEST_RESULTS_PATH` and also returns them.
    """
    df = fetch_history(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.BACKTEST_HISTORY_CANDLES, cfg.BACKTEST_REQUEST_PAUSE)

    warmup = _warmup_candles(cfg)
    lookahead = cfg.BACKTEST_LOOKAHEAD_CANDLES
    min_needed = warmup + lookahead + 10

    if df.empty or len(df) < min_needed:
        return {
            "error": f"Only got {len(df)} historical candles, need at least {min_needed} to backtest.",
            "product_id": cfg.PRODUCT_ID,
            "granularity": cfg.GRANULARITY,
        }

    df = indicators.compute_all(df, cfg)
    closes = df["close"]

    strength_buckets = {}    # "direction_strength" -> [hits, total]
    divergence_buckets = {}  # "bullish"/"bearish" -> [hits, total]
    indicator_buckets = {}   # "indicatorname_direction" -> [hits, total]

    n = len(df)
    for i in range(warmup, n - lookahead):
        pair = df.iloc[i - 1:i + 1]  # last 2 rows -> crossover-aware signal read at point i
        snap = signals.evaluate_price_only(pair, cfg)

        current_close = closes.iloc[i]
        future_close = closes.iloc[i + lookahead]
        if future_close > current_close:
            actual = "bullish"
        elif future_close < current_close:
            actual = "bearish"
        else:
            actual = None  # exact tie -- skip for direction scoring

        if actual is not None and snap.composite != "neutral":
            key = f"{snap.composite}_{snap.strength}"
            hits, total = strength_buckets.get(key, [0, 0])
            strength_buckets[key] = [hits + (1 if actual == snap.composite else 0), total + 1]

        if actual is not None:
            for name, direction in snap.individual.items():
                if direction == "neutral":
                    continue
                key = f"{name}_{direction}"
                hits, total = indicator_buckets.get(key, [0, 0])
                indicator_buckets[key] = [hits + (1 if actual == direction else 0), total + 1]

        div_window = df.iloc[max(0, i - cfg.DIVERGENCE_LOOKBACK + 1): i + 1]
        div = divergence.detect_divergence(div_window, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)
        if actual is not None and div["type"]:
            dhits, dtotal = divergence_buckets.get(div["type"], [0, 0])
            divergence_buckets[div["type"]] = [dhits + (1 if actual == div["type"] else 0), dtotal + 1]

    results = {
        "product_id": cfg.PRODUCT_ID,
        "granularity": cfg.GRANULARITY,
        "lookahead_candles": lookahead,
        "candles_used": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strength_buckets": {k: {"hits": h, "total": t} for k, (h, t) in strength_buckets.items()},
        "divergence_buckets": {k: {"hits": h, "total": t} for k, (h, t) in divergence_buckets.items()},
        "indicator_buckets": {k: {"hits": h, "total": t} for k, (h, t) in indicator_buckets.items()},
    }

    with open(cfg.BACKTEST_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_results(cfg) -> Optional[dict]:
    """
    Loads cached backtest results, but only if they match the current
    product/granularity -- confidence numbers from a BTC-USD 1-minute
    backtest don't apply to ETH-USD or a different candle size.
    """
    if not os.path.isfile(cfg.BACKTEST_RESULTS_PATH):
        return None
    try:
        with open(cfg.BACKTEST_RESULTS_PATH) as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if results.get("product_id") != cfg.PRODUCT_ID or results.get("granularity") != cfg.GRANULARITY:
        return None
    return results


def get_confidence(results: dict, price_snapshot, divergence_result: dict, min_sample: int) -> dict:
    """
    Looks up the historical hit rate for the current price-based signal
    state and divergence state. Returns {"available": False} if there's no
    matching backtest yet.
    """
    if not results or "error" in results:
        return {"available": False}

    out = {"available": True, "lookahead_candles": results.get("lookahead_candles")}

    if price_snapshot.composite != "neutral":
        key = f"{price_snapshot.composite}_{price_snapshot.strength}"
        bucket = results.get("strength_buckets", {}).get(key)
        if bucket and bucket["total"] > 0:
            out["signal"] = {
                "direction": price_snapshot.composite,
                "rate": bucket["hits"] / bucket["total"],
                "n": bucket["total"],
                "low_sample": bucket["total"] < min_sample,
            }

    div_type = divergence_result.get("type") if divergence_result else None
    if div_type:
        bucket = results.get("divergence_buckets", {}).get(div_type)
        if bucket and bucket["total"] > 0:
            out["divergence"] = {
                "direction": div_type,
                "rate": bucket["hits"] / bucket["total"],
                "n": bucket["total"],
                "low_sample": bucket["total"] < min_sample,
            }

    return out


def _print_summary(results: dict):
    if "error" in results:
        print(f"[error] {results['error']}")
        return

    print(
        f"Done. Used {results['candles_used']} candles of {results['product_id']} "
        f"@ {results['granularity']}s, lookahead={results['lookahead_candles']} candles."
    )
    print(f"Saved to {cfg.BACKTEST_RESULTS_PATH}\n")

    print("Signal-strength hit rates (price-based indicators only, no order book):")
    for key in sorted(results["strength_buckets"]):
        b = results["strength_buckets"][key]
        rate = (b["hits"] / b["total"] * 100) if b["total"] else 0.0
        flag = "  (low sample)" if b["total"] < cfg.BACKTEST_MIN_SAMPLE else ""
        print(f"  {key:14s}  {rate:5.1f}%  (n={b['total']}){flag}")

    print("\nDivergence hit rates:")
    if not results["divergence_buckets"]:
        print("  none detected in this window")
    for key in sorted(results["divergence_buckets"]):
        b = results["divergence_buckets"][key]
        rate = (b["hits"] / b["total"] * 100) if b["total"] else 0.0
        flag = "  (low sample)" if b["total"] < cfg.BACKTEST_MIN_SAMPLE else ""
        print(f"  {key:14s}  {rate:5.1f}%  (n={b['total']}){flag}")

    print("\nPer-indicator hit rates (used to compute weighted conviction -- see weighting.py):")
    for key in sorted(results.get("indicator_buckets", {})):
        b = results["indicator_buckets"][key]
        rate = (b["hits"] / b["total"] * 100) if b["total"] else 0.0
        flag = "  (low sample)" if b["total"] < cfg.BACKTEST_MIN_SAMPLE else ""
        print(f"  {key:18s}  {rate:5.1f}%  (n={b['total']}){flag}")


if __name__ == "__main__":
    print(
        f"Backtesting {cfg.PRODUCT_ID} @ {cfg.GRANULARITY}s granularity, "
        f"target {cfg.BACKTEST_HISTORY_CANDLES} candles (this pulls a lot of history, "
        f"give it a bit)...\n"
    )
    results = run_backtest(cfg)
    _print_summary(results)
