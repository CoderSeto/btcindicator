"""
Backtests the bot's price-based signal patterns against real historical
BTC data to answer: "when this same signal combination has shown up
before, how often did price actually turn within the next few candles?"

Important limitation: no exchange publishes historical order-book
snapshots, so this backtest only covers the 6 candle-based indicators
(SMA/EMA cross, RSI, MACD, Bollinger, volume spike) -- not the two
order-book signals (`orderbook` imbalance and `book_flow`), which only
exist live and have no path to historical backtesting at all. Live depth
could be recorded going forward to build such a history, but nothing
downloadable exists to backtest against today. The aggregated exchange
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


def effective_sample_size(positions, hits, max_lag: int) -> float:
    """
    How many independent observations a bucket's hits are worth.

    Every candle is scored against the move over the next `lookahead`
    candles, so two positions less than `lookahead` apart share part of
    their outcome window and their hits are correlated -- and a state vote
    like sma_cross fires on long runs of consecutive candles. Counting each
    row as one observation overstated the evidence ~3.5x for those signals
    (REVIEW_FINDINGS D2).

    This is the variance of the mean hit with covariances included up to lag
    `max_lag - 1` (a truncated, unweighted HAC estimate), divided by the
    variance the naive count implies:

        design_effect = (sum x_i^2 + 2 * sum_{0 < pos_j - pos_i < max_lag} x_i x_j) / sum x_i^2
        n_eff = n / design_effect,   x = hit - mean(hit)

    Truncating at the overlap is exact under the no-edge null: positions at
    least `max_lag` apart have disjoint future windows, so under a random
    walk their hits are uncorrelated whatever the signal did. The design
    effect is floored at 1 (n_eff never exceeds n); a bucket with no variance
    (all hits or all misses) has nothing to correct and reports n.

    `positions` must be ascending candle indices, `hits` 0/1 per position.
    """
    n = len(hits)
    if n == 0:
        return 0.0
    p = sum(hits) / n
    x = [h - p for h in hits]
    base = sum(v * v for v in x)
    if base <= 0:
        return float(n)
    total = base
    for a in range(n):
        b = a + 1
        while b < n and positions[b] - positions[a] < max_lag:
            total += 2.0 * x[a] * x[b]
            b += 1
    design_effect = max(total / base, 1.0)
    return n / design_effect


def _summarize(observations: dict, base_rates: dict, direction_of, lookahead: int) -> dict:
    """
    observations: {bucket_key: [(position, hit), ...]} in position order.
    Returns {key: {"hits", "total", "n_eff", "base_rate"}}; base_rate is the
    share of ALL scored windows in this backtest that moved in the bucket's
    direction -- what an indicator with no skill would score.
    """
    out = {}
    for key, obs in observations.items():
        positions = [pos for pos, _ in obs]
        hits = [h for _, h in obs]
        out[key] = {
            "hits": sum(hits),
            "total": len(hits),
            "n_eff": round(effective_sample_size(positions, hits, lookahead), 1),
            "base_rate": base_rates.get(direction_of(key)),
        }
    return out


def _direction_of(key: str) -> str:
    """'rsi_bullish' -> 'bullish', 'bearish_3' -> 'bearish', 'bullish' -> 'bullish'."""
    return "bullish" if "bullish" in key else "bearish"


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
    # History ends at "now": its last candle may still be forming, and a partial
    # close is no outcome to score against.
    df, _ = data_fetch.drop_forming_candle(df, cfg.GRANULARITY)

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

    # Each bucket keeps its (position, hit) observations, so the effective
    # sample size can account for overlapping outcome windows afterwards.
    strength_obs = {}    # "direction_strength" -> [(i, hit), ...]
    divergence_obs = {}  # "bullish"/"bearish" -> [(i, hit), ...]
    indicator_obs = {}   # "indicatorname_direction" -> [(i, hit), ...]
    ups = scored = 0     # base rate: how often ANY scored window moved up

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

        if actual is not None:
            scored += 1
            ups += actual == "bullish"

        if actual is not None and snap.composite != "neutral":
            key = f"{snap.composite}_{snap.strength}"
            strength_obs.setdefault(key, []).append((i, int(actual == snap.composite)))

        if actual is not None:
            for name, direction in snap.individual.items():
                if direction == "neutral":
                    continue
                indicator_obs.setdefault(f"{name}_{direction}", []).append((i, int(actual == direction)))

        div_window = df.iloc[max(0, i - cfg.DIVERGENCE_LOOKBACK + 1): i + 1]
        div = divergence.detect_divergence(div_window, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)
        if actual is not None and div["type"]:
            divergence_obs.setdefault(div["type"], []).append((i, int(actual == div["type"])))

    base_rates = {
        "bullish": ups / scored if scored else None,
        "bearish": (scored - ups) / scored if scored else None,
        "total": scored,
    }

    def summarize(obs):
        return _summarize(obs, base_rates, _direction_of, lookahead)

    results = {
        "product_id": cfg.PRODUCT_ID,
        "granularity": cfg.GRANULARITY,
        "lookahead_candles": lookahead,
        "candles_used": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_rates": base_rates,
        "strength_buckets": summarize(strength_obs),
        "divergence_buckets": summarize(divergence_obs),
        "indicator_buckets": summarize(indicator_obs),
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


def _rate_entry(bucket: dict, min_sample: int) -> dict:
    n_eff = bucket.get("n_eff", bucket["total"])
    return {
        "rate": bucket["hits"] / bucket["total"],
        "n": bucket["total"],
        "n_eff": n_eff,
        "base_rate": bucket.get("base_rate"),
        "low_sample": n_eff < min_sample,
    }


def get_confidence(results: dict, price_snapshot, divergence_result: dict, min_sample: int) -> dict:
    """
    Looks up the historical hit rate for the current price-based signal
    state and divergence state. Returns {"available": False} if there's no
    matching backtest yet.

    Each entry carries the raw row count `n`, the overlap-corrected `n_eff`
    (equal to n for results saved before it existed) and the `base_rate` for
    that direction (None for older results); `low_sample` is judged on n_eff.
    """
    if not results or "error" in results:
        return {"available": False}

    out = {"available": True, "lookahead_candles": results.get("lookahead_candles")}

    if price_snapshot.composite != "neutral":
        key = f"{price_snapshot.composite}_{price_snapshot.strength}"
        bucket = results.get("strength_buckets", {}).get(key)
        if bucket and bucket["total"] > 0:
            out["signal"] = {"direction": price_snapshot.composite, **_rate_entry(bucket, min_sample)}

    div_type = divergence_result.get("type") if divergence_result else None
    if div_type:
        bucket = results.get("divergence_buckets", {}).get(div_type)
        if bucket and bucket["total"] > 0:
            out["divergence"] = {"direction": div_type, **_rate_entry(bucket, min_sample)}

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

    base = results.get("base_rates", {})
    if base.get("total"):
        print(f"Base rate: {base['bullish'] * 100:.1f}% of the {base['total']} scored "
              f"{results['lookahead_candles']}-candle windows moved up, "
              f"{base['bearish'] * 100:.1f}% down. A call with no skill scores its direction's base rate.\n")

    def line(key, b, width):
        rate = (b["hits"] / b["total"] * 100) if b["total"] else 0.0
        n_eff = b.get("n_eff", b["total"])
        flag = "  (low sample)" if n_eff < cfg.BACKTEST_MIN_SAMPLE else ""
        vs = f"  vs base {b['base_rate'] * 100:4.1f}%" if b.get("base_rate") is not None else ""
        print(f"  {key:{width}s}  {rate:5.1f}%{vs}  (n={b['total']}, ~{n_eff:.0f} independent){flag}")

    print("Signal-strength hit rates (price-based indicators only, no order book):")
    for key in sorted(results["strength_buckets"]):
        line(key, results["strength_buckets"][key], 14)

    print("\nDivergence hit rates:")
    if not results["divergence_buckets"]:
        print("  none detected in this window")
    for key in sorted(results["divergence_buckets"]):
        line(key, results["divergence_buckets"][key], 14)

    print("\nPer-indicator hit rates (used to compute weighted conviction -- see weighting.py):")
    for key in sorted(results.get("indicator_buckets", {})):
        line(key, results["indicator_buckets"][key], 18)
    print("\n'independent' is the effective sample size: rows whose outcome windows overlap "
          "are correlated, so they're worth less than one observation each.")


if __name__ == "__main__":
    print(
        f"Backtesting {cfg.PRODUCT_ID} @ {cfg.GRANULARITY}s granularity, "
        f"target {cfg.BACKTEST_HISTORY_CANDLES} candles (this pulls a lot of history, "
        f"give it a bit)...\n"
    )
    results = run_backtest(cfg)
    _print_summary(results)
