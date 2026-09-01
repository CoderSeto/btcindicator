"""
Reconstructs historical agg_delta readings and backtests them, using
Binance.US's public aggTrades REST history -- the only one of the three
exchanges agg_delta draws from live (Coinbase, Binance.US, Bybit) that has
a practical time-range historical trades endpoint:

  - Coinbase's /trades endpoint is cursor/trade-ID paginated, not
    time-range -- covering multiple days would take an impractical number
    of requests for a liquid pair like BTC-USD.
  - Bybit's recent-trade endpoint caps at the last 60 trades, no
    pagination at all -- not usable for backtesting.
  - Binance.US's aggTrades endpoint supports startTime/endTime range
    queries (up to 1000 trades per request, paginated forward), so it's
    the one exchange this can practically be built from.

**This means the backtested agg_delta reflects Binance.US's own trade
flow alone, not the combined 3-exchange signal the live bot computes.**
It's an approximation of the live signal, not a reconstruction of it --
real historical grounding instead of none, but don't expect the hit rate
here to exactly match how the live 3-exchange agg_delta performs. Binance.US
also has thinner liquidity than Coinbase, so some candles in the backtest
window may have little or no Binance.US trade coverage -- those are
skipped rather than guessed at.

What this does:
  1. Loads the existing candle-based backtest (run `python backtest.py`
     first if you haven't -- this ADDS to that, it doesn't replace it)
  2. Fetches Binance.US aggTrades covering the last AGG_DELTA_BACKTEST_HOURS
  3. Reconstructs what the rolling-window agg_delta reading would have been
     at each candle, using the same AGG_WINDOW_SECONDS/AGG_DELTA_THRESHOLD
     logic as the live signal (same window-boundary semantics as
     exchange_feeds.TradeAggregator, so this is a faithful replay of that
     same math against historical data instead of a live stream)
  4. Checks the same "did price move as predicted within
     BACKTEST_LOOKAHEAD_CANDLES" outcome as every other backtested indicator
  5. Merges the resulting agg_delta_bullish/agg_delta_bearish hit-rate
     buckets into backtest_results.json, so weighting.py picks them up
     automatically from here on

Run standalone with:  python backtest_agg_delta.py
"""
import json
from datetime import datetime, timezone

import pandas as pd

import backtest
import config as cfg
import exchange_feeds


def reconstruct_agg_delta_series(trades: list, candle_times: list, window_seconds: float) -> list:
    """
    trades: list of {"time": float epoch seconds, "side": "buy"/"sell",
    "price": float, "size": float} (any order -- sorted here).
    candle_times: list of float epoch-second timestamps, ascending, to
    compute a reading for.

    For each candle time t, replays the same window definition as
    exchange_feeds.TradeAggregator: trades with time in
    [t - window_seconds, t] (inclusive both ends, matching the live
    aggregator's `< cutoff` pruning rule exactly).

    Returns a list the same length as candle_times, each entry
    {"delta_ratio": float or None, "total_notional": float}. None/0 means
    no Binance.US trades were found in that candle's window -- typically
    because history wasn't fetched back that far, or Binance.US simply had
    no trades in that particular minute.
    """
    trades_sorted = sorted(trades, key=lambda tr: tr["time"])
    trade_times = [tr["time"] for tr in trades_sorted]
    n = len(trades_sorted)

    results = []
    left = 0
    right = 0
    buy = 0.0
    sell = 0.0

    for t in candle_times:
        window_start = t - window_seconds

        while right < n and trade_times[right] <= t:
            tr = trades_sorted[right]
            notional = tr["price"] * tr["size"]
            if tr["side"] == "buy":
                buy += notional
            else:
                sell += notional
            right += 1

        while left < right and trade_times[left] < window_start:
            tr = trades_sorted[left]
            notional = tr["price"] * tr["size"]
            if tr["side"] == "buy":
                buy -= notional
            else:
                sell -= notional
            left += 1

        total = buy + sell
        delta_ratio = (buy - sell) / total if total > 0 else None
        results.append({"delta_ratio": delta_ratio, "total_notional": total})

    return results


def run_agg_delta_backtest(cfg) -> dict:
    """
    Fetches Binance.US trade history, reconstructs agg_delta readings
    aligned to the candle-based backtest's timestamps, computes hit-rate
    buckets, merges them into backtest_results.json, and returns the
    updated results dict. Returns {"error": ...} without writing anything
    if a base backtest hasn't been run yet, the pair has no Binance.US
    mapping, or no usable trade history comes back.
    """
    base_results = backtest.load_results(cfg)
    if not base_results or "error" in base_results:
        return {"error": "No base backtest found -- run `python backtest.py` first."}

    symbol = exchange_feeds.to_binance_symbol(cfg.PRODUCT_ID)
    if not symbol:
        return {"error": f"No Binance.US symbol mapping for {cfg.PRODUCT_ID} -- can't backtest agg_delta for this pair."}

    # Re-fetch the same candle history the base backtest used, for exact
    # aligned timestamps and closing prices to check outcomes against.
    df = backtest.fetch_history(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.BACKTEST_HISTORY_CANDLES, cfg.BACKTEST_REQUEST_PAUSE)
    warmup = backtest._warmup_candles(cfg)
    lookahead = cfg.BACKTEST_LOOKAHEAD_CANDLES
    min_needed = warmup + lookahead + 10
    if df.empty or len(df) < min_needed:
        return {"error": f"Only got {len(df)} candles re-fetching history -- need at least {min_needed}."}

    # .values.astype("datetime64[ns]") forces nanosecond resolution before
    # the int64 cast, regardless of what resolution pandas happened to
    # construct df["time"] at (this varies by pandas version -- e.g. pandas
    # 3.x's pd.to_datetime(..., unit="s") produces datetime64[s], not [ns],
    # so blindly dividing by 1e9 after an int64 cast silently produces
    # near-1970 timestamps instead of raising an error).
    candle_times = (df["time"].values.astype("datetime64[ns]").astype("int64") / 1e9).tolist()

    end_ms = int(candle_times[-1] * 1000)
    start_ms = int(max(candle_times[0], candle_times[-1] - cfg.AGG_DELTA_BACKTEST_HOURS * 3600) * 1000)

    # Sanity check: if this ever computes an implausible range again (any
    # future resolution/timezone surprise), fail loudly with the actual
    # numbers instead of silently returning zero trades with no clue why.
    _MIN_PLAUSIBLE_MS = 1_600_000_000_000  # ~Sept 2020
    if end_ms < _MIN_PLAUSIBLE_MS:
        end_readable = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
        return {
            "error": (
                f"Computed an implausible end time ({end_readable.isoformat()}) from the "
                f"candle history -- this looks like a timestamp conversion bug, not a real "
                f"data issue. end_ms={end_ms}, start_ms={start_ms}. Please report this."
            )
        }

    print(
        f"Fetching Binance.US trade history for {symbol} ({cfg.AGG_DELTA_BACKTEST_HOURS}h window: "
        f"{datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to "
        f"{datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC)..."
    )
    trades = exchange_feeds.fetch_binance_us_trade_history(
        symbol, start_ms, end_ms,
        pause=cfg.AGG_DELTA_BACKTEST_REQUEST_PAUSE, max_requests=cfg.AGG_DELTA_BACKTEST_MAX_REQUESTS,
    )
    print(f"Fetched {len(trades)} trades.")

    if not trades:
        return {"error": "No Binance.US trades returned for this window -- nothing to reconstruct."}

    series = reconstruct_agg_delta_series(trades, candle_times, cfg.AGG_WINDOW_SECONDS)

    closes = df["close"].tolist()
    n = len(df)
    agg_bullish_hits = agg_bullish_total = 0
    agg_bearish_hits = agg_bearish_total = 0

    for i in range(warmup, n - lookahead):
        reading = series[i]
        if reading["total_notional"] <= 0 or reading["delta_ratio"] is None:
            continue  # no Binance.US coverage for this candle's window

        delta = reading["delta_ratio"]
        if delta > cfg.AGG_DELTA_THRESHOLD:
            direction = "bullish"
        elif delta < -cfg.AGG_DELTA_THRESHOLD:
            direction = "bearish"
        else:
            continue  # neutral -- not a directional call

        current_close = closes[i]
        future_close = closes[i + lookahead]
        if future_close > current_close:
            actual = "bullish"
        elif future_close < current_close:
            actual = "bearish"
        else:
            continue  # exact tie

        if direction == "bullish":
            agg_bullish_total += 1
            agg_bullish_hits += 1 if actual == "bullish" else 0
        else:
            agg_bearish_total += 1
            agg_bearish_hits += 1 if actual == "bearish" else 0

    base_results.setdefault("indicator_buckets", {})
    if agg_bullish_total > 0:
        base_results["indicator_buckets"]["agg_delta_bullish"] = {"hits": agg_bullish_hits, "total": agg_bullish_total}
    if agg_bearish_total > 0:
        base_results["indicator_buckets"]["agg_delta_bearish"] = {"hits": agg_bearish_hits, "total": agg_bearish_total}

    base_results["agg_delta_backtest"] = {
        "source": "binance_us_only",
        "hours": cfg.AGG_DELTA_BACKTEST_HOURS,
        "trades_fetched": len(trades),
        "candles_with_coverage": sum(1 for r in series if r["total_notional"] > 0),
        "candles_total": n,
    }

    with open(cfg.BACKTEST_RESULTS_PATH, "w") as f:
        json.dump(base_results, f, indent=2)

    return base_results


def _print_summary(results: dict, cfg):
    if "error" in results:
        print(f"[error] {results['error']}")
        return

    meta = results.get("agg_delta_backtest", {})
    print(
        f"\nDone. Reconstructed agg_delta from {meta.get('trades_fetched', '?')} Binance.US "
        f"trades over {meta.get('hours', '?')}h."
    )
    print(
        f"{meta.get('candles_with_coverage', '?')} of {meta.get('candles_total', '?')} candles "
        f"had Binance.US trade coverage in their window.\n"
    )

    for key in ["agg_delta_bullish", "agg_delta_bearish"]:
        b = results.get("indicator_buckets", {}).get(key)
        if not b:
            print(f"  {key:18s}    no directional readings in this window")
            continue
        rate = b["hits"] / b["total"] * 100 if b["total"] else 0.0
        flag = "  (low sample)" if b["total"] < cfg.BACKTEST_MIN_SAMPLE else ""
        print(f"  {key:18s}  {rate:5.1f}%  (n={b['total']}){flag}")

    print(
        f"\nSaved to {cfg.BACKTEST_RESULTS_PATH} -- weighting.py will pick this up "
        f"automatically from here on, and analyze_correlation.py now has real historical "
        f"data to compare agg_delta against, without needing to wait on live accumulation."
    )
    print(
        "\nRemember: this reflects Binance.US alone, not the combined 3-exchange live signal -- "
        "treat it as an approximation, not an exact replay of what the live agg_delta would "
        "have shown."
    )


if __name__ == "__main__":
    results = run_agg_delta_backtest(cfg)
    _print_summary(results, cfg)
