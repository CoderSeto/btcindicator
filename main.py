"""
BTC Indicator Bot
------------------
Polls Coinbase's public market data for BTC-USD, computes a set of
technical + order-book indicators, and prints/logs a signal whenever
enough of them agree on direction.

This is a monitoring/alerting tool -- it does NOT place any trades.
See README.md for details and config.py for tuning.
"""
import csv
import os
import time
from datetime import datetime, timezone

import requests

import backtest
import config as cfg
import data_fetch
import divergence
import exchange_feeds
import indicators
import orderbook
import outlook
import signals
import weighting


# Fixed column order for both alert and signal logs, independent of which
# indicators happen to be present in any given snapshot -- agg_delta in
# particular only shows up once the multi-exchange feeds have collected
# trades, so without a fixed schema the CSV header would drift between
# "warmed up" and "not yet warmed up" cycles and break later parsing
# (this exact bug previously existed in log_alert() below -- fixed here).
ALL_SIGNAL_NAMES = [
    "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "orderbook", "agg_delta",
]


def log_alert(snapshot: signals.SignalSnapshot):
    file_exists = os.path.isfile(cfg.ALERT_LOG_PATH)
    with open(cfg.ALERT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "price", "composite", "strength", "imbalance"]
                + ALL_SIGNAL_NAMES
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                snapshot.price,
                snapshot.composite,
                snapshot.strength,
                round(snapshot.imbalance, 3),
            ]
            + [snapshot.individual.get(name, "") for name in ALL_SIGNAL_NAMES]
        )


def log_signal_snapshot(snapshot: signals.SignalSnapshot):
    """
    Logs every indicator's vote for this cycle, every cycle -- not just
    alert-triggering moments like log_alert(). Only active when
    cfg.LOG_ALL_SIGNALS is True (off by default, since it's a
    continuously-growing file). Used by analyze_correlation.py.
    Any indicator missing from this cycle's snapshot (most commonly
    agg_delta, before the feeds have warmed up) is logged as an empty cell.
    """
    if not cfg.LOG_ALL_SIGNALS:
        return
    file_exists = os.path.isfile(cfg.SIGNAL_LOG_PATH)
    with open(cfg.SIGNAL_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "price"] + ALL_SIGNAL_NAMES)
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), snapshot.price]
            + [snapshot.individual.get(name, "") for name in ALL_SIGNAL_NAMES]
        )


_ABBR = {"bullish": "BULL", "bearish": "BEAR", "neutral": "NEUT"}


def format_status_line(snapshot: signals.SignalSnapshot) -> str:
    tags = " ".join(f"{name}:{_ABBR[val]}" for name, val in snapshot.individual.items())
    return (
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"${snapshot.price:,.2f}  "
        f"composite={snapshot.composite.upper()} ({snapshot.strength}/{len(snapshot.individual)})  "
        f"imbalance={snapshot.imbalance:+.2f}  "
        f"{tags}"
    )


def _print_whale_alert(event: dict):
    t = datetime.fromtimestamp(event["time"]).strftime("%H:%M:%S")
    print(
        f"  \U0001F433 [{t}] whale trade on {event['exchange']}: "
        f"{event['side'].upper()} ${event['notional']:,.0f} ({event['size']:.4f} @ ${event['price']:,.2f})"
    )


def _print_liquidation_alert(event: dict):
    t = datetime.fromtimestamp(event["time"]).strftime("%H:%M:%S")
    print(
        f"  \U0001F4A5 [{t}] liquidation on {event['exchange']}: "
        f"{event['liquidated_side'].upper()} position, ${event['notional']:,.0f} "
        f"({event['size']:.4f} @ ${event['price']:,.2f})"
    )


def run():
    print(f"Starting BTC indicator bot for {cfg.PRODUCT_ID}")
    print(
        f"Polling every {cfg.POLL_INTERVAL}s, {cfg.GRANULARITY}s candles, "
        f"composite alert threshold = {cfg.COMPOSITE_ALERT_THRESHOLD} aligned signals\n"
    )

    feed_handle = None
    if cfg.ENABLE_EXCHANGE_FEEDS:
        print("Starting live multi-exchange feeds (Coinbase/Binance/Bybit trades + liquidations)...")
        feed_handle = exchange_feeds.start_feeds(
            cfg, on_whale_trade=_print_whale_alert, on_liquidation=_print_liquidation_alert,
        )

    try:
        while True:
            try:
                df = data_fetch.get_candles(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.CANDLE_HISTORY)
                book = data_fetch.get_order_book(cfg.PRODUCT_ID, level=2)

                df = indicators.compute_all(df, cfg)
                imbalance = orderbook.compute_imbalance(book, cfg.ORDER_BOOK_DEPTH)

                agg_delta = None
                if feed_handle is not None:
                    feed_snapshot = feed_handle.snapshot()
                    if feed_snapshot["total_notional"] > 0:
                        agg_delta = feed_snapshot["delta_ratio"]

                snapshot = signals.evaluate(df, imbalance, cfg, agg_delta=agg_delta)
                price_only_snapshot = signals.evaluate_price_only(df, cfg)
                div = divergence.detect_divergence(df, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)

                backtest_results = backtest.load_results(cfg)
                confidence = backtest.get_confidence(backtest_results, price_only_snapshot, div, cfg.BACKTEST_MIN_SAMPLE)
                weights = weighting.compute_weights(backtest_results, cfg)
                weighted = weighting.weighted_conviction(snapshot, weights)
                outlook_result = outlook.build_outlook(snapshot, div, confidence, weighted)
                log_signal_snapshot(snapshot)

                print(format_status_line(snapshot))
                print(f"  outlook: {outlook_result['headline']}")
                if outlook_result["detail"]:
                    print(f"           {outlook_result['detail']}")
                if outlook_result["caution"]:
                    print(f"  !! {outlook_result['caution']}")
                if outlook_result["weighted_line"]:
                    print(f"  weighted: {outlook_result['weighted_line']}")
                for line in outlook_result["confidence_lines"]:
                    print(f"  confidence: {line}")
                if not confidence.get("available"):
                    print("  confidence: no backtest yet -- run `python backtest.py` to see historical hit rates")

                if snapshot.strength >= cfg.COMPOSITE_ALERT_THRESHOLD:
                    print(f"  >>> ALERT: {snapshot.strength} indicators aligned {snapshot.composite.upper()}")
                    log_alert(snapshot)

            except requests.exceptions.RequestException as e:
                print(f"[network error] {e}")
            except Exception as e:
                print(f"[error] {e}")

            time.sleep(cfg.POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if feed_handle is not None:
            feed_handle.stop()


if __name__ == "__main__":
    run()
