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
from typing import Optional

import requests

import backtest
import config as cfg
import data_fetch
import divergence
import exchange_feeds
import indicators
import orderbook
import orderbook_feeds
import outlook
import signals
import weighting
from csv_utils import append_row


# Fixed column order for both alert and signal logs, independent of which
# indicators happen to be present in any given snapshot -- agg_delta and
# book_flow in particular only show up once the multi-exchange feeds have
# collected enough data, so without a fixed schema the CSV header would
# drift between "warmed up" and "not yet warmed up" cycles and break later
# parsing (this exact bug previously existed in log_alert() below -- fixed
# here).
ALL_SIGNAL_NAMES = [
    "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume",
    "orderbook", "book_flow", "agg_delta",
]

# Numeric order-book context logged alongside the votes. Each is blank
# whenever the depth feeds aren't running or haven't warmed up -- same
# fixed-schema rule as the votes above.
#
# `imbalance` and `book_imbalance` are deliberately SEPARATE columns rather
# than one column whose meaning depends on what was connected at the time.
# The first is the legacy single-snapshot Coinbase REST reading (unweighted
# base volume over a fixed level count); the second is the multi-venue,
# notional, distance-weighted one. They are on different scales and are not
# interchangeable, and quietly writing both into one column would make the
# resulting file impossible to analyze after the fact.
BOOK_LOG_FIELDS = ["book_imbalance", "book_agreement", "book_persistence_s", "book_flow_value"]

ALERT_LOG_HEADER = (
    ["timestamp", "price", "composite", "strength", "imbalance"]
    + BOOK_LOG_FIELDS + ALL_SIGNAL_NAMES
)
SIGNAL_LOG_HEADER = ["timestamp", "price"] + BOOK_LOG_FIELDS + ALL_SIGNAL_NAMES


def _book_log_values(snapshot: signals.SignalSnapshot) -> list:
    """Blank cells, not zeros, when there's no book read -- zero is a real
    reading (a balanced book) and must not be confused with 'no data'."""
    book = snapshot.book or {}
    if not book.get("available"):
        return ["" for _ in BOOK_LOG_FIELDS]
    flow = book.get("flow_value")
    return [
        round(book["imbalance"], 4),
        round(book["agreement"], 3),
        round(book["persistence_seconds"], 1),
        round(flow, 4) if flow is not None else "",
    ]


def _now() -> datetime:
    """The logs' clock (a function so tests can move it)."""
    return datetime.now(timezone.utc)


def _last_row(path: str, header: list) -> Optional[dict]:
    """
    The last data row of a log as {column: text}, or None if there isn't one
    (missing file, header only, or a last line that doesn't fit the header).
    Reads only the file's tail, so it stays cheap however long the log gets.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 16_384))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    values = next(csv.reader([lines[-1]]))
    if values == header or len(values) != len(header):
        return None
    return dict(zip(header, values))


def _is_repeat(path: str, header: list, row: list, same_fields=()) -> bool:
    """
    True if the log's last row is less than LOG_MIN_INTERVAL_SECONDS old and
    matches `row` on every field in `same_fields` -- i.e. this would be the
    same observation logged again by another rerun, tab or process.
    """
    if cfg.LOG_MIN_INTERVAL_SECONDS <= 0:
        return False
    last = _last_row(path, header)
    if last is None:
        return False
    try:
        age = (_now() - datetime.fromisoformat(last["timestamp"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return False
    if not 0 <= age < cfg.LOG_MIN_INTERVAL_SECONDS:
        return False
    new = dict(zip(header, row))
    return all(last[f] == str(new[f]) for f in same_fields)


def log_alert(snapshot: signals.SignalSnapshot):
    """
    Appends one alert row -- unless the log's last row is the same alert
    (direction and strength) from less than LOG_MIN_INTERVAL_SECONDS ago,
    which is a rerun or another tab/process seeing the same thing.
    """
    row = (
        [
            _now().isoformat(),
            snapshot.price,
            snapshot.composite,
            snapshot.strength,
            round(snapshot.imbalance, 3) if snapshot.imbalance is not None else "",
        ]
        + _book_log_values(snapshot)
        + [snapshot.individual.get(name, "") for name in ALL_SIGNAL_NAMES]
    )
    # append_row checks for a repeat under its file lock, so two writers can't
    # both decide the row is new.
    append_row(cfg.ALERT_LOG_PATH, ALERT_LOG_HEADER, row,
               skip_if=lambda: _is_repeat(cfg.ALERT_LOG_PATH, ALERT_LOG_HEADER, row,
                                          same_fields=("composite", "strength")))


def log_signal_snapshot(snapshot: signals.SignalSnapshot):
    """
    Logs every indicator's vote for this cycle, every cycle -- not just
    alert-triggering moments like log_alert(). Only active when
    cfg.LOG_ALL_SIGNALS is True (off by default, since it's a
    continuously-growing file). Used by analyze_correlation.py.
    Any indicator missing from this cycle's snapshot (most commonly
    agg_delta or book_flow, before the feeds have warmed up) is logged as
    an empty cell. At most one row per LOG_MIN_INTERVAL_SECONDS, so the row
    count measures observations rather than dashboard reruns.
    """
    if not cfg.LOG_ALL_SIGNALS:
        return
    row = (
        [_now().isoformat(), snapshot.price]
        + _book_log_values(snapshot)
        + [snapshot.individual.get(name, "") for name in ALL_SIGNAL_NAMES]
    )
    append_row(cfg.SIGNAL_LOG_PATH, SIGNAL_LOG_HEADER, row,
               skip_if=lambda: _is_repeat(cfg.SIGNAL_LOG_PATH, SIGNAL_LOG_HEADER, row))


_ABBR = {"bullish": "BULL", "bearish": "BEAR", "neutral": "NEUT"}


def format_status_line(snapshot: signals.SignalSnapshot) -> str:
    tags = " ".join(f"{name}:{_ABBR[val]}" for name, val in snapshot.individual.items())
    book = snapshot.book or {}
    if book.get("available"):
        imbalance_txt = (
            f"book={book['imbalance']:+.2f} "
            f"({book['venue_count']}v, {book['agreement'] * 100:.0f}% agree, "
            f"held {book['persistence_seconds']:.0f}s)"
        )
    else:
        imbalance_txt = (f"imbalance={snapshot.imbalance:+.2f}"
                          if snapshot.imbalance is not None else "imbalance=--")
    return (
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"${snapshot.price:,.2f}  "
        f"composite={snapshot.composite.upper()} ({snapshot.strength}/{len(snapshot.individual)})  "
        f"{imbalance_txt}  "
        f"{tags}"
    )


def format_book_lines(snapshot: signals.SignalSnapshot) -> list:
    """
    The order-book detail lines printed under the status line: what the
    aggregated book looks like, which way resting liquidity is moving, and
    any walls that have actually held rather than flickered.

    Held walls are the only ones shown. A large level that appears and
    vanishes as price approaches is indistinguishable from a spoof, and
    printing it as though it were a real barrier would be worse than
    saying nothing.
    """
    book = snapshot.book or {}
    if not book.get("available"):
        return []

    lines = [
        f"book: bid ${book['bid_notional']:,.0f} / ask ${book['ask_notional']:,.0f} "
        f"within band  |  near {book['imbalance_near']:+.2f}, far {book['imbalance_far']:+.2f}, "
        f"microprice {book['microprice_bps']:+.2f}bps"
    ]

    if book.get("flow_available"):
        lines.append(
            f"flow: resting bids {book['bid_change_pct']:+.1f}%, asks {book['ask_change_pct']:+.1f}% "
            f"over {book['flow_window_seconds']:.0f}s (net {book['flow_value']:+.2f})"
        )
    elif book.get("flow_reason"):
        lines.append(f"flow: unavailable -- {book['flow_reason']}")

    for wall in book.get("held_walls", [])[:3]:
        side_word = "support" if wall["side"] == "bid" else "resistance"
        lines.append(
            f"wall: ${wall['notional']:,.0f} resting {wall['distance_bps']:.0f}bps "
            f"{'below' if wall['side'] == 'bid' else 'above'} mid on {wall['venue']} "
            f"({side_word}, held {wall['age_seconds']:.0f}s)"
        )

    return lines


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

    book_handle = None
    if cfg.ENABLE_BOOK_FEEDS:
        print("Starting live level-2 order-book feeds (Coinbase/Binance.US/Bybit depth)...")
        print(
            f"  the order-book signal needs {cfg.BOOK_PERSISTENCE_SECONDS:.0f}s of a held lean "
            f"and flow needs {cfg.BOOK_FLOW_WINDOW_SECONDS:.0f}s of history, so both stay "
            "neutral for the first minute or so."
        )
        book_handle = orderbook_feeds.start_book_feeds(cfg)

    try:
        while True:
            try:
                df = data_fetch.get_candles(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.CANDLE_HISTORY)
                df = indicators.compute_all(df, cfg)

                book_snapshot = book_handle.snapshot() if book_handle is not None else None

                # The REST book is only needed for the fallback path. Once
                # the depth feeds are live they already carry a far better
                # read of the same thing, so skip the extra request.
                imbalance = None
                if not (book_snapshot and book_snapshot.get("available")):
                    rest_book = data_fetch.get_order_book(cfg.PRODUCT_ID, level=2)
                    imbalance = orderbook.compute_imbalance(rest_book, cfg.ORDER_BOOK_DEPTH)

                agg_delta = None
                if feed_handle is not None:
                    feed_snapshot = feed_handle.snapshot()
                    if feed_snapshot["total_notional"] > 0:
                        agg_delta = feed_snapshot["delta_ratio"]

                snapshot = signals.evaluate(df, imbalance, cfg, agg_delta=agg_delta,
                                            book_snapshot=book_snapshot)
                price_only_snapshot = signals.evaluate_price_only(df, cfg)
                div = divergence.detect_divergence(df, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)

                backtest_results = backtest.load_results(cfg)
                confidence = backtest.get_confidence(backtest_results, price_only_snapshot, div, cfg.BACKTEST_MIN_SAMPLE)
                weights = weighting.compute_weights(backtest_results, cfg)
                weighted = weighting.weighted_conviction(snapshot, weights)
                outlook_result = outlook.build_outlook(snapshot, div, confidence, weighted)
                log_signal_snapshot(snapshot)

                print(format_status_line(snapshot))
                for line in format_book_lines(snapshot):
                    print(f"  {line}")
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
        if book_handle is not None:
            book_handle.stop()


if __name__ == "__main__":
    run()
