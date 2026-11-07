"""
Fetch + compute pipeline for the dashboard, factored out from main.py's
polling loop so the Streamlit app can call it once per rerun and handle
errors gracefully (rather than retrying forever in a while-loop).
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import backtest
import data_fetch
import divergence
import indicators
import orderbook
import outlook as outlook_mod
import signals
import weighting


@dataclass
class DashboardState:
    df: Optional[pd.DataFrame] = None           # COMPLETED candles, with indicators
    # The price now, from Coinbase's ticker (data_fetch.latest_price). Signals
    # are scored on completed candles only (see data_fetch.drop_forming_candle);
    # what a person reads as "the price now", and the Kalshi model, use this.
    # If the ticker failed it's the newest candle's close, which can lag by a
    # minute or more (data_fetch.CANDLE_CACHE_NOTE) -- live_price_source says so.
    live_price: Optional[float] = None
    live_price_source: Optional[str] = None     # "ticker" or "candles"
    live_price_note: Optional[str] = None       # why it isn't the ticker, if it isn't
    price_df: Optional[pd.DataFrame] = None     # raw candles, forming one included (vol input)
    imbalance: Optional[float] = None
    snapshot: Optional["signals.SignalSnapshot"] = None
    divergence: dict = field(default_factory=dict)
    outlook: dict = field(default_factory=dict)
    backtest_results: Optional[dict] = None
    feed_snapshot: Optional[dict] = None
    book_snapshot: Optional[dict] = None
    weighted: dict = field(default_factory=dict)
    error: Optional[str] = None


def load_state(cfg, feed_snapshot: Optional[dict] = None,
               book_snapshot: Optional[dict] = None) -> DashboardState:
    """
    feed_snapshot is the current exchange_feeds snapshot dict (from
    FeedHandle.snapshot()), if the multi-exchange trade feeds are running.
    book_snapshot is the current orderbook_feeds snapshot dict (from
    BookFeedHandle.snapshot()), if the level-2 depth feeds are running.
    Pass None for either to run without it -- with no book_snapshot the
    order-book signal falls back to a single Coinbase REST snapshot.
    """
    have_book = bool(book_snapshot and book_snapshot.get("available"))
    try:
        raw = data_fetch.get_candles(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.CANDLE_HISTORY)
        # Only needed for the fallback path -- the live depth feeds already
        # carry a better read of the same thing, so skip the extra request.
        rest_book = None if have_book else data_fetch.get_order_book(cfg.PRODUCT_ID, level=2)
    except Exception as e:
        return DashboardState(error=str(e))

    df, _ = data_fetch.drop_forming_candle(raw, cfg.GRANULARITY)
    min_needed = max(cfg.SMA_SLOW, cfg.BOLLINGER_PERIOD, cfg.MACD_SLOW, cfg.RSI_PERIOD) + 2
    if df.empty or len(df) < min_needed:
        return DashboardState(error="Not enough candle history yet -- try again shortly.")

    df = indicators.compute_all(df, cfg)
    # None, not 0.0: the REST snapshot was never taken, which is a
    # different thing from a book that read as balanced.
    imbalance = None if have_book else orderbook.compute_imbalance(rest_book, cfg.ORDER_BOOK_DEPTH)

    agg_delta = None
    if feed_snapshot and feed_snapshot.get("total_notional", 0) > 0:
        agg_delta = feed_snapshot["delta_ratio"]

    snap = signals.evaluate(df, imbalance, cfg, agg_delta=agg_delta,
                            book_snapshot=book_snapshot)
    price_only_snap = signals.evaluate_price_only(df, cfg)
    div = divergence.detect_divergence(df, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)

    backtest_results = backtest.load_results(cfg)
    confidence = backtest.get_confidence(backtest_results, price_only_snap, div, cfg.BACKTEST_MIN_SAMPLE)
    weights = weighting.compute_weights(backtest_results, cfg)
    weighted = weighting.weighted_conviction(snap, weights)
    out = outlook_mod.build_outlook(snap, div, confidence, weighted)

    spot = data_fetch.latest_price(cfg.PRODUCT_ID, raw)
    return DashboardState(
        df=df, live_price=spot["price"], live_price_source=spot["source"],
        live_price_note=spot["reason"], price_df=raw,
        imbalance=imbalance, snapshot=snap, divergence=div,
        outlook=out, backtest_results=backtest_results, feed_snapshot=feed_snapshot,
        book_snapshot=book_snapshot, weighted=weighted,
    )
