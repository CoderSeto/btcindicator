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
    df: Optional[pd.DataFrame] = None
    imbalance: float = 0.0
    snapshot: Optional["signals.SignalSnapshot"] = None
    divergence: dict = field(default_factory=dict)
    outlook: dict = field(default_factory=dict)
    backtest_results: Optional[dict] = None
    feed_snapshot: Optional[dict] = None
    weighted: dict = field(default_factory=dict)
    error: Optional[str] = None


def load_state(cfg, feed_snapshot: Optional[dict] = None) -> DashboardState:
    """
    feed_snapshot is the current exchange_feeds snapshot dict (from
    FeedHandle.snapshot()), if the multi-exchange feeds are running --
    pass None to run on Coinbase REST data alone.
    """
    try:
        df = data_fetch.get_candles(cfg.PRODUCT_ID, cfg.GRANULARITY, cfg.CANDLE_HISTORY)
        book = data_fetch.get_order_book(cfg.PRODUCT_ID, level=2)
    except Exception as e:
        return DashboardState(error=str(e))

    min_needed = max(cfg.SMA_SLOW, cfg.BOLLINGER_PERIOD, cfg.MACD_SLOW, cfg.RSI_PERIOD) + 2
    if df.empty or len(df) < min_needed:
        return DashboardState(error="Not enough candle history yet -- try again shortly.")

    df = indicators.compute_all(df, cfg)
    imbalance = orderbook.compute_imbalance(book, cfg.ORDER_BOOK_DEPTH)

    agg_delta = None
    if feed_snapshot and feed_snapshot.get("total_notional", 0) > 0:
        agg_delta = feed_snapshot["delta_ratio"]

    snap = signals.evaluate(df, imbalance, cfg, agg_delta=agg_delta)
    price_only_snap = signals.evaluate_price_only(df, cfg)
    div = divergence.detect_divergence(df, cfg.DIVERGENCE_LOOKBACK, cfg.DIVERGENCE_SWING_ORDER)

    backtest_results = backtest.load_results(cfg)
    confidence = backtest.get_confidence(backtest_results, price_only_snap, div, cfg.BACKTEST_MIN_SAMPLE)
    weights = weighting.compute_weights(backtest_results, cfg)
    weighted = weighting.weighted_conviction(snap, weights)
    out = outlook_mod.build_outlook(snap, div, confidence, weighted)

    return DashboardState(
        df=df, imbalance=imbalance, snapshot=snap, divergence=div,
        outlook=out, backtest_results=backtest_results, feed_snapshot=feed_snapshot,
        weighted=weighted,
    )
