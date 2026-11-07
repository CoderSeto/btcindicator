"""
Tests for the candle-derived votes in signals._evaluate_price_signals at their boundaries.
Expected values are worked out by hand from the indicator definitions: on a perfectly flat
series every moving average equals the price, MACD and its signal line are both exactly 0, and
the Bollinger bands collapse onto the price (std = 0).
"""
import pandas as pd

import config
import indicators
import signals


def _flat(n=60, price=100.0):
    return pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="min"),
                         "open": price, "high": price, "low": price, "close": price, "volume": 1.0})


def test_a_perfectly_flat_market_casts_no_directional_votes():
    """
    Ties used to fall through to "bearish" (`"bullish" if fast > slow else "bearish"`), and a
    zero-width Bollinger band put the close "at or above the upper band". A dead-flat market --
    an illiquid pair, an exchange halt -- therefore read as 4 bearish votes: a composite bearish
    alert (threshold 3) out of no movement at all. Equal is not a lean.
    """
    df = indicators.compute_all(_flat(), config)
    snap = signals.evaluate_price_only(df, config)
    assert set(snap.individual.values()) == {"neutral"}, snap.individual
    assert snap.composite == "neutral"
    assert snap.strength == 0


def test_a_real_crossover_still_votes():
    # Rising then flat: fast SMA above slow, MACD above zero -> the state votes are bullish.
    closes = [100.0 + i for i in range(40)] + [139.0] * 3
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=len(closes), freq="min"),
                       "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0})
    df = indicators.compute_all(df, config)
    snap = signals.evaluate_price_only(df, config)
    assert snap.individual["sma_cross"] == "bullish"
    assert snap.individual["ema_cross"] == "bullish"
