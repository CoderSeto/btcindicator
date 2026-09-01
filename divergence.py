"""
RSI/price divergence detector -- a classic technique for flagging potential
trend reversals, as opposed to the indicators in indicators.py, which mostly
describe where price already is or has just been.

Bearish divergence: price makes a higher high while RSI makes a lower high
at the same point -- upward momentum is fading even as price pushes up.

Bullish divergence: price makes a lower low while RSI makes a higher low
at the same point -- downward momentum is fading even as price pushes down.
"""
import pandas as pd


def _swing_points(series: pd.Series, order: int):
    """
    Indices (positional) where `series` is a strict local max/min within
    a +/- `order` window. Returns (high_indices, low_indices).
    """
    values = series.values
    n = len(values)
    highs, lows = [], []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == window.max() and (window == values[i]).sum() == 1:
            highs.append(i)
        if values[i] == window.min() and (window == values[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def detect_divergence(df: pd.DataFrame, lookback: int, order: int) -> dict:
    """
    Scans the last `lookback` candles for the most recent divergence pattern.
    Returns {"type": "bullish" | "bearish" | None, "detail": str}
    """
    window = df.tail(lookback).reset_index(drop=True)
    if len(window) < order * 2 + 2 or "rsi" not in window:
        return {"type": None, "detail": "Not enough candle history yet for divergence detection."}

    price = window["close"]
    rsi = window["rsi"]
    price_highs, price_lows = _swing_points(price, order)

    if len(price_highs) >= 2:
        i1, i2 = price_highs[-2], price_highs[-1]
        if price.iloc[i2] > price.iloc[i1] and rsi.iloc[i2] < rsi.iloc[i1]:
            return {
                "type": "bearish",
                "detail": (
                    f"Price made a higher high (${price.iloc[i1]:,.0f} to ${price.iloc[i2]:,.0f}) "
                    f"while RSI made a lower high ({rsi.iloc[i1]:.0f} to {rsi.iloc[i2]:.0f}) -- "
                    "upward momentum looks like it's fading."
                ),
            }

    if len(price_lows) >= 2:
        i1, i2 = price_lows[-2], price_lows[-1]
        if price.iloc[i2] < price.iloc[i1] and rsi.iloc[i2] > rsi.iloc[i1]:
            return {
                "type": "bullish",
                "detail": (
                    f"Price made a lower low (${price.iloc[i1]:,.0f} to ${price.iloc[i2]:,.0f}) "
                    f"while RSI made a higher low ({rsi.iloc[i1]:.0f} to {rsi.iloc[i2]:.0f}) -- "
                    "downward momentum looks like it's fading."
                ),
            }

    return {"type": None, "detail": "No clear divergence in the recent window."}
