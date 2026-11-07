"""
Tests for divergence.py.

Price series are built with a small zig-zag helper so every swing high/low is
known by construction (with order=2: the peak sits at index 2 + 4*j, interior
troughs at 4*j). RSI is then set by hand at exactly those indices, so which
pair of swings divergence detection compares -- and the verdict -- is known
before the code runs. _swing_points is also checked against a brute force
over random series full of ties.
"""
import random

import numpy as np
import pandas as pd
import pytest

import divergence

ORDER = 2


def zigzag(peaks, troughs=None):
    """peak j at index 4j+2, trough j at index 4j. Mid-points keep each peak/trough
    a strict extremum of its +/-2 window."""
    troughs = troughs or [1.0] * (len(peaks) + 1)
    assert len(troughs) == len(peaks) + 1
    xs = []
    for j, p in enumerate(peaks):
        xs += [troughs[j], (troughs[j] + p) / 2, p, (p + troughs[j + 1]) / 2]
    xs.append(troughs[-1])
    return xs


def build(peaks, rsi_at_peaks=None, rsi_at_troughs=None, troughs=None, base_rsi=50.0):
    price = zigzag(peaks, troughs)
    rsi = [base_rsi] * len(price)
    for j, v in enumerate(rsi_at_peaks or []):
        rsi[4 * j + 2] = v
    for j, v in (rsi_at_troughs or {}).items():
        rsi[4 * j] = v
    return pd.DataFrame({"close": price, "rsi": rsi})


def run(df, lookback=None):
    return divergence.detect_divergence(df, lookback or len(df), ORDER)


# ---------- _swing_points ----------

def test_swing_points_zigzag_known_positions():
    s = pd.Series(zigzag([3, 5, 4]))
    highs, lows = divergence._swing_points(s, ORDER)
    assert highs == [2, 6, 10]
    assert lows == [4, 8]            # outer troughs (idx 0 and 12) are too close to the edge


def test_swing_points_match_brute_force_on_random_series_with_ties():
    rng = random.Random(42)
    for _ in range(200):
        n = rng.randint(1, 40)
        order = rng.randint(1, 4)
        vals = [rng.randint(0, 5) for _ in range(n)]      # small range -> lots of ties
        highs, lows = divergence._swing_points(pd.Series(vals), order)
        want_h, want_l = [], []
        for i in range(order, n - order):
            neighbours = vals[i - order:i] + vals[i + 1:i + order + 1]
            if all(vals[i] > v for v in neighbours):
                want_h.append(i)
            if all(vals[i] < v for v in neighbours):
                want_l.append(i)
        assert (highs, lows) == (want_h, want_l), vals


def test_swing_points_plateau_is_not_a_swing():
    # Two equal tops at idx 2 and 3: neither is a *strict* maximum of its window.
    assert divergence._swing_points(pd.Series([1, 2, 5, 5, 2, 1, 0, 1]), 2) == ([], [])


def test_swing_points_too_short_series():
    assert divergence._swing_points(pd.Series([1.0, 2.0, 3.0, 4.0]), 2) == ([], [])
    assert divergence._swing_points(pd.Series([], dtype=float), 2) == ([], [])


# ---------- bearish ----------

def test_bearish_higher_high_with_lower_rsi_high():
    out = run(build([3, 5], rsi_at_peaks=[70, 60]))
    assert out["type"] == "bearish"
    assert "$3 to $5" in out["detail"] and "70 to 60" in out["detail"]


def test_no_bearish_when_rsi_confirms_the_new_high():
    out = run(build([3, 5], rsi_at_peaks=[60, 70]))
    assert out["type"] is None and out["detail"] == "No clear divergence in the recent window."


def test_no_bearish_for_lower_high_or_equal_high():
    assert run(build([5, 3], rsi_at_peaks=[70, 60]))["type"] is None      # lower high: not a divergence
    assert run(build([4, 4], rsi_at_peaks=[70, 60]))["type"] is None      # equal highs are not "higher"


def test_no_bearish_when_rsi_is_equal():
    assert run(build([3, 5], rsi_at_peaks=[65, 65]))["type"] is None


def test_only_the_two_most_recent_highs_are_compared():
    # First pair (3 -> 5, rsi 70 -> 60) WOULD be bearish; the last pair (5 -> 4) is not.
    assert run(build([3, 5, 4], rsi_at_peaks=[70, 60, 65]))["type"] is None
    # Last pair is the divergent one (4 -> 6, rsi 65 -> 55).
    assert run(build([3, 4, 6], rsi_at_peaks=[50, 65, 55]))["type"] == "bearish"


# ---------- bullish ----------

def test_bullish_lower_low_with_higher_rsi_low():
    # Three peaks give two interior troughs (idx 4 and 8): 2.0 then 1.0 -> a lower low.
    df = build([6, 6.5, 7], troughs=[3, 2.0, 1.0, 3], rsi_at_troughs={1: 30, 2: 40})
    out = run(df)
    assert out["type"] == "bullish"
    assert "$2 to $1" in out["detail"] and "30 to 40" in out["detail"]


def test_no_bullish_when_rsi_confirms_the_new_low():
    df = build([6, 6.5, 7], troughs=[3, 2.0, 1.0, 3], rsi_at_troughs={1: 40, 2: 30})
    assert run(df)["type"] is None


def test_no_bullish_for_higher_or_equal_low():
    assert run(build([6, 6.5, 7], troughs=[3, 1.0, 2.0, 3], rsi_at_troughs={1: 30, 2: 40}))["type"] is None
    assert run(build([6, 6.5, 7], troughs=[3, 2.0, 2.0, 3], rsi_at_troughs={1: 30, 2: 40}))["type"] is None


# ---------- precedence, windows, guards ----------

def test_bearish_wins_when_both_patterns_are_present():
    # Highs 5 -> 6 with RSI down (bearish) AND lows 2.0 -> 1.5 with RSI up (bullish).
    df = build([3, 5, 6], rsi_at_peaks=[50, 70, 60], troughs=[1, 2.0, 1.5, 1], rsi_at_troughs={1: 30, 2: 40})
    assert run(df)["type"] == "bearish"
    # Remove the bearish half (RSI now confirms the high) and bullish is reported.
    df2 = build([3, 5, 6], rsi_at_peaks=[50, 60, 70], troughs=[1, 2.0, 1.5, 1], rsi_at_troughs={1: 30, 2: 40})
    assert run(df2)["type"] == "bullish"


def test_lookback_window_ignores_older_candles():
    df = build([3, 5], rsi_at_peaks=[70, 60])
    junk = pd.DataFrame({"close": [100.0] * 10, "rsi": [50.0] * 10})
    padded = pd.concat([junk, df], ignore_index=True)
    assert divergence.detect_divergence(padded, len(df), ORDER)["type"] == "bearish"
    # Shrink the window so the first peak (index 2 of the pattern) falls outside it.
    assert divergence.detect_divergence(padded, len(df) - 3, ORDER)["type"] is None


def test_detect_does_not_depend_on_dataframe_index():
    df = build([3, 5], rsi_at_peaks=[70, 60])
    df.index = range(1000, 1000 + len(df))
    assert run(df)["type"] == "bearish"


def test_not_enough_history_and_missing_rsi():
    short = pd.DataFrame({"close": [1.0] * (ORDER * 2 + 1), "rsi": [50.0] * (ORDER * 2 + 1)})
    assert "Not enough candle history" in run(short)["detail"] and run(short)["type"] is None
    no_rsi = pd.DataFrame({"close": zigzag([3, 5])})
    out = divergence.detect_divergence(no_rsi, 20, ORDER)
    assert out["type"] is None and "Not enough candle history" in out["detail"]
    # Exactly order*2+2 rows is the smallest window it will analyse (no swings -> plain "no divergence").
    edge = pd.DataFrame({"close": np.arange(ORDER * 2 + 2, dtype=float), "rsi": [50.0] * (ORDER * 2 + 2)})
    assert run(edge)["detail"] == "No clear divergence in the recent window."


def test_flat_market_has_no_divergence():
    flat = pd.DataFrame({"close": [100.0] * 40, "rsi": [50.0] * 40})
    assert run(flat)["type"] is None


def test_always_returns_type_and_detail():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(0, 60))
        df = pd.DataFrame({"close": rng.normal(100, 2, n), "rsi": rng.uniform(0, 100, n)})
        out = divergence.detect_divergence(df, 40, 3)
        assert set(out) == {"type", "detail"} and out["type"] in (None, "bullish", "bearish")
