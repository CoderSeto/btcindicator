"""
Tests for how the order-book read turns into votes.

The persistence gate is the reason the live book is maintained at all, so
it gets covered explicitly: a lopsided book that has only just appeared
must NOT vote, and the same book once it has held must.
"""
import pandas as pd
import pytest

import config as base_cfg
import signals


class Cfg:
    """A copy of the real config with only what these tests read, so a
    later tuning change to config.py can't quietly invalidate them."""
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    IMBALANCE_THRESHOLD = 0.20
    AGG_DELTA_THRESHOLD = 0.15
    BOOK_IMBALANCE_THRESHOLD = 0.18
    BOOK_PERSISTENCE_SECONDS = 20.0
    BOOK_FLOW_THRESHOLD = 0.20
    ENABLE_BOOK_FLOW_SIGNAL = True


def flat_candles():
    """Two candles with every candle-based indicator sitting neutral-ish,
    so the order-book signals are the only thing under test."""
    row = {
        "close": 100.0, "open": 100.0, "sma_9": 100.0, "sma_21": 100.0,
        "ema_12": 100.0, "ema_26": 100.0, "rsi": 50.0, "macd": 0.0,
        "macd_signal": 0.0, "bb_upper": 110.0, "bb_lower": 90.0,
        "volume_spike": False,
    }
    return pd.DataFrame([row, row])


def book_snapshot(imbalance=0.5, direction="bullish", held_seconds=30.0,
                   flow_value=0.0, flow_available=True):
    return {
        "available": True,
        "aggregate": {
            "imbalance": imbalance, "imbalance_near": imbalance, "imbalance_far": imbalance,
            "microprice_bps": 0.0, "agreement": 1.0,
            "bid_notional": 1_000_000.0, "ask_notional": 500_000.0,
            "venues": {}, "venue_count": 2, "excluded": {}, "walls": [], "band_bps": 25.0,
        },
        "persistence": {"seconds": held_seconds, "direction": direction,
                        "mean_imbalance": imbalance},
        "flow": ({"available": True, "value": flow_value, "bid_change_pct": 0.0,
                  "ask_change_pct": 0.0, "window_seconds": 60.0}
                 if flow_available else {"available": False, "reason": "not enough history yet"}),
        "walls": [], "samples": 120, "history_seconds": 120.0, "status": {},
    }


# --- the persistence gate -------------------------------------------------

def test_a_held_lean_votes():
    snap = signals.evaluate(flat_candles(), 0.0, Cfg,
                            book_snapshot=book_snapshot(direction="bullish", held_seconds=30.0))
    assert snap.individual["orderbook"] == "bullish"


def test_a_lean_that_has_not_held_long_enough_stays_neutral():
    """
    Same lopsided book, seen for 5 seconds instead of 30. This is the
    whole point of the gate: one frame of a lopsided book is noise.
    """
    snap = signals.evaluate(flat_candles(), 0.0, Cfg,
                            book_snapshot=book_snapshot(direction="bullish", held_seconds=5.0))
    assert snap.individual["orderbook"] == "neutral"


def test_no_lean_at_all_stays_neutral():
    snap = signals.evaluate(flat_candles(), 0.0, Cfg,
                            book_snapshot=book_snapshot(direction=None, held_seconds=0.0))
    assert snap.individual["orderbook"] == "neutral"


def test_a_held_bearish_lean_votes_bearish():
    snap = signals.evaluate(flat_candles(), 0.0, Cfg,
                            book_snapshot=book_snapshot(imbalance=-0.5, direction="bearish",
                                                         held_seconds=45.0))
    assert snap.individual["orderbook"] == "bearish"


# --- fallback when the depth feeds aren't running -------------------------

def test_falls_back_to_the_single_snapshot_reading_without_depth_feeds():
    bullish = signals.evaluate(flat_candles(), 0.5, Cfg, book_snapshot=None)
    assert bullish.individual["orderbook"] == "bullish"

    bearish = signals.evaluate(flat_candles(), -0.5, Cfg, book_snapshot=None)
    assert bearish.individual["orderbook"] == "bearish"

    flat = signals.evaluate(flat_candles(), 0.05, Cfg, book_snapshot=None)
    assert flat.individual["orderbook"] == "neutral"


def test_a_missing_legacy_reading_does_not_crash_the_fallback():
    """imbalance=None means the REST snapshot was skipped -- the fallback
    must read that as 'nothing to say', not blow up."""
    snap = signals.evaluate(flat_candles(), None, Cfg, book_snapshot=None)
    assert snap.individual["orderbook"] == "neutral"
    assert snap.imbalance is None


def test_an_unavailable_book_snapshot_uses_the_fallback_not_a_crash():
    snap = signals.evaluate(flat_candles(), 0.5, Cfg,
                            book_snapshot={"available": False, "reason": "still connecting"})
    assert snap.individual["orderbook"] == "bullish"
    assert snap.book == {"available": False}


# --- book flow ------------------------------------------------------------

def test_book_flow_votes_on_direction_past_the_threshold():
    bullish = signals.evaluate(flat_candles(), 0.0, Cfg,
                               book_snapshot=book_snapshot(flow_value=0.6))
    assert bullish.individual["book_flow"] == "bullish"

    bearish = signals.evaluate(flat_candles(), 0.0, Cfg,
                               book_snapshot=book_snapshot(flow_value=-0.6))
    assert bearish.individual["book_flow"] == "bearish"

    inside = signals.evaluate(flat_candles(), 0.0, Cfg,
                              book_snapshot=book_snapshot(flow_value=0.05))
    assert inside.individual["book_flow"] == "neutral"


def test_book_flow_is_absent_entirely_until_it_has_data():
    """
    Not "neutral" -- absent. A missing signal must not be recorded as a
    neutral vote it didn't earn; the loggers write a blank cell for it.
    """
    snap = signals.evaluate(flat_candles(), 0.0, Cfg,
                            book_snapshot=book_snapshot(flow_available=False))
    assert "book_flow" not in snap.individual


def test_book_flow_can_be_switched_off_without_touching_the_rest():
    class NoFlow(Cfg):
        ENABLE_BOOK_FLOW_SIGNAL = False

    snap = signals.evaluate(flat_candles(), 0.0, NoFlow,
                            book_snapshot=book_snapshot(flow_value=0.9))
    assert "book_flow" not in snap.individual
    assert "orderbook" in snap.individual


# --- summarize_book -------------------------------------------------------

def test_summarize_book_reports_unavailable_rather_than_zeros():
    assert signals.summarize_book(None) == {"available": False}
    assert signals.summarize_book({}) == {"available": False}
    assert signals.summarize_book({"available": False}) == {"available": False}


def test_summarize_book_separates_held_walls_from_all_walls():
    snapshot = book_snapshot()
    snapshot["walls"] = [
        {"side": "ask", "price": 101.0, "notional": 1e6, "held": True, "age_seconds": 40.0},
        {"side": "bid", "price": 99.0, "notional": 2e6, "held": False, "age_seconds": 2.0},
    ]
    summary = signals.summarize_book(snapshot)
    assert len(summary["walls"]) == 2
    assert [w["price"] for w in summary["held_walls"]] == [101.0]


def test_config_defaults_keep_the_persistence_gate_switched_on():
    """A regression guard: setting BOOK_PERSISTENCE_SECONDS to 0 silently
    reverts the signal to the old single-snapshot behavior."""
    assert base_cfg.BOOK_PERSISTENCE_SECONDS > 0
    assert base_cfg.BOOK_IMBALANCE_THRESHOLD > 0
