"""
Tests for the CSV logs' fixed schema.

This covers the failure described in CLAUDE.md's gotcha #1: a field that
only appears partway through a session (agg_delta once the trade feeds
warm up, book_flow once the depth feeds have enough history) used to widen
later rows past the header, and pd.read_csv() broke on the result in
production. The fix was a fixed column list; these tests pin it down, and
extend the same rule to what happens when the schema itself changes
between releases.
"""
import csv

import pandas as pd
import pytest

import config as cfg
import main
import signals


@pytest.fixture
def log_paths(tmp_path, monkeypatch):
    alert = tmp_path / "alerts_log.csv"
    signal = tmp_path / "signal_log.csv"
    monkeypatch.setattr(cfg, "ALERT_LOG_PATH", str(alert))
    monkeypatch.setattr(cfg, "SIGNAL_LOG_PATH", str(signal))
    monkeypatch.setattr(cfg, "LOG_ALL_SIGNALS", True)
    return alert, signal


COLD_SIGNALS = {
    "sma_cross": "bullish", "ema_cross": "bullish", "rsi": "neutral",
    "macd": "bearish", "bollinger": "neutral", "volume": "neutral",
    "orderbook": "neutral",
}

WARM_SIGNALS = dict(COLD_SIGNALS, book_flow="bullish", agg_delta="bullish")

WARM_BOOK = {
    "available": True, "imbalance": 0.42, "agreement": 0.83,
    "persistence_seconds": 37.5, "flow_value": 0.61,
}


def _snapshot(individual, book=None, imbalance=0.1):
    return signals.SignalSnapshot(price=100_000.0, individual=dict(individual),
                                  imbalance=imbalance, book=book or {})


def test_row_width_is_constant_when_signals_appear_mid_session(log_paths):
    """The exact scenario from gotcha #1, now with two late-arriving
    signals instead of one."""
    alert, _ = log_paths

    main.log_alert(_snapshot(COLD_SIGNALS))          # feeds not warmed up
    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK))  # feeds warmed up
    main.log_alert(_snapshot(COLD_SIGNALS))          # a feed dropped again

    with open(alert, newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 4  # header + 3
    widths = {len(row) for row in rows}
    assert len(widths) == 1, f"ragged CSV: row widths {widths}"

    # And the thing that actually broke last time: pandas can read it.
    frame = pd.read_csv(alert)
    assert len(frame) == 3


def test_missing_signals_are_blank_cells_not_dropped_columns(log_paths):
    alert, _ = log_paths
    main.log_alert(_snapshot(COLD_SIGNALS))
    frame = pd.read_csv(alert)

    assert list(frame.columns) == main.ALERT_LOG_HEADER
    assert pd.isna(frame.loc[0, "book_flow"])
    assert pd.isna(frame.loc[0, "agg_delta"])
    assert frame.loc[0, "sma_cross"] == "bullish"


def test_book_columns_are_blank_rather_than_zero_when_there_is_no_book(log_paths):
    """
    Zero is a real reading -- a perfectly balanced book -- so "no data"
    has to be distinguishable from it, or any later analysis of the log
    silently treats downtime as balance.
    """
    alert, _ = log_paths
    main.log_alert(_snapshot(COLD_SIGNALS, book={"available": False}))
    frame = pd.read_csv(alert)
    for column in main.BOOK_LOG_FIELDS:
        assert pd.isna(frame.loc[0, column])


def test_book_columns_are_written_when_the_book_is_available(log_paths):
    alert, _ = log_paths
    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK))
    frame = pd.read_csv(alert)
    assert frame.loc[0, "book_imbalance"] == pytest.approx(0.42)
    assert frame.loc[0, "book_agreement"] == pytest.approx(0.83)
    assert frame.loc[0, "book_persistence_s"] == pytest.approx(37.5)
    assert frame.loc[0, "book_flow_value"] == pytest.approx(0.61)


def test_signal_log_has_the_same_fixed_schema(log_paths):
    _, signal = log_paths
    main.log_signal_snapshot(_snapshot(COLD_SIGNALS))
    main.log_signal_snapshot(_snapshot(WARM_SIGNALS, WARM_BOOK))

    frame = pd.read_csv(signal)
    assert list(frame.columns) == main.SIGNAL_LOG_HEADER
    assert len(frame) == 2


def test_signal_log_stays_off_unless_enabled(log_paths, monkeypatch):
    _, signal = log_paths
    monkeypatch.setattr(cfg, "LOG_ALL_SIGNALS", False)
    main.log_signal_snapshot(_snapshot(WARM_SIGNALS, WARM_BOOK))
    assert not signal.exists()


# --- schema changes between releases -------------------------------------

def test_an_older_layout_is_rotated_aside_instead_of_being_appended_to(log_paths, capsys):
    """
    Appending wider rows under a narrower header is the same corruption
    gotcha #1 describes, just triggered by a release rather than by a feed
    warming up. Both files must stay internally consistent and readable.
    """
    alert, _ = log_paths
    legacy_header = ["timestamp", "price", "composite", "strength", "imbalance",
                      "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume",
                      "orderbook", "agg_delta"]
    with open(alert, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerow(["2026-01-01T00:00:00+00:00", 90_000.0, "bullish", 4, 0.3,
                          "bullish", "bullish", "neutral", "bullish", "neutral",
                          "neutral", "bullish", "bullish"])

    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK))

    # The new file carries only the new schema.
    new_frame = pd.read_csv(alert)
    assert list(new_frame.columns) == main.ALERT_LOG_HEADER
    assert len(new_frame) == 1

    # The old rows survive, still readable, under their own header.
    archived = list(alert.parent.glob("alerts_log.csv.*.bak"))
    assert len(archived) == 1
    old_frame = pd.read_csv(archived[0])
    assert list(old_frame.columns) == legacy_header
    assert len(old_frame) == 1

    assert "older column layout" in capsys.readouterr().out


def test_a_matching_header_is_appended_to_normally(log_paths):
    alert, _ = log_paths
    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK))
    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK))

    assert list(alert.parent.glob("*.bak")) == []
    assert len(pd.read_csv(alert)) == 2


def test_every_signal_name_the_evaluator_can_emit_is_in_the_fixed_schema():
    """
    A new signal added to signals.py without being added to
    ALL_SIGNAL_NAMES would silently vanish from both logs. This pins the
    two together.
    """
    emitted = {
        "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume",
        "orderbook", "book_flow", "agg_delta",
    }
    assert emitted == set(main.ALL_SIGNAL_NAMES)


def test_legacy_imbalance_is_blank_when_the_rest_snapshot_was_skipped(log_paths):
    """
    With the depth feeds up, the extra Coinbase REST book request is
    skipped, so there is no legacy reading to log. Writing 0.0 there would
    claim the book was balanced.
    """
    alert, _ = log_paths
    main.log_alert(_snapshot(WARM_SIGNALS, WARM_BOOK, imbalance=None))
    frame = pd.read_csv(alert)
    assert pd.isna(frame.loc[0, "imbalance"])
    # The aggregated reading is still there -- the two columns are independent.
    assert frame.loc[0, "book_imbalance"] == pytest.approx(0.42)
