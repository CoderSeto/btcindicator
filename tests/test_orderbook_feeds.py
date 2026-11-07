"""
Tests for the level-2 depth feeds.

Every parser here is checked against the exchange's OWN documented example
payload, copied verbatim -- never a field name guessed from memory. That's
the house rule (CLAUDE.md) and it exists because the liquidation feeds
already produced one real bug from two exchanges using the same field
letter for opposite meanings.

Sources for the payloads below:
  - Coinbase Exchange level2/level2_batch channel docs (snapshot, l2update)
  - Binance spot WebSocket streams docs, "Diff. Depth Stream" and
    "How to manage a local order book correctly"
  - Bybit v5 WebSocket public orderbook docs (snapshot, delta)
"""
import json
import math
import threading
from types import SimpleNamespace

import pytest

import orderbook as ob
import orderbook_feeds as obf


class _Cfg:
    BOOK_BAND_BPS_NEAR = 5.0
    BOOK_BAND_BPS = 25.0
    BOOK_BAND_BPS_FAR = 100.0
    BOOK_DECAY_BPS = 10.0
    BOOK_MAX_SPREAD_BPS = 8.0
    BOOK_RETAIN_BPS = 500.0
    BOOK_SAMPLE_INTERVAL_SECONDS = 1.0
    BOOK_HISTORY_SECONDS = 900
    BOOK_IMBALANCE_THRESHOLD = 0.18
    BOOK_FLOW_WINDOW_SECONDS = 60.0
    BOOK_FLOW_THRESHOLD = 0.20
    BOOK_WALL_MULTIPLE = 4.0
    BOOK_WALL_MIN_NOTIONAL = 250_000
    BOOK_WALL_MIN_AGE_SECONDS = 15.0
    BOOK_WALL_MATCH_BPS = 2.0


# --- Coinbase -------------------------------------------------------------

COINBASE_SNAPSHOT = {
    "type": "snapshot",
    "product_id": "BTC-USD",
    "bids": [["10101.10", "0.45054140"]],
    "asks": [["10102.55", "0.57753524"]],
}

COINBASE_L2UPDATE = {
    "type": "l2update",
    "product_id": "BTC-USD",
    "time": "2019-08-14T20:42:27.265Z",
    "changes": [["buy", "10101.80000000", "0.162567"]],
}


def test_parse_coinbase_snapshot():
    parsed = obf.parse_coinbase_l2(COINBASE_SNAPSHOT)
    assert parsed["kind"] == "snapshot"
    assert parsed["bids"] == [["10101.10", "0.45054140"]]
    assert parsed["asks"] == [["10102.55", "0.57753524"]]


def test_parse_coinbase_update_maps_buy_sell_to_bid_ask():
    parsed = obf.parse_coinbase_l2(COINBASE_L2UPDATE)
    assert parsed["kind"] == "update"
    assert parsed["changes"] == [("bid", 10101.8, 0.162567)]

    sell_side = obf.parse_coinbase_l2({
        "type": "l2update", "product_id": "BTC-USD",
        "changes": [["sell", "10102.55", "0"]],
    })
    # "sell" is the resting side, not a taker direction -> ask.
    assert sell_side["changes"] == [("ask", 10102.55, 0.0)]


def test_parse_coinbase_ignores_other_message_types():
    assert obf.parse_coinbase_l2({"type": "subscriptions", "channels": []}) is None
    assert obf.parse_coinbase_l2({"type": "heartbeat"}) is None


# --- Binance --------------------------------------------------------------

BINANCE_DEPTH_UPDATE = {
    "e": "depthUpdate",
    "E": 1672515782136,
    "s": "BNBBTC",
    "U": 157,
    "u": 160,
    "b": [["0.0024", "10"]],
    "a": [["0.0026", "100"]],
}


def test_parse_binance_depth_update():
    parsed = obf.parse_binance_depth_update(BINANCE_DEPTH_UPDATE)
    assert parsed["first_id"] == 157
    assert parsed["last_id"] == 160
    assert parsed["bids"] == [["0.0024", "10"]]
    assert parsed["asks"] == [["0.0026", "100"]]


def test_parse_binance_ignores_non_depth_events():
    assert obf.parse_binance_depth_update({"e": "trade", "p": "1", "q": "1", "m": False}) is None
    assert obf.parse_binance_depth_update({"result": None, "id": 1}) is None


# --- Bybit ----------------------------------------------------------------

BYBIT_SNAPSHOT = {
    "topic": "orderbook.50.BTCUSDT",
    "type": "snapshot",
    "ts": 1672304484978,
    "data": {
        "s": "BTCUSDT",
        "b": [["16493.50", "0.006"], ["16493.00", "0.100"]],
        "a": [["16611.00", "0.029"], ["16612.00", "0.213"]],
        "u": 18521288,
        "seq": 7961638724,
    },
    "cts": 1672304484976,
}

BYBIT_DELTA = {
    "topic": "orderbook.50.BTCUSDT",
    "type": "delta",
    "ts": 1687940967466,
    "data": {
        "s": "BTCUSDT",
        "b": [["30247.20", "30.028"], ["30245.40", "0.224"]],
        "a": [],
        "u": 177400507,
        "seq": 7961638724,
    },
    "cts": 1687940967464,
}


def test_parse_bybit_snapshot_and_delta():
    snapshot = obf.parse_bybit_orderbook(BYBIT_SNAPSHOT)
    assert snapshot["kind"] == "snapshot"
    assert snapshot["update_id"] == 18521288
    assert snapshot["bids"][0] == ["16493.50", "0.006"]

    delta = obf.parse_bybit_orderbook(BYBIT_DELTA)
    assert delta["kind"] == "update"
    assert delta["update_id"] == 177400507
    assert delta["asks"] == []


def test_parse_bybit_treats_update_id_1_as_a_service_restart():
    """
    Bybit's docs: u == 1 means their orderbook service restarted and the
    local book must be overwritten, not patched.
    """
    restarted = dict(BYBIT_DELTA)
    restarted["data"] = dict(BYBIT_DELTA["data"], u=1)
    assert obf.parse_bybit_orderbook(restarted)["kind"] == "snapshot"


def test_parse_bybit_ignores_other_topics():
    assert obf.parse_bybit_orderbook({"topic": "publicTrade.BTCUSDT", "data": []}) is None
    assert obf.parse_bybit_orderbook({"op": "subscribe", "success": True}) is None


# --- LocalBook ------------------------------------------------------------

def test_local_book_zero_size_removes_a_level():
    book = obf.LocalBook()
    book.apply_snapshot([["100", "1"], ["99", "2"]], [["101", "1"]])
    assert book.materialize()["bids"] == [(100.0, 1.0), (99.0, 2.0)]

    book.apply_changes([("bid", 100.0, 0.0)])
    assert book.materialize()["bids"] == [(99.0, 2.0)]


def test_local_book_snapshot_replaces_rather_than_merges():
    book = obf.LocalBook()
    book.apply_snapshot([["100", "1"]], [["101", "1"]])
    book.apply_snapshot([["90", "1"]], [["91", "1"]])
    assert book.materialize()["bids"] == [(90.0, 1.0)]


def test_materialize_sorts_outward_from_the_touch():
    book = obf.LocalBook()
    book.apply_snapshot([["98", "1"], ["100", "1"], ["99", "1"]],
                        [["103", "1"], ["101", "1"], ["102", "1"]])
    materialized = book.materialize()
    assert [p for p, _ in materialized["bids"]] == [100.0, 99.0, 98.0]
    assert [p for p, _ in materialized["asks"]] == [101.0, 102.0, 103.0]


def test_materialize_retain_band_trims_far_levels_only():
    book = obf.LocalBook()
    # mid is 100_000, so a 500bps retain band keeps 95_000 - 105_000.
    book.apply_snapshot(
        [["99999", "1"], ["96000", "1"], ["50000", "1"]],
        [["100001", "1"], ["104000", "1"], ["120000", "1"]],
    )
    trimmed = book.materialize(retain_bps=500.0)
    assert [p for p, _ in trimmed["bids"]] == [99999.0, 96000.0]
    assert [p for p, _ in trimmed["asks"]] == [100001.0, 104000.0]

    # No band -> nothing dropped.
    full = book.materialize(retain_bps=None)
    assert len(full["bids"]) == 3 and len(full["asks"]) == 3


def test_materialize_of_a_one_sided_book_is_empty():
    book = obf.LocalBook()
    book.apply_snapshot([["100", "1"]], [])
    assert book.materialize() == {"bids": [], "asks": []}


# --- Binance sequence handling -------------------------------------------
#
# This is the fiddliest logic in the module and the easiest to get subtly
# wrong, so it's exercised directly rather than only through a live socket.

def _event(first_id, last_id, bids=None, asks=None):
    return {"first_id": first_id, "last_id": last_id,
            "bids": bids or [], "asks": asks or []}


def test_sequenced_update_rejects_events_before_the_book_is_seeded():
    agg = obf.BookAggregator(_Cfg)
    assert agg.apply_sequenced_update("binance_us", _event(10, 12)) == "not_seeded"


def test_sequenced_update_ignores_events_the_snapshot_already_covers():
    agg = obf.BookAggregator(_Cfg)
    agg.seed_sequenced_book("binance_us", [["100", "1"]], [["101", "1"]],
                            last_update_id=100, buffered=[])
    assert agg.apply_sequenced_update("binance_us", _event(90, 100)) == "stale"


def test_sequenced_update_applies_a_contiguous_event():
    agg = obf.BookAggregator(_Cfg)
    agg.seed_sequenced_book("binance_us", [["100", "1"]], [["101", "1"]],
                            last_update_id=100, buffered=[])
    result = agg.apply_sequenced_update(
        "binance_us", _event(101, 105, bids=[["100", "7"]]))
    assert result == "applied"
    assert agg._books["binance_us"].materialize()["bids"] == [(100.0, 7.0)]
    assert agg._books["binance_us"].update_id == 105


def test_sequenced_update_discards_the_book_on_a_gap():
    """
    Binance's rule: U > local_update_id + 1 means events were missed, and
    the only correct response is to throw the book away. Patching over a
    hole would leave a book that looks fine and is quietly wrong.
    """
    agg = obf.BookAggregator(_Cfg)
    agg.seed_sequenced_book("binance_us", [["100", "1"]], [["101", "1"]],
                            last_update_id=100, buffered=[])
    assert agg.apply_sequenced_update("binance_us", _event(150, 160)) == "gap"
    assert agg.is_ready("binance_us") is False


def test_seed_replays_buffered_events_and_drops_covered_ones():
    agg = obf.BookAggregator(_Cfg)
    buffered = [
        _event(90, 100, bids=[["100", "999"]]),   # covered by the snapshot, must be ignored
        _event(101, 104, bids=[["100", "5"]]),
        _event(105, 108, asks=[["101", "0"]]),    # removes the ask level
    ]
    assert agg.seed_sequenced_book(
        "binance_us", [["100", "1"]], [["101", "1"]],
        last_update_id=100, buffered=buffered) is True

    book = agg._books["binance_us"]
    assert book.bids == {100.0: 5.0}
    assert book.asks == {}
    assert book.update_id == 108


def test_seed_fails_when_the_buffer_does_not_meet_the_snapshot():
    agg = obf.BookAggregator(_Cfg)
    assert agg.seed_sequenced_book(
        "binance_us", [["100", "1"]], [["101", "1"]],
        last_update_id=100, buffered=[_event(200, 210)]) is False
    assert agg.is_ready("binance_us") is False


# --- persistence ----------------------------------------------------------

def _history(pairs):
    """pairs: [(ts, imbalance)] -> the sample shape the aggregator records."""
    return [
        {"ts": ts, "imbalance": imb, "imbalance_near": imb, "microprice_bps": 0.0,
         "bid_notional": 1_000.0, "ask_notional": 1_000.0, "agreement": 1.0,
         "venue_count": 1}
        for ts, imb in pairs
    ]


def test_persistence_measures_the_current_unbroken_run():
    agg = obf.BookAggregator(_Cfg)
    # Threshold is 0.18: below it until t=5, then held bullish through t=10.
    history = _history([(0, 0.0), (1, 0.05), (2, 0.10), (3, 0.0), (4, 0.10),
                        (5, 0.40), (6, 0.45), (7, 0.50), (8, 0.42), (9, 0.39), (10, 0.44)])
    result = agg._persistence(history)
    assert result["direction"] == "bullish"
    assert result["seconds"] == pytest.approx(5.0)
    assert result["mean_imbalance"] == pytest.approx(
        (0.40 + 0.45 + 0.50 + 0.42 + 0.39 + 0.44) / 6)


def test_persistence_resets_when_the_side_flips():
    agg = obf.BookAggregator(_Cfg)
    history = _history([(0, 0.5), (1, 0.5), (2, 0.5), (3, -0.5), (4, -0.5)])
    result = agg._persistence(history)
    assert result["direction"] == "bearish"
    assert result["seconds"] == pytest.approx(1.0)


def test_no_persistence_while_the_current_reading_is_inside_the_threshold():
    agg = obf.BookAggregator(_Cfg)
    # A long strong run, but it has just lapsed -- that's not a held lean.
    history = _history([(0, 0.9), (1, 0.9), (2, 0.9), (3, 0.02)])
    result = agg._persistence(history)
    assert result["direction"] is None
    assert result["seconds"] == 0.0


def test_persistence_of_an_empty_history_is_zero():
    agg = obf.BookAggregator(_Cfg)
    assert agg._persistence([])["seconds"] == 0.0


# --- flow -----------------------------------------------------------------

def _flow_history(points):
    """points: [(ts, bid_notional, ask_notional)]"""
    return [
        {"ts": ts, "imbalance": 0.0, "imbalance_near": 0.0, "microprice_bps": 0.0,
         "bid_notional": bid, "ask_notional": ask, "agreement": 1.0, "venue_count": 1}
        for ts, bid, ask in points
    ]


def test_flow_is_the_log_ratio_of_ratios():
    agg = obf.BookAggregator(_Cfg)
    # Over the 60s window: bids doubled, asks unchanged.
    history = _flow_history([(0.0, 100.0, 100.0), (60.0, 200.0, 100.0)])
    result = agg._flow(history)
    assert result["available"] is True
    assert result["value"] == pytest.approx(math.log(2.0))
    assert result["bid_change_pct"] == pytest.approx(100.0)
    assert result["ask_change_pct"] == pytest.approx(0.0)


def test_flow_is_antisymmetric():
    """Bids doubling reads exactly as strongly as asks doubling, opposite sign."""
    agg = obf.BookAggregator(_Cfg)
    up = agg._flow(_flow_history([(0.0, 100.0, 100.0), (60.0, 200.0, 100.0)]))
    down = agg._flow(_flow_history([(0.0, 100.0, 100.0), (60.0, 100.0, 200.0)]))
    assert up["value"] == pytest.approx(-down["value"])


def test_flow_cancels_a_book_wide_liquidity_swing():
    """Both sides halving is not a directional event and must read as zero."""
    agg = obf.BookAggregator(_Cfg)
    result = agg._flow(_flow_history([(0.0, 800.0, 400.0), (60.0, 400.0, 200.0)]))
    assert result["value"] == pytest.approx(0.0)


def test_flow_unavailable_until_the_window_is_covered():
    agg = obf.BookAggregator(_Cfg)
    result = agg._flow(_flow_history([(0.0, 100.0, 100.0), (10.0, 200.0, 100.0)]))
    assert result["available"] is False
    assert "history" in result["reason"]


def test_flow_unavailable_when_a_side_was_empty():
    agg = obf.BookAggregator(_Cfg)
    result = agg._flow(_flow_history([(0.0, 0.0, 100.0), (60.0, 100.0, 100.0)]))
    assert result["available"] is False


# --- wall ageing ----------------------------------------------------------

def _wall(price, notional=1_000_000.0, side="ask", venue="coinbase"):
    return {"side": side, "price": price, "size": notional / price,
            "notional": notional, "distance_bps": 10.0, "multiple": 8.0, "venue": venue}


def test_wall_age_accumulates_across_samples():
    agg = obf.BookAggregator(_Cfg)
    for t in range(1000, 1021):                                              # sampled every second
        agg._track_walls([_wall(100_100.0)], now=float(t))

    walls = agg._aged_walls(1020.0)
    assert len(walls) == 1
    assert walls[0]["age_seconds"] == pytest.approx(20.0)
    assert walls[0]["held"] is True  # past BOOK_WALL_MIN_AGE_SECONDS = 15


def test_a_wall_repriced_by_a_tick_keeps_its_age():
    """
    Matching is by price bucket, not exact price -- otherwise a wall that
    shuffles by a dollar would reset to age zero every sample and nothing
    would ever be reported as held.
    """
    agg = obf.BookAggregator(_Cfg)
    agg._track_walls([_wall(100_100.0)], now=1000.0)
    for t in range(1001, 1021):                                              # sampled every second
        agg._track_walls([_wall(100_100.5)], now=float(t))                  # 0.05bps away
    walls = agg._aged_walls(1020.0)
    assert len(walls) == 1
    assert walls[0]["age_seconds"] == pytest.approx(20.0)


def test_a_flickering_wall_never_becomes_held():
    agg = obf.BookAggregator(_Cfg)
    agg._track_walls([_wall(100_100.0)], now=1000.0)
    # Gone for well past the grace period, then a new one far away.
    agg._track_walls([], now=1010.0)
    agg._track_walls([_wall(100_900.0)], now=1011.0)

    walls = agg._aged_walls(1011.0)
    assert len(walls) == 1
    assert walls[0]["age_seconds"] == pytest.approx(0.0)
    assert walls[0]["held"] is False


def test_a_wall_missing_from_one_sample_keeps_its_age():
    """One sample's absence (partially filled, then replenished) is within
    the grace period and must not reset the clock."""
    agg = obf.BookAggregator(_Cfg)
    agg._track_walls([_wall(100_100.0)], now=1000.0)
    agg._track_walls([], now=1001.0)
    agg._track_walls([_wall(100_100.0)], now=1002.0)
    walls = agg._aged_walls(1002.0)
    assert len(walls) == 1
    assert walls[0]["age_seconds"] == pytest.approx(2.0)


# --- end-to-end sampling --------------------------------------------------

def test_sample_produces_an_aggregate_from_seeded_books():
    agg = obf.BookAggregator(_Cfg)
    agg.apply_snapshot("coinbase", [["99999", "10"]], [["100001", "1"]])
    agg.apply_snapshot("bybit", [["99998", "10"]], [["100002", "1"]])

    aggregate = agg.sample()
    assert aggregate["available"] is True
    assert aggregate["venue_count"] == 2
    assert aggregate["imbalance"] > 0  # both venues heavily bid

    snapshot = agg.snapshot()
    assert snapshot["available"] is True
    assert snapshot["samples"] == 1
    # One sample can't establish persistence or flow yet.
    assert snapshot["persistence"]["seconds"] == 0.0
    assert snapshot["flow"]["available"] is False


def test_snapshot_before_any_book_connects_is_unavailable_not_neutral():
    agg = obf.BookAggregator(_Cfg)
    snapshot = agg.snapshot()
    assert snapshot["available"] is False
    assert "reason" in snapshot


def test_a_dropped_feed_clears_the_cached_read_instead_of_freezing_it():
    """
    Every venue disconnecting at once is exactly when a stale-but-plausible
    book would be most misleading, so the cached aggregate has to be
    cleared rather than left in place.
    """
    agg = obf.BookAggregator(_Cfg)
    agg.apply_snapshot("coinbase", [["99999", "10"]], [["100001", "1"]])
    assert agg.sample()["available"] is True
    assert agg.snapshot()["available"] is True

    agg.reset_book("coinbase")
    assert agg.sample() is None
    assert agg.snapshot()["available"] is False


def test_a_stalled_sampler_is_reported_rather_than_served_as_live():
    agg = obf.BookAggregator(_Cfg)
    agg.apply_snapshot("coinbase", [["99999", "10"]], [["100001", "1"]])
    agg.sample()
    assert agg.snapshot()["available"] is True

    # Backdate the last sample well past the staleness allowance.
    agg._last_sample_ts -= 600.0
    stale = agg.snapshot()
    assert stale["available"] is False
    assert "stale" in stale["reason"]


# --- review regressions: gaps, venue churn, Binance resync (see REVIEW_FINDINGS.md) ---

def test_persistence_does_not_bridge_a_gap_in_the_history():
    """
    The history only gets a sample while some venue has a usable book, so an outage (every
    venue dropped, the machine asleep) leaves a hole. A lean seen before and after the hole has
    not been *observed* holding through it -- "time is evidence" only for time we watched.
    """
    agg = obf.BookAggregator(_Cfg)
    history = _history([(0, 0.5), (1, 0.5), (2, 0.5), (300, 0.5), (301, 0.5)])
    result = agg._persistence(history)
    assert result["direction"] == "bullish"
    assert result["seconds"] == pytest.approx(1.0)


def test_wall_age_does_not_bridge_a_sampling_gap():
    # _track_walls only runs when a sample is taken, so the grace check never saw the outage.
    agg = obf.BookAggregator(_Cfg)
    agg._track_walls([_wall(100_100.0)], now=1000.0)
    agg._track_walls([_wall(100_100.0)], now=1001.0)
    agg._track_walls([_wall(100_100.0)], now=1300.0)                        # 5 minutes unobserved
    walls = agg._aged_walls(1300.0)
    assert len(walls) == 1
    assert walls[0]["age_seconds"] == pytest.approx(0.0)
    assert walls[0]["held"] is False


def _clocked(monkeypatch, start=1_000_000.0):
    clock = {"t": start}
    monkeypatch.setattr(obf, "time", SimpleNamespace(time=lambda: clock["t"], sleep=lambda s: None))
    return clock


def test_flow_ignores_a_venue_leaving_the_aggregate(monkeypatch):
    """
    The flow numerator/denominator were the *summed* notional of whichever venues were usable at
    each end of the window. A bid-heavy venue dropping out (disconnect, or the spread gate during
    volatility) then reads as resting bids being pulled -- a book_flow vote nobody placed. Flow
    must compare like with like: only venues present at both ends.
    """
    clock = _clocked(monkeypatch)
    agg = obf.BookAggregator(_Cfg)
    agg.apply_snapshot("coinbase", [["99999", "10"]], [["100001", "10"]])  # balanced, unchanged
    agg.apply_snapshot("bybit", [["99998", "30"]], [["100002", "10"]])     # bid-heavy
    assert agg.sample()["venue_count"] == 2
    clock["t"] += 61.0
    agg.reset_book("bybit")                                                  # bybit drops out
    assert agg.sample()["venue_count"] == 1

    flow = agg.snapshot()["flow"]
    # The old code reported log(10/40) - log(10/20) = -0.69: a strong "bearish" flow.
    assert flow["available"] is True
    assert flow["value"] == pytest.approx(0.0, abs=1e-12)
    assert flow["bid_change_pct"] == pytest.approx(0.0, abs=1e-9)


def test_flow_is_unavailable_when_no_venue_spans_the_window(monkeypatch):
    clock = _clocked(monkeypatch)
    agg = obf.BookAggregator(_Cfg)
    agg.apply_snapshot("bybit", [["99998", "30"]], [["100002", "10"]])
    agg.sample()
    clock["t"] += 61.0
    agg.reset_book("bybit")
    agg.apply_snapshot("coinbase", [["99999", "10"]], [["100001", "10"]])
    agg.sample()
    flow = agg.snapshot()["flow"]
    assert flow["available"] is False
    assert "venue" in flow["reason"]
