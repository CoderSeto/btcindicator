"""
Tests for the order-book metrics.

The house standard (CLAUDE.md) is that anything touching real math gets
cross-validated against an independent reference rather than just checked
for "runs without crashing". So the banded/decayed depth sums here are
checked against a deliberately naive brute-force implementation written
from the definition -- no sorting assumptions, no early exit -- across
randomized books, and the closed-form quantities are checked against
hand-computed values.
"""
import math
import random

import pytest

import config as cfg
import orderbook as ob


# --- an independent, deliberately naive reference implementation ----------

def brute_force_side_depth(levels, mid, band_bps, decay_bps, is_bid):
    """
    Straight from the definition: look at every level, no assumption that
    the list is sorted and no early break. This exists specifically to
    catch a bug in the real implementation's sorted-order early exit.
    """
    notional = 0.0
    weighted = 0.0
    count = 0
    for price, size in levels:
        if is_bid:
            distance = (mid - price) / mid * 10_000.0
        else:
            distance = (price - mid) / mid * 10_000.0
        distance = max(distance, 0.0)
        if distance > band_bps:
            continue
        value = price * size
        notional += value
        weighted += value * (1.0 if decay_bps <= 0 else math.exp(-distance / decay_bps))
        count += 1
    return notional, weighted, count


def random_book(rng, mid=100_000.0, n=60):
    bids = []
    asks = []
    for i in range(n):
        bid_price = mid * (1 - rng.uniform(0.00001, 0.02))
        ask_price = mid * (1 + rng.uniform(0.00001, 0.02))
        bids.append((bid_price, rng.uniform(0.01, 5.0)))
        asks.append((ask_price, rng.uniform(0.01, 5.0)))
    return ob.normalize_book({"bids": bids, "asks": asks})


# --- normalization --------------------------------------------------------

def test_normalize_levels_drops_zero_and_malformed():
    raw = [
        ["100.5", "1.5"],
        ["100.4", "0"],          # removed level
        ["0", "3"],              # nonsense price
        ["not-a-number", "1"],   # unparseable
        ["100.3"],               # truncated
        ["100.2", "2.0", "7"],   # Coinbase REST appends an order count
    ]
    assert ob.normalize_levels(raw) == [(100.5, 1.5), (100.2, 2.0)]


def test_normalize_book_sorts_both_sides():
    book = ob.normalize_book({
        "bids": [["99", "1"], ["101", "1"], ["100", "1"]],
        "asks": [["105", "1"], ["103", "1"], ["104", "1"]],
    })
    assert [p for p, _ in book["bids"]] == [101.0, 100.0, 99.0]
    assert [p for p, _ in book["asks"]] == [103.0, 104.0, 105.0]


# --- top of book ----------------------------------------------------------

def test_mid_and_spread_are_exact():
    book = ob.normalize_book({"bids": [["100", "1"]], "asks": [["101", "1"]]})
    assert ob.mid_price(book) == 100.5
    # (101 - 100) / 100.5 * 10_000
    assert ob.spread_bps(book) == pytest.approx(99.50248756, rel=1e-9)


def test_microprice_is_pulled_toward_the_thinner_side():
    # 2 BTC bid against 8 BTC offered: the bid clears first, so the true
    # clearing price sits near the bid, below the arithmetic mid.
    heavy_ask = ob.normalize_book({"bids": [["100", "2"]], "asks": [["101", "8"]]})
    assert ob.microprice(heavy_ask) == pytest.approx(100.2)
    assert ob.microprice_bps(heavy_ask) < 0

    # Mirror image: heavy bid, thin offer -> microprice above mid.
    heavy_bid = ob.normalize_book({"bids": [["100", "8"]], "asks": [["101", "2"]]})
    assert ob.microprice(heavy_bid) == pytest.approx(100.8)
    assert ob.microprice_bps(heavy_bid) > 0

    # Balanced book: microprice and mid coincide exactly.
    balanced = ob.normalize_book({"bids": [["100", "3"]], "asks": [["101", "3"]]})
    assert ob.microprice(balanced) == pytest.approx(ob.mid_price(balanced))
    assert ob.microprice_bps(balanced) == pytest.approx(0.0)


def test_one_sided_book_reports_no_data_rather_than_zero():
    empty_asks = ob.normalize_book({"bids": [["100", "1"]], "asks": []})
    assert ob.mid_price(empty_asks) is None
    assert ob.microprice(empty_asks) is None
    assert ob.spread_bps(empty_asks) is None
    assert ob.book_metrics(empty_asks, cfg)["available"] is False


# --- banded depth vs the brute-force reference ----------------------------

@pytest.mark.parametrize("band_bps", [1.0, 5.0, 25.0, 100.0])
@pytest.mark.parametrize("decay_bps", [0.0, 10.0])
def test_side_depth_matches_brute_force(band_bps, decay_bps):
    rng = random.Random(20260901)
    for _ in range(25):
        book = random_book(rng)
        mid = ob.mid_price(book)

        for side_key, is_bid in (("bids", True), ("asks", False)):
            got = ob.side_depth(book[side_key], mid, band_bps, decay_bps, is_bid=is_bid)
            want_notional, want_weighted, want_count = brute_force_side_depth(
                book[side_key], mid, band_bps, decay_bps, is_bid
            )
            assert got["notional"] == pytest.approx(want_notional, rel=1e-12)
            assert got["weighted"] == pytest.approx(want_weighted, rel=1e-12)
            assert got["levels"] == want_count


def test_band_edges_are_inclusive_and_exclusive_as_documented():
    mid = 100_000.0
    # Levels placed at exactly 10bps and just past it.
    book = ob.normalize_book({
        "bids": [[mid * (1 - 10 / 10_000), "1"], [mid * (1 - 10.0001 / 10_000), "1"]],
        "asks": [[mid * (1 + 10 / 10_000), "1"], [mid * (1 + 10.0001 / 10_000), "1"]],
    })
    inside = ob.band_imbalance(book, band_bps=10.0, decay_bps=0.0)
    assert inside["bid_levels"] == 1
    assert inside["ask_levels"] == 1

    wider = ob.band_imbalance(book, band_bps=11.0, decay_bps=0.0)
    assert wider["bid_levels"] == 2
    assert wider["ask_levels"] == 2


def test_decay_weights_the_touch_more_than_the_band_edge():
    """
    Equal money on both sides, but the bid's sits at the touch while the
    ask's sits 20bps out. Flat weighting can't tell those apart; decayed
    weighting -- the whole reason the band isn't a plain sum -- should
    read strongly bid.
    """
    book = ob.normalize_book({
        "bids": [["99999.5", "1"]],
        "asks": [["100000.5", "0.001"], ["100200", "1"]],
    })
    assert ob.mid_price(book) == pytest.approx(100_000.0)

    flat = ob.band_imbalance(book, band_bps=25.0, decay_bps=0.0)
    decayed = ob.band_imbalance(book, band_bps=25.0, decay_bps=10.0)

    # Flat: ~$100k of bids against ~$100k of asks -> near enough balanced.
    assert abs(flat["imbalance"]) < 0.01
    # Decayed: the ask's money is 20bps away (weight e^-2), the bid's is at
    # the touch, so the same dollars count for far less.
    assert decayed["imbalance"] > 0.5


def test_imbalance_is_signed_the_documented_way():
    mid = 100_000.0
    bid_heavy = ob.normalize_book({
        "bids": [[mid * 0.9999, "10"]], "asks": [[mid * 1.0001, "1"]],
    })
    assert ob.band_imbalance(bid_heavy, 25.0, 0.0)["imbalance"] > 0
    ask_heavy = ob.normalize_book({
        "bids": [[mid * 0.9999, "1"]], "asks": [[mid * 1.0001, "10"]],
    })
    assert ob.band_imbalance(ask_heavy, 25.0, 0.0)["imbalance"] < 0


def test_empty_band_reports_zero_imbalance_without_dividing_by_zero():
    mid = 100_000.0
    book = ob.normalize_book({
        "bids": [[mid * (1 - 0.01), "1"]], "asks": [[mid * (1 + 0.01), "1"]],
    })
    result = ob.band_imbalance(book, band_bps=1.0, decay_bps=10.0)
    assert result["imbalance"] == 0.0
    assert result["bid_levels"] == 0 and result["ask_levels"] == 0


# --- walls ----------------------------------------------------------------

def _uniform_book(mid=100_000.0, n=20, size=0.05):
    bids = [[mid - 1 - i, size] for i in range(n)]
    asks = [[mid + 1 + i, size] for i in range(n)]
    return bids, asks


def test_detect_walls_finds_the_outlier_and_only_the_outlier():
    bids, asks = _uniform_book()
    asks[5] = [asks[5][0], 0.05 * 12]  # one level 12x its neighbours
    book = ob.normalize_book({"bids": bids, "asks": asks})

    walls = ob.detect_walls(book, band_bps=100.0, multiple=4.0, min_notional=1_000)
    assert len(walls) == 1
    assert walls[0]["side"] == "ask"
    assert walls[0]["price"] == pytest.approx(100_006.0)
    # Not exactly 12x: the median it is measured against includes the
    # wall itself, and the levels are at slightly different prices.
    assert walls[0]["multiple"] == pytest.approx(12.0, rel=1e-3)


def test_wall_detection_uses_the_median_so_a_wall_cannot_hide_itself():
    """
    A wall large enough to drag the MEAN above its own size would suppress
    its own detection. The median is unmoved by one outlier, which is why
    it's the comparison point -- this test fails if that's ever changed.
    """
    bids, asks = _uniform_book(n=10, size=0.05)
    asks[3] = [asks[3][0], 0.05 * 5]  # 5x a normal level, in a book of 10
    book = ob.normalize_book({"bids": bids, "asks": asks})

    mean_notional = sum(p * s for p, s in book["asks"]) / len(book["asks"])
    wall_notional = book["asks"][3][0] * book["asks"][3][1]
    # Confirm the premise: against the mean, this wall is under a 4x multiple.
    assert wall_notional / mean_notional < 4.0

    walls = ob.detect_walls(book, band_bps=100.0, multiple=4.0, min_notional=1_000)
    assert [w["side"] for w in walls] == ["ask"]


def test_min_notional_floor_filters_out_small_relative_outliers():
    bids, asks = _uniform_book(size=0.00001)
    asks[5] = [asks[5][0], 0.00001 * 20]
    book = ob.normalize_book({"bids": bids, "asks": asks})
    # 20x its neighbours, but only a few dollars in absolute terms.
    assert ob.detect_walls(book, band_bps=100.0, multiple=4.0, min_notional=250_000) == []


def test_too_few_levels_yields_no_walls():
    book = ob.normalize_book({"bids": [["100", "1"], ["99", "50"]], "asks": [["101", "1"]]})
    assert ob.detect_walls(book, band_bps=1000.0, multiple=2.0, min_notional=1) == []


# --- notional in a price range --------------------------------------------

def test_notional_between_is_inclusive_and_order_independent():
    book = ob.normalize_book({
        "bids": [["100", "1"], ["99", "2"], ["98", "3"]],
        "asks": [["101", "1"], ["102", "2"], ["103", "3"]],
    })
    # asks at 101 and 102 -> 101*1 + 102*2 = 305
    assert ob.notional_between(book, 101, 102, "ask") == pytest.approx(305.0)
    assert ob.notional_between(book, 102, 101, "ask") == pytest.approx(305.0)
    # bids at 99 and 100 -> 99*2 + 100*1 = 298
    assert ob.notional_between(book, 99, 100, "bid") == pytest.approx(298.0)
    assert ob.notional_between(book, 200, 300, "ask") == 0.0


# --- cross-venue aggregation ----------------------------------------------

class _Cfg:
    """A minimal stand-in so aggregation tests don't depend on config.py's
    live values drifting."""
    BOOK_BAND_BPS_NEAR = 5.0
    BOOK_BAND_BPS = 25.0
    BOOK_BAND_BPS_FAR = 100.0
    BOOK_DECAY_BPS = 10.0
    BOOK_MAX_SPREAD_BPS = 8.0
    BOOK_WALL_MULTIPLE = 4.0
    BOOK_WALL_MIN_NOTIONAL = 250_000


def _venue(imbalance, bid_notional, ask_notional, spread=1.0, micro=0.0):
    return {
        "available": True, "imbalance": imbalance, "imbalance_near": imbalance,
        "imbalance_far": imbalance, "microprice_bps": micro, "spread_bps": spread,
        "bid_notional": bid_notional, "ask_notional": ask_notional, "walls": [],
    }


def test_aggregate_weights_venues_by_their_in_band_notional():
    per_venue = {
        "deep": _venue(0.60, 900_000, 100_000),    # weight 1_000_000
        "thin": _venue(-0.40, 50_000, 50_000),     # weight   100_000
    }
    result = ob.aggregate_metrics(per_venue, _Cfg)
    # (0.60 * 1e6 + -0.40 * 1e5) / 1.1e6
    assert result["imbalance"] == pytest.approx((0.60 * 1_000_000 - 0.40 * 100_000) / 1_100_000)
    assert result["venue_count"] == 2


def test_aggregate_drops_a_venue_with_a_blown_out_spread():
    per_venue = {
        "good": _venue(0.5, 500_000, 500_000, spread=1.0),
        "stale": _venue(-0.9, 500_000, 500_000, spread=40.0),
    }
    result = ob.aggregate_metrics(per_venue, _Cfg)
    assert result["venue_count"] == 1
    assert result["imbalance"] == pytest.approx(0.5)
    assert "stale" in result["excluded"]


def test_agreement_separates_broad_consensus_from_one_loud_venue():
    consensus = {
        "a": _venue(0.30, 500_000, 500_000),
        "b": _venue(0.30, 500_000, 500_000),
        "c": _venue(0.30, 500_000, 500_000),
    }
    assert ob.aggregate_metrics(consensus, _Cfg)["agreement"] == pytest.approx(1.0)

    # One venue dragging two flat-but-opposite ones along: same rough
    # direction, materially weaker agreement.
    lopsided = {
        "loud": _venue(0.95, 500_000, 500_000),
        "b": _venue(-0.05, 500_000, 500_000),
        "c": _venue(-0.05, 500_000, 500_000),
    }
    result = ob.aggregate_metrics(lopsided, _Cfg)
    assert result["imbalance"] > 0
    assert result["agreement"] == pytest.approx(1 / 3)


def test_aggregate_reports_unavailable_when_nothing_is_usable():
    result = ob.aggregate_metrics({"a": {"available": False, "reason": "one-sided"}}, _Cfg)
    assert result["available"] is False
    assert result["venue_count"] == 0


def test_aggregate_of_a_single_venue_is_that_venue():
    per_venue = {"only": _venue(0.42, 300_000, 200_000)}
    result = ob.aggregate_metrics(per_venue, _Cfg)
    assert result["imbalance"] == pytest.approx(0.42)
    assert result["agreement"] == pytest.approx(1.0)
    assert result["bid_notional"] == 300_000


# --- legacy path ----------------------------------------------------------

def test_legacy_compute_imbalance_is_unchanged():
    book = {"bids": [["100", "3"], ["99", "1"]], "asks": [["101", "1"], ["102", "1"]]}
    # base volume only: (4 - 2) / 6
    assert ob.compute_imbalance(book, depth=25) == pytest.approx(1 / 3)
    assert ob.compute_imbalance({"bids": [], "asks": []}, depth=25) == 0.0
