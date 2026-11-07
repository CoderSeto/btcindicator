"""Tests for weighting.py, against hand-computed expected values."""
from types import SimpleNamespace

import pytest

import weighting


def _cfg(**kw):
    base = dict(BACKTEST_MIN_SAMPLE=15, INDICATOR_WEIGHT_SCALE=2.0,
                INDICATOR_WEIGHT_MIN=0.2, INDICATOR_WEIGHT_MAX=2.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _snap(**votes):
    return SimpleNamespace(individual=votes)


# ---------- _weight_from_bucket ----------

@pytest.mark.parametrize("hits,total,expected", [
    (50, 100, 1.0),      # coin flip -> neutral
    (60, 100, 1.2),      # 1 + 2*0.1
    (40, 100, 0.8),      # 1 + 2*(-0.1)
    (75, 100, 1.5),
    (15, 15, 2.0),       # 1 + 2*0.5 = 2.0, exactly the cap
])
def test_weight_formula(hits, total, expected):
    assert weighting._weight_from_bucket({"hits": hits, "total": total}, _cfg()) == pytest.approx(expected)


def test_weight_clipped_to_bounds():
    cfg = _cfg(INDICATOR_WEIGHT_SCALE=10.0)
    assert weighting._weight_from_bucket({"hits": 90, "total": 100}, cfg) == 2.0   # raw 5.0
    assert weighting._weight_from_bucket({"hits": 10, "total": 100}, cfg) == 0.2   # raw -3.0


def test_weight_small_sample_and_missing_fall_back_to_one():
    cfg = _cfg()
    assert weighting._weight_from_bucket(None, cfg) == 1.0
    assert weighting._weight_from_bucket({}, cfg) == 1.0
    assert weighting._weight_from_bucket({"hits": 14, "total": 14}, cfg) == 1.0   # n < 15
    assert weighting._weight_from_bucket({"hits": 15, "total": 15}, cfg) == 2.0   # n == 15 counts


# ---------- compute_weights ----------

def test_compute_weights_reads_keyed_buckets():
    results = {"indicator_buckets": {
        "rsi_bullish": {"hits": 60, "total": 100},
        "rsi_bearish": {"hits": 40, "total": 100},
        "agg_delta_bullish": {"hits": 70, "total": 100},
    }}
    w = weighting.compute_weights(results, _cfg())
    assert w["rsi"] == {"bullish": pytest.approx(1.2), "bearish": pytest.approx(0.8)}
    assert w["agg_delta"] == {"bullish": pytest.approx(1.4), "bearish": 1.0}
    assert w["macd"] == {"bullish": 1.0, "bearish": 1.0}


def test_compute_weights_covers_expected_indicators_only():
    w = weighting.compute_weights({}, _cfg())
    assert set(w) == {"sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "agg_delta"}
    assert "orderbook" not in w and "book_flow" not in w
    assert all(v == {"bullish": 1.0, "bearish": 1.0} for v in w.values())


@pytest.mark.parametrize("results", [None, {}, {"indicator_buckets": {}}])
def test_compute_weights_tolerates_missing_results(results):
    assert weighting.compute_weights(results, _cfg())["rsi"]["bullish"] == 1.0


# ---------- weighted_conviction ----------

WEIGHTS = {"rsi": {"bullish": 1.5, "bearish": 0.5}, "macd": {"bullish": 1.0, "bearish": 1.25}}


def test_conviction_exact_values():
    out = weighting.weighted_conviction(_snap(rsi="bullish", macd="bearish"), WEIGHTS)
    assert out["available"] is True
    assert out["direction"] == "bullish"
    assert out["bullish_weight"] == 1.5 and out["bearish_weight"] == 1.25
    assert out["weighted_pct"] == pytest.approx(1.5 / 2.75)
    assert out["breakdown"] == {"rsi": {"vote": "bullish", "weight": 1.5},
                                "macd": {"vote": "bearish", "weight": 1.25}}


def test_conviction_uses_vote_direction_specific_weight():
    out = weighting.weighted_conviction(_snap(rsi="bearish"), WEIGHTS)
    assert out["bearish_weight"] == 0.5 and out["direction"] == "bearish"
    assert out["weighted_pct"] == 1.0


def test_conviction_unlisted_signals_weigh_one():
    out = weighting.weighted_conviction(_snap(orderbook="bullish", book_flow="bullish", rsi="bearish"), WEIGHTS)
    assert out["bullish_weight"] == 2.0
    assert out["bearish_weight"] == 0.5
    assert out["breakdown"]["orderbook"]["weight"] == 1.0


def test_conviction_ignores_neutral_votes():
    out = weighting.weighted_conviction(_snap(rsi="neutral", macd="bullish"), WEIGHTS)
    assert "rsi" not in out["breakdown"]
    assert out["bullish_weight"] == 1.0


def test_conviction_unavailable_when_nothing_directional():
    assert weighting.weighted_conviction(_snap(), WEIGHTS) == {"available": False}
    assert weighting.weighted_conviction(_snap(rsi="neutral", macd="neutral"), WEIGHTS) == {"available": False}


def test_conviction_tie_is_neutral_at_half():
    out = weighting.weighted_conviction(_snap(a="bullish", b="bearish"), {})
    assert out["direction"] == "neutral"
    assert out["weighted_pct"] == 0.5


def test_conviction_pct_always_in_half_to_one():
    import itertools
    for votes in itertools.product(["bullish", "bearish", "neutral"], repeat=3):
        snap = _snap(rsi=votes[0], macd=votes[1], volume=votes[2])
        out = weighting.weighted_conviction(snap, WEIGHTS)
        if out["available"]:
            assert 0.5 <= out["weighted_pct"] <= 1.0
