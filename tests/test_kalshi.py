"""
Tests for kalshi.py's pure math.

Independent references (nothing here re-implements the code under test):
  - scipy.stats.norm.cdf / statistics.NormalDist for the normal CDF.
  - numpy.std(ddof=1) for realized volatility.
  - A Monte Carlo simulation of driftless GBM for the crossing probability,
    which checks the closed form against the process it claims to describe.
  - Black-Scholes N(d2) written out with scipy for the same inputs.
"""
import math
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import kalshi

MIN_PER_YEAR = 365.25 * 24 * 60


def _df(closes):
    return pd.DataFrame({"close": closes})


# ---------- normal CDF ----------

def test_norm_cdf_matches_statistics_normaldist():
    nd = NormalDist()
    for x in np.linspace(-8, 8, 161):
        assert kalshi._norm_cdf(float(x)) == pytest.approx(nd.cdf(float(x)), abs=1e-15)


def test_norm_cdf_matches_scipy_to_machine_precision():
    stats = pytest.importorskip("scipy.stats")
    xs = np.linspace(-10, 10, 401)
    ours = np.array([kalshi._norm_cdf(float(x)) for x in xs])
    assert np.max(np.abs(ours - stats.norm.cdf(xs))) < 1e-15


def test_norm_cdf_known_values():
    assert kalshi._norm_cdf(0.0) == 0.5
    assert kalshi._norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)


# ---------- crossing probability ----------

def test_crossing_matches_black_scholes_n_d2():
    stats = pytest.importorskip("scipy.stats")
    for price, strike, mins, vol in [
        (100_000, 100_500, 8, 0.6), (100_000, 99_000, 15, 0.4),
        (67_000, 67_000, 5, 0.8), (3_000, 3_010, 1, 1.2),
    ]:
        tau = mins / MIN_PER_YEAR
        d2 = (np.log(price / strike) - 0.5 * vol ** 2 * tau) / (vol * np.sqrt(tau))
        assert kalshi.crossing_probability(price, strike, mins, vol) == pytest.approx(
            stats.norm.cdf(d2), abs=1e-14)


def test_crossing_matches_monte_carlo_gbm():
    price, strike, mins, vol = 100_000.0, 100_150.0, 10.0, 0.7
    tau = mins / MIN_PER_YEAR
    rng = np.random.default_rng(12345)
    z = rng.standard_normal(2_000_000)
    terminal = price * np.exp(-0.5 * vol ** 2 * tau + vol * np.sqrt(tau) * z)
    mc = float(np.mean(terminal > strike))
    se = math.sqrt(mc * (1 - mc) / len(z))
    assert kalshi.crossing_probability(price, strike, mins, vol) == pytest.approx(mc, abs=5 * se)


def test_crossing_properties():
    p = kalshi.crossing_probability
    # Monotone: higher strike -> lower P(above); higher price -> higher.
    assert p(100, 101, 10, 0.5) < p(100, 100, 10, 0.5) < p(100, 99, 10, 0.5)
    assert p(101, 100, 10, 0.5) > p(100, 100, 10, 0.5)
    # Bounded.
    for args in [(100, 1, 10, 0.5), (1, 100, 10, 0.5), (100, 100, 1e-6, 5.0)]:
        assert 0.0 <= p(*args) <= 1.0
    # Far in/out of the money saturates.
    assert p(100_000, 50_000, 5, 0.5) == pytest.approx(1.0)
    assert p(100_000, 200_000, 5, 0.5) == pytest.approx(0.0)
    # Less time -> more decisive for an in-the-money contract.
    assert p(101, 100, 1, 0.5) > p(101, 100, 60, 0.5)
    # Slightly below 0.5 at the money (the -sigma^2*tau/2 Ito term).
    assert p(100, 100, 15, 0.5) < 0.5


def test_crossing_expired_is_decided_by_price():
    p = kalshi.crossing_probability
    assert p(101, 100, 0, 0.5) == 1.0
    assert p(99, 100, -3, 0.5) == 0.0
    # KXBTC15M is strike_type "greater_or_equal" (tests/fixtures/kalshi_open_markets.json):
    # finishing exactly on the strike settles YES, not a coin flip.
    assert p(100, 100, 0, 0.5) == 1.0
    # vol is irrelevant (and may be None) once time is up.
    assert p(101, 100, 0, None) == 1.0


def test_crossing_rejects_bad_input():
    with pytest.raises(ValueError):
        kalshi.crossing_probability(0, 100, 5, 0.5)
    with pytest.raises(ValueError):
        kalshi.crossing_probability(100, -1, 5, 0.5)
    with pytest.raises(ValueError):
        kalshi.crossing_probability(100, 100, 5, 0)
    with pytest.raises(ValueError):
        kalshi.crossing_probability(100, 100, 5, None)


# ---------- realized volatility ----------

def test_realized_vol_matches_numpy():
    rng = np.random.default_rng(7)
    closes = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.0005, 80)))
    lookback = 30
    out = kalshi.realized_volatility(_df(closes), lookback)
    window = closes[-(lookback + 1):]
    expected = np.std(np.diff(np.log(window)), ddof=1) * math.sqrt(MIN_PER_YEAR)
    assert out["annual_vol"] == pytest.approx(expected, rel=1e-12)
    assert out["samples_used"] == lookback
    assert out["samples_requested"] == lookback


def test_realized_vol_hand_computed():
    # Closes chosen so the log returns are exactly +r, -r, +r, -r.
    r = 0.001
    closes = [100.0]
    for s in (1, -1, 1, -1):
        closes.append(closes[-1] * math.exp(s * r))
    out = kalshi.realized_volatility(_df(closes), 4)
    # mean 0, sample variance = 4 r^2 / 3
    expected = math.sqrt(4 * r * r / 3) * math.sqrt(MIN_PER_YEAR)
    assert out["annual_vol"] == pytest.approx(expected, rel=1e-12)


def test_realized_vol_reports_short_history():
    out = kalshi.realized_volatility(_df([100, 101, 100.5, 101.2, 100.9]), 60)
    assert out["samples_used"] == 4
    assert out["samples_requested"] == 60


def test_realized_vol_unavailable_cases():
    assert kalshi.realized_volatility(_df([100, 101]), 10)["annual_vol"] is None
    flat = kalshi.realized_volatility(_df([100.0] * 10), 5)
    assert flat["annual_vol"] is None and "reason" in flat
    # Non-positive prices are skipped; too few valid observations left.
    bad = kalshi.realized_volatility(_df([100, 0, 0, 101]), 5)
    assert bad["annual_vol"] is None


# ---------- fair_value wrapper ----------

def test_fair_value_wires_inputs_together():
    rng = np.random.default_rng(3)
    closes = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.0004, 100)))
    df = _df(closes)
    out = kalshi.fair_value(df, strike=closes[-1] + 50, minutes_remaining=9, lookback_minutes=40)
    assert out["available"] is True
    vol = kalshi.realized_volatility(df, 40)["annual_vol"]
    assert out["annual_vol"] == pytest.approx(vol)
    assert out["current_price"] == closes[-1]
    assert out["p_yes"] == pytest.approx(
        kalshi.crossing_probability(closes[-1], closes[-1] + 50, 9, vol))
    assert out["p_yes"] + out["p_no"] == pytest.approx(1.0)
    tau = 9 / MIN_PER_YEAR
    assert out["price_std"] == pytest.approx(closes[-1] * vol * math.sqrt(tau))


def test_fair_value_unavailable():
    assert kalshi.fair_value(pd.DataFrame({"close": []}), 100, 5, 10)["available"] is False
    assert kalshi.fair_value(_df([100, 101, 102, 101]), -5, 5, 10)["available"] is False
    flat = kalshi.fair_value(_df([100.0] * 10), 100, 5, 5)
    assert flat["available"] is False and flat["reason"]


def test_fair_value_expired_has_zero_std():
    out = kalshi.fair_value(_df([100, 101, 100, 102, 101]), 100, 0, 4)
    assert out["p_yes"] == 1.0 and out["price_std"] == 0.0


# ---------- quarter-hour boundary ----------

@pytest.mark.parametrize("h,m,s,us,expected", [
    (12, 0, 0, 0, 15.0),        # exactly on a boundary -> a full window ahead
    (12, 14, 0, 0, 1.0),
    (12, 14, 30, 0, 0.5),
    (12, 15, 0, 0, 15.0),
    (12, 7, 30, 0, 7.5),
    (12, 59, 0, 0, 1.0),        # rolls over the hour
    (23, 59, 59, 500_000, 0.5 / 60),
    (12, 44, 59, 999_999, 1e-6 / 60),
])
def test_minutes_until_next_close(h, m, s, us, expected):
    now = datetime(2026, 1, 1, h, m, s, us, tzinfo=timezone.utc)
    assert kalshi.minutes_until_next_kalshi_close(now) == pytest.approx(expected, abs=1e-9)


def _averaged_settlement_mc(spot, strike, minutes, vol, n=100_000, seed=7):
    """
    Independent reference for KXBTC15M's real settlement rule: YES iff the simple average of the
    60 one-second index prints before close is >= strike (rules_primary in
    tests/fixtures/kalshi_open_markets.json). Driftless GBM simulated second by second.
    Only valid for minutes >= 1, where none of the averaging window has elapsed yet.
    """
    rng = np.random.default_rng(seed)
    mpy = 365.25 * 24 * 60
    dt = 1 / 60 / mpy
    pre = minutes / mpy - 60 * dt
    log_s = np.log(spot) - 0.5 * vol ** 2 * pre + vol * np.sqrt(pre) * rng.standard_normal(n)
    total = np.zeros(n)
    for _ in range(60):
        log_s += -0.5 * vol ** 2 * dt + vol * np.sqrt(dt) * rng.standard_normal(n)
        total += np.exp(log_s)
    p = float(((total / 60) >= strike).mean())
    return p, (p * (1 - p) / n) ** 0.5


@pytest.mark.parametrize("minutes", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("side", [+1, -1])
def test_settlement_averaging_makes_the_true_probability_more_extreme(minutes, side):
    # Pins the DIRECTION of the documented averaging limitation. Averaging the last 60s lowers the
    # terminal variance from sigma^2*T to about sigma^2*(T - 2/3 min), so the true P(YES) sits
    # further from 50% than the point-in-time N(d2) -- the model is too *timid*, not too extreme.
    # (Measured: at 1 min left and spot 5bp above strike, model 0.79 vs averaged ~0.92.)
    spot = 75_000.0
    strike = spot / (1 + side * 5e-4)                                       # spot 5bp above / below
    model = kalshi.crossing_probability(spot, strike, minutes, 0.45)
    true_p, se = _averaged_settlement_mc(spot, strike, minutes, 0.45)
    if side > 0:
        assert model > 0.5 and true_p > model + 5 * se
    else:
        assert model < 0.5 and true_p < model - 5 * se
