"""
Tests for the Step 3 half of kalshi_study.py: the Coinbase minute cache, the
gap-aware MinuteGrid, the model inputs (no look-ahead), the basis
measurement, and the trade-toward-the-model simulation. All data here is
hand-built; nothing touches the network.
"""
import math

import numpy as np
import pandas as pd
import pytest

import kalshi
import kalshi_study as ks

T0 = 1_788_000_000 - (1_788_000_000 % 86400)          # a UTC midnight


def candles(closes, start=T0, drop=()):
    ts = start + 60 * np.arange(len(closes))
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame({"ts": ts, "open": c, "high": c + 2, "low": c - 2, "close": c, "volume": 1.0})
    return df[~df.index.isin(list(drop))].reset_index(drop=True)


def walk(n, seed=0, s0=75_000.0, step_vol=4e-4):
    rng = np.random.default_rng(seed)
    return s0 * np.exp(np.cumsum(rng.normal(0, step_vol, n)))


# --- MinuteGrid ------------------------------------------------------------------

def test_grid_reads_the_candle_before_t_and_keeps_missing_minutes_missing():
    g = ks.MinuteGrid(candles([100, 101, 102, 103], drop=[2]))
    assert g.last_close_before(T0 + 60) == 100           # candle [T0, T0+60)
    assert g.last_close_before(T0 + 120) == 101
    assert g.last_close_before(T0 + 180) is None         # minute 2 missing
    assert g.last_close_before(T0 + 240) == 103
    assert g.last_close_before(T0) is None               # before the grid
    assert g.minute_average_before(T0 + 120) == pytest.approx(101)   # (o+h+l+c)/4 = c here
    assert list(g.closes_before(T0 + 120, 2)) == [100, 101]
    assert g.closes_before(T0 + 120, 3) is None          # runs off the start


# --- model inputs --------------------------------------------------------------

def test_model_inputs_match_an_independent_realized_vol():
    px = walk(400, seed=1)
    g = ks.MinuteGrid(candles(px))
    t = T0 + 60 * 300
    spot, vol = ks.model_inputs(g, t)
    assert spot == px[299]
    window = px[299 - 120: 300]                          # 121 closes ending at the spot candle
    r = np.diff(np.log(window))
    assert vol == pytest.approx(r.std(ddof=1) * math.sqrt(365.25 * 24 * 60), rel=1e-12)


def test_model_inputs_use_nothing_from_after_quote_time():
    px = walk(400, seed=2)
    t = T0 + 60 * 300
    base = ks.model_inputs(ks.MinuteGrid(candles(px)), t)
    future = px.copy()
    future[300:] *= 1.05                                 # every candle starting at or after t
    assert ks.model_inputs(ks.MinuteGrid(candles(future)), t) == base


def test_model_inputs_refuse_a_thin_vol_window_or_a_missing_spot_minute():
    px = walk(400, seed=3)
    t = T0 + 60 * 300
    assert ks.model_inputs(ks.MinuteGrid(candles(px, drop=[299])), t) is None
    thin = list(range(180, 192))                         # 12 of the 121 window candles gone
    assert ks.model_inputs(ks.MinuteGrid(candles(px, drop=thin)), t) is None
    few = list(range(180, 185))                          # 5 gone: carried forward, still priced
    assert ks.model_inputs(ks.MinuteGrid(candles(px, drop=few)), t) is not None


# --- model observations and the basis ------------------------------------------------

def _obs_row(open_ts, close_ts, strike, ev, h, outcome=1, bid=0.4, ask=0.42):
    return {"ticker": "X", "open_ts": open_ts, "close_ts": close_ts, "strike": strike,
            "expiration_value": ev, "outcome": outcome, "horizon": h,
            "quote_ts": close_ts - 60 * h, "bid": bid, "ask": ask}


def test_basis_is_measured_against_the_index_and_removed_by_the_adjusted_model():
    # Coinbase trades 5bp above an index that is otherwise identical.
    idx = walk(600, seed=4)
    cb = idx * 1.0005
    g = ks.MinuteGrid(candles(cb))
    open_ts, close_ts = T0 + 60 * 400, T0 + 60 * 415
    strike = idx[399]                                    # the index over the minute before open
    ev = idx[414]
    obs = pd.DataFrame([_obs_row(open_ts, close_ts, strike, ev, h=5)])
    out = ks.model_observations(obs, g).iloc[0]
    assert out.basis_open_bp == pytest.approx((1 / 1.0005 - 1) * 1e4)
    assert out.basis_close_bp == pytest.approx((1 / 1.0005 - 1) * 1e4)
    spot, vol = ks.model_inputs(g, close_ts - 300)
    assert out.p_model == pytest.approx(kalshi.settlement_probability(spot, strike, 5, vol))
    # The adjusted model prices the index's own level: spot / 1.0005.
    assert out.p_model_adj == pytest.approx(kalshi.settlement_probability(spot / 1.0005, strike, 5, vol))


def test_rows_the_model_cannot_price_are_nan_not_dropped():
    g = ks.MinuteGrid(candles(walk(200, seed=5)))
    obs = pd.DataFrame([_obs_row(T0 + 60 * 20, T0 + 60 * 35, 75_000, 75_010, h=5)])   # vol window off the start
    out = ks.model_observations(obs, g)
    assert len(out) == 1 and np.isnan(out.p_model.iloc[0]) and np.isnan(out.p_model_adj.iloc[0])


# --- trading toward the model --------------------------------------------------------

def test_model_trades_buy_the_side_with_edge_after_costs():
    side, pnl = ks.model_trades(prob=[0.70, 0.30, 0.50, 0.70],
                                bid=[0.55, 0.55, 0.49, 0.55], ask=[0.56, 0.56, 0.51, 0.56],
                                outcome=[1, 1, 1, 0])
    assert list(side) == [1, -1, 0, 1]
    assert pnl[0] == pytest.approx(ks.ev_buy_yes(1, 0.56))
    assert pnl[1] == pytest.approx(ks.ev_buy_no(1, 0.55))
    assert np.isnan(pnl[2])
    assert pnl[3] == pytest.approx(ks.ev_buy_yes(0, 0.56))


def test_a_margin_only_removes_trades():
    rng = np.random.default_rng(6)
    p = rng.uniform(0, 1, 2000)
    bid = rng.uniform(0.02, 0.9, 2000)
    ask = bid + 0.01
    y = rng.random(2000) < p
    s0, _ = ks.model_trades(p, bid, ask, y, 0.0)
    s5, _ = ks.model_trades(p, bid, ask, y, 0.05)
    assert ((s5 != 0) <= (s0 != 0)).all() and (s5[s5 != 0] == s0[s5 != 0]).all()


def test_an_edge_smaller_than_the_fee_does_not_trade():
    # p exceeds the ask by 1 cent, but the fee at 0.50 is 1.75 cents.
    side, _ = ks.model_trades([0.51], [0.49], [0.50], [1])
    assert side[0] == 0


# --- Coinbase cache --------------------------------------------------------------

def fake_range(product, gran, start, end):
    """Coinbase-shaped frame: one candle per minute in [start, end] inclusive (so chunks overlap)."""
    s, e = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    ts = np.arange(s - s % 60, e + 1, 60)
    return pd.DataFrame({"time": pd.to_datetime(ts, unit="s"), "low": 1.0, "high": 1.0,
                         "open": 1.0, "close": ts.astype(float), "volume": 1.0})


def test_a_day_is_fetched_in_chunks_without_gaps_or_duplicates():
    calls = []

    def f(*a):
        calls.append(a)
        return fake_range(*a)
    day = pd.to_datetime(T0, unit="s").strftime("%Y-%m-%d")
    df = ks.fetch_coinbase_day(day, f, pause=0)
    assert len(df) == 1440 and df.ts.is_unique
    assert (np.diff(df.ts) == 60).all() and df.ts.iloc[0] == T0
    assert (df.close == df.ts).all()
    assert len(calls) == math.ceil(1440 / ks.COINBASE_CHUNK_MIN)


def test_past_days_are_cached_and_today_is_always_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(ks.time, "sleep", lambda s: None)
    calls = []

    def f(*a):
        calls.append(a)
        return fake_range(*a)
    d0 = pd.to_datetime(T0, unit="s").strftime("%Y-%m-%d")
    d1 = pd.to_datetime(T0 + 86400, unit="s").strftime("%Y-%m-%d")
    df = ks.load_or_fetch_coinbase(T0, T0 + 86400 + 3600, str(tmp_path), f, today=d1)
    assert len(df) == 2880
    first = len(calls)
    ks.load_or_fetch_coinbase(T0, T0 + 86400 + 3600, str(tmp_path), f, today=d1)
    assert len(calls) - first == first // 2                # only "today" again
    assert (tmp_path / "coinbase_1m" / f"{d0}.csv").exists()
    assert not (tmp_path / "coinbase_1m" / f"{d1}.csv").exists()
