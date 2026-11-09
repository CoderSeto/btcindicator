"""The edge hunt's own machinery (review/kalshi_edge_hunt.py): no network, hand-built data."""
import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "kalshi_edge_hunt", os.path.join(os.path.dirname(__file__), "..", "review", "kalshi_edge_hunt.py"))
eh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eh)


def test_ridge_logistic_recovers_the_sign_and_shrinks_toward_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 1))
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
    w_small = eh.ridge_logistic(x, y, lam=0.01)
    w_big = eh.ridge_logistic(x, y, lam=1e4)
    assert 1.6 < w_small[1] < 2.4
    assert 0 < w_big[1] < 0.2                       # heavy penalty shrinks the slope
    p = eh.predict(w_small, np.array([[-1.0], [1.0]]))
    assert p[0] < 0.5 < p[1]


def test_momentum_uses_only_candles_completed_before_the_quote(tmp_path):
    # Candles stamped by bucket START. Quote at 12:10:00 (ts=600 past the hour): the 12:09 candle
    # (ts 540, ends 600) is the last completed one; the 12:10 candle (ts 600) is still forming.
    base = 1_768_003_200                                  # 2026-01-10 00:00 UTC
    k = np.arange(-15, 20)                                # minutes relative to the hour
    ts = base + 60 * k
    close = 100.0 + k                                     # rises 1.0 per minute
    close[k == 10] = 500.0                                # the forming candle: a wild print that must not leak
    hi = close + 0.5
    lo = close - 0.5
    pd.DataFrame({"ts": ts, "open": close, "high": hi, "low": lo, "close": close, "volume": 1.0}).to_csv(
        tmp_path / "2026-01-10.csv", index=False)
    obs = pd.DataFrame({"day": ["2026-01-10"], "quote_ts": [base + 600]})
    m = eh._momentum(obs, cache_dir=str(tmp_path))
    # 5-min return: close[9] = 109 vs the candle completed 5 min earlier, close[4] = 104
    assert m.ret5.iloc[0] == pytest.approx(np.log(109 / 104))
    # 15-min range over completed candles k = -6..9: high 109.5, low 93.5 -- never the 500 print
    assert m.range15.iloc[0] == pytest.approx(np.log(109.5 / 93.5))


def test_trade_prob_enters_once_per_market_at_the_touch_plus_fee():
    df = pd.DataFrame({"close_ts": [1, 1, 2], "horizon": [10, 5, 10], "bid": [0.40, 0.40, 0.60],
                       "ask": [0.42, 0.42, 0.62], "outcome": [1, 1, 0]})
    side, pnl = eh.trade_prob(df, [0.60, 0.60, 0.20], margin=0.0)
    assert list(side) == [1, 0, -1]                       # market 1 once (earliest horizon), market 2 NO
    assert pnl[0] == pytest.approx(1 - 0.42 - 0.02)      # fee at 42c: 0.07*0.42*0.58 = 1.7c -> 2c
    assert pnl[2] == pytest.approx(1 - 0.40 - 0.02)      # NO at 1 - 0.60 = 40c, outcome 0 so NO pays 1
    _, worse = eh.trade_prob(df, [0.60, 0.60, 0.20], margin=0.0, slip=0.01)
    assert worse[0] == pytest.approx(pnl[0] - 0.01)


import pytest  # noqa: E402
