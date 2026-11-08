"""
kalshi_decision: the live decision and the walk-forward that picks its rule.
Expected values are worked by hand from Kalshi's fee table (fee = ceil(7% * p * (1 - p)) per
one-contract order, so 2c at 44c and at 60c) or planted in synthetic data with a known edge.
"""
import numpy as np
import pandas as pd
import pytest

import kalshi_decision as kd
import kalshi_study as ks

R1 = kd.Rule(margin=0.0, shrink=1.0, horizons=(5,))


def test_buy_yes_by_hand():
    # p .60, YES ask .44 (fee 2c): edge .60 - .44 - .02 = .14
    out = kd.decide_action(R1, 0.60, 0.40, 0.44, 5.0)
    assert out["action"] == "BUY YES" and out["side"] == "yes"
    assert out["price"] == pytest.approx(0.44) and out["fee"] == pytest.approx(0.02)
    assert out["edge"] == pytest.approx(0.14) and out["breakeven"] == pytest.approx(0.46)


def test_buy_no_by_hand():
    # p(YES) .20; NO costs 1 - .40 = .60 (fee 2c): edge .80 - .60 - .02 = .18
    out = kd.decide_action(R1, 0.20, 0.40, 0.44, 5.0)
    assert out["action"] == "BUY NO" and out["price"] == pytest.approx(0.60) and out["edge"] == pytest.approx(0.18)


def test_no_trade_cases_give_a_reason():
    assert kd.decide_action(R1, 0.42, 0.40, 0.44, 5.0)["action"] == "NO TRADE"          # inside the cost
    assert kd.decide_action(R1, 0.60, 0.40, 0.44, 8.0)["action"] == "NO TRADE"          # outside the horizon window
    for bid, ask in ((None, 0.5), (0.2, 0.6), (0.0, 0.5)):                              # none, wide spread, one-sided
        out = kd.decide_action(R1, 0.9, bid, ask, 5.0)
        assert out["action"] == "NO TRADE" and out["reasons"]
    out = kd.decide_action(None, 0.9, 0.4, 0.44, 5.0)
    assert out["action"] == "NO TRADE" and "backtest" in out["reasons"][0]
    assert kd.decide_action(R1, None, 0.4, 0.44, 5.0)["action"] == "NO TRADE"


def test_every_decision_carries_the_caveat():
    assert "NO TRADE is the default" in kd.decide_action(R1, 0.6, 0.4, 0.44, 5.0)["caveat"]


def test_margin_and_band_gate():
    assert kd.decide_action(kd.Rule(0.15, 1.0, (5,)), 0.60, 0.40, 0.44, 5.0)["action"] == "NO TRADE"   # edge .14 < .15
    assert kd.decide_action(kd.Rule(0.0, 1.0, (5,), band=0.05), 0.60, 0.40, 0.44, 5.0)["action"] == "NO TRADE"  # mid .42 off 50c
    assert kd.decide_action(kd.Rule(0.0, 1.0, (5,), band=0.25), 0.60, 0.40, 0.44, 5.0)["action"] == "BUY YES"


def test_shrink_zero_is_the_market_and_never_trades():
    rng = np.random.default_rng(0)
    mid = rng.uniform(0.2, 0.8, 500)
    side, _ = kd.rule_side(kd.Rule(0.0, 0.0, (5,)), mid - 0.02, mid + 0.02, rng.uniform(0, 1, 500), np.full(500, 5.0))
    assert not side.any()


def test_shrink_one_matches_the_studys_model_rule():
    rng = np.random.default_rng(1)
    mid = rng.uniform(0.15, 0.85, 4000)
    bid, ask = np.round(mid - 0.02, 2), np.round(mid + 0.02, 2)
    p = rng.uniform(0, 1, 4000)
    for margin in (0.0, 0.02, 0.05):
        side, _ = kd.rule_side(kd.Rule(margin, 1.0, (5,)), bid, ask, p, np.full(4000, 5.0))
        ref, _ = ks.model_trades(p, bid, ask, np.zeros(4000), margin, 1)
        assert (side == ref.astype(int)).all()


def test_tolerance_window_edges():
    f = lambda m: int(kd.rule_side(R1, [0.4], [0.44], [0.9], [m])[0][0])
    assert f(4.0) == 1 and f(6.0) == 1 and f(3.9) == 0 and f(6.1) == 0


def test_exit_rarely_fires_and_only_past_the_margin():
    # holding a YES the market prices at 42c mid: sale nets bid .40 - 2c fee = .38 < .42
    assert kd.decide_exit("yes", None, 0.40, 0.44, 5.0)["action"] == "HOLD TO SETTLE"
    # a model that says YES is only 10% likely: hold value .10 vs net sale .38 -> exit
    assert kd.decide_exit("yes", 0.10, 0.40, 0.44, 5.0, rule=R1)["action"] == "EXIT NOW"
    assert kd.decide_exit("yes", 0.60, 0.40, 0.44, 5.0, rule=R1)["action"] == "HOLD TO SETTLE"
    assert kd.decide_exit("no", None, None, None, 5.0)["action"] == "HOLD TO SETTLE"


# ---------- walk-forward ----------

def synthetic(informative: bool, days=60, per_day=120, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for i in range(per_day):
            true_p = rng.uniform(0.2, 0.8)
            mid = float(np.clip(true_p + rng.normal(0, 0.08), 0.1, 0.9)) if informative else true_p
            p_model = true_p if informative else float(np.clip(true_p + rng.normal(0, 0.08), 0.01, 0.99))
            outcome = int(rng.random() < true_p)
            ts = 1_700_000_000 + d * 86400 + i * 600
            for h in ks.HORIZONS_MIN:
                rows.append({"close_ts": ts, "day": f"d{d:03d}", "horizon": h, "bid": round(mid - 0.02, 2),
                             "ask": round(mid + 0.02, 2), "p_model": p_model, "outcome": outcome})
    return pd.DataFrame(rows)


SMALL_GRID = tuple(kd.Rule(m, k, h) for m in (0.0, 0.05) for k in (0.5, 1.0) for h in ((5,), (10, 5, 2)))


def test_walk_forward_finds_a_planted_edge_out_of_sample():
    res = kd.walk_forward(synthetic(True), grid=SMALL_GRID)
    assert res["rule"] is not None
    assert res["oos"]["n"] > 150 and res["oos"]["mean"] > 0.02 and res["oos"]["ci95"][0] > 0


def test_walk_forward_abstains_when_the_market_is_right():
    res = kd.walk_forward(synthetic(False), grid=SMALL_GRID)
    # market = truth, model = noise: any chosen rule loses out of sample; the bar should mostly keep it out
    assert res["oos"]["n"] == 0 or res["oos"]["mean"] < 0.01
    assert res["rule"] is None or res["oos"]["folds_traded"] < res["oos"]["folds"]


def test_a_market_is_entered_once_at_its_earliest_horizon():
    df = pd.DataFrame({"close_ts": [1, 1, 1, 2, 2], "horizon": [2, 10, 5, 5, 2], "bid": [0.4] * 5,
                       "ask": [0.44] * 5, "p_model": [0.9] * 5, "outcome": [1] * 5})
    side, pnl = kd.rule_pnl(kd.Rule(0.0, 1.0, (10, 5, 2)), df)
    assert side.tolist() == [0, 1, 0, 1, 0]                       # h=10 for market 1, h=5 for market 2
    assert np.isfinite(pnl).tolist() == [False, True, False, True, False]
    assert pnl[1] == pytest.approx(1 - 0.44 - 0.02)


def test_walk_forward_never_trains_on_its_test_fold():
    d = synthetic(True, days=30)
    for train_days, test_days in kd.fold_days(d["day"]):
        assert max(train_days) < min(test_days)


def test_save_and_load_roundtrip(tmp_path):
    res = {"rule": {"margin": 0.05, "shrink": 0.5, "horizons": [5, 2], "band": None}, "oos": {"n": 0}}
    p = str(tmp_path / "r.json")
    kd.save_result({**res, "table": "ignored"}, p, asof="2026-11-08")
    rule, back = kd.load_active(p)
    assert rule == kd.Rule(0.05, 0.5, (5, 2), None) and back["asof"] == "2026-11-08" and "table" not in back
    assert kd.load_active(str(tmp_path / "missing.json")) == (None, None)


def test_max_drawdown():
    assert kd.max_drawdown([1, -2, -1, 4, -3]) == 3
    assert kd.max_drawdown([]) == 0.0
