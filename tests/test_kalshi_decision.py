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


# --- entry quality (hand-worked) ---------------------------------------------

def test_entry_cost_is_half_spread_plus_the_rounded_up_fee():
    # bid 48 / ask 52: mid 50, half-spread 2c; fee at 52c = 0.07 * 0.52 * 0.48 = 1.7472c -> 2c per contract
    c = kd.entry_cost(0.48, 0.52, 1)
    assert c["mid"] == pytest.approx(0.50)
    assert c["half_spread"] == pytest.approx(0.02)
    assert c["fee_yes"] == pytest.approx(0.02) and c["fee_no"] == pytest.approx(0.02)
    assert c["cost_yes"] == pytest.approx(0.04) and c["cost_no"] == pytest.approx(0.04)
    assert c["breakeven_yes"] == pytest.approx(0.54) and c["breakeven_no"] == pytest.approx(0.54)
    # 100 contracts at 52c: fee 174.72c -> 175c per order -> 1.75c each
    assert kd.entry_cost(0.48, 0.52, 100)["fee_yes"] == pytest.approx(0.0175)


def test_entry_quality_grades_by_measured_spread_and_stale_print():
    good = kd.entry_quality(0.48, 0.52, 0.50, 10.0)
    assert good["grade"] == "GOOD" and good["cost_c"] == pytest.approx(4.0) and good["reasons"] == []
    exp = kd.entry_quality(0.45, 0.52, 0.50, 10.0)          # 7c spread
    assert exp["grade"] == "EXPENSIVE" and "spread 7c" in exp["reasons"][0]
    avoid = kd.entry_quality(0.40, 0.52, None, 10.0)        # 12c spread
    assert avoid["grade"] == "AVOID" and "spread 12c" in avoid["reasons"][0]
    stale = kd.entry_quality(0.48, 0.52, 0.60, 10.0)        # last 10c off the mid
    assert stale["grade"] == "EXPENSIVE" and "stale or placeholder" in stale["reasons"][0]
    assert kd.entry_quality(0.0, 1.0, None, 10.0)["grade"] == "AVOID"           # placeholder book
    assert kd.entry_quality(None, 0.5, None, 10.0)["reasons"] == ["no tradable two-sided book"]
    # 5c exactly is EXPENSIVE (>=), 4.9c is not; 10c exactly is AVOID
    assert kd.entry_quality(0.47, 0.52, None, 10.0)["grade"] == "EXPENSIVE"
    assert kd.entry_quality(0.42, 0.52, None, 10.0)["grade"] == "AVOID"


def test_entry_quality_names_the_payout_asymmetry_at_the_extremes_without_a_side_call():
    q = kd.entry_quality(0.94, 0.96, 0.95, 5.0)
    assert q["grade"] == "GOOD"
    assert q["reasons"] == ["payout asymmetry: YES at 96c wins 4c, risks 96c + 1c fee"]
    q = kd.entry_quality(0.04, 0.06, None, 5.0)
    assert q["reasons"] == ["payout asymmetry: NO at 96c wins 4c, risks 96c + 1c fee"]
    assert kd.entry_quality(0.50, 0.52, None, 5.0)["reasons"] == []


def test_mid_calibration_looks_up_the_frozen_band_and_horizon():
    cal = kd.mid_calibration(0.95, 5.2)
    assert (cal["n"], cal["n_eff"], cal["gap_c"], cal["lo_c"], cal["hi_c"]) == (4853, 6541, 0.89, 0.44, 1.34)
    assert cal["band"] == (0.90, 1.00) and cal["horizon"] == 5
    assert kd.mid_calibration(0.10, 10.0)["band"] == (0.0, 0.10)            # upper edge inclusive
    assert kd.mid_calibration(0.101, 10.0)["band"] == (0.10, 0.25)
    assert kd.mid_calibration(0.0, 2.0)["band"] == (0.0, 0.10)
    assert kd.mid_calibration(0.5, 7.5) is None                               # between windows
    line = kd.calibration_line(cal)
    assert "90-100c, 5 min" in line and "+0.89c" in line and "n_eff=6541" in line
    assert kd.calibration_line(None) == ""


def test_wait_hint_only_where_the_cache_showed_a_change():
    assert kd.wait_hint(0.01, 10.0) == ""                       # tight books did not change
    assert kd.wait_hint(0.03, 2.0) == ""                        # nothing later to wait for
    assert kd.wait_hint(0.03, 7.5) == ""                        # outside every window
    h = kd.wait_hint(0.03, 10.0)
    assert "narrower at 5 min in 74% of 5508 markets" in h and "97% were still tradable" in h
    h = kd.wait_hint(0.06, 5.0)
    assert "narrower at 2 min in 71% of 791 markets" in h and "68% were still tradable" in h
    assert "median -3c" in h
    assert kd.wait_hint(0.02, 10.0) != ""                       # 2c exactly qualifies


# --- entry timing, exit evidence, the one line (hand-worked) -------------------

def test_entry_timing_waits_only_for_a_wide_book_at_the_ten_minute_snapshot():
    wide = kd.entry_quality(0.45, 0.52, None, 10.0)                  # 7c spread, EXPENSIVE
    t = kd.entry_timing(wide, 10.0)
    assert t["timing"] == "WAIT"
    assert "fell 1.9c [1.7, 2.2] by 5 min in 773 such markets, 88% still tradable" in t["reason"]
    assert "~21c" in t["reason"]
    t = kd.entry_timing(wide, 5.0)                                     # 5 -> 2: only 66% still tradable
    assert t["timing"] == "ENTER NOW" and "saved only 1.7c in 643 such markets and 34% of books had closed" in t["reason"]
    tight = kd.entry_quality(0.50, 0.51, None, 10.0)
    t = kd.entry_timing(tight, 10.0)
    assert t["timing"] == "ENTER NOW" and "saved only 0.4c in 17341 such markets and 1% of books" in t["reason"]
    mid = kd.entry_quality(0.48, 0.51, None, 10.0)                   # 3c spread: saves 0.75c, under the 1c bar
    assert kd.entry_timing(mid, 10.0)["timing"] == "ENTER NOW"


def test_entry_timing_never_waits_at_the_extremes_or_the_last_snapshot_and_skips_avoid():
    ext = kd.entry_quality(0.90, 0.96, None, 10.0)                   # 6c spread but mid 93c
    t = kd.entry_timing(ext, 10.0)
    assert t["timing"] == "ENTER NOW" and "did not fall by 5 min (+0.1c) and 10% of books had closed" in t["reason"]
    assert "30% of books" in kd.entry_timing(ext, 5.0)["reason"]
    last = kd.entry_timing(kd.entry_quality(0.45, 0.52, None, 2.0), 2.0)
    assert last["timing"] == "ENTER NOW" and "nothing later" in last["reason"]
    skip = kd.entry_timing(kd.entry_quality(0.40, 0.52, None, 10.0), 10.0)
    assert skip["timing"] == "SKIP" and "spread 12c" in skip["reason"]
    assert kd.entry_timing(kd.entry_quality(None, 0.5, None, 10.0), 10.0) == {"timing": "SKIP", "reason": "no tradable two-sided book"}
    between = kd.entry_timing(kd.entry_quality(0.45, 0.52, None, 7.5), 7.5)   # uses the 10-min state, says so
    assert between["timing"] == "WAIT" and between["reason"].startswith("between snapshots")


def test_exit_carries_the_measured_sell_back_loss():
    ev = kd.exit_evidence("yes", 8.0)
    assert "lost 1.6c [1.2, 1.9] per contract against holding (n=18567, n_eff=18166)" in ev
    assert "lost 2.0c [1.6, 2.4]" in kd.exit_evidence("no", 4.0)
    r = kd.decide_exit("yes", 0.55, 0.50, 0.52, 4.0)
    assert r["action"] == "HOLD TO SETTLE" and r["evidence"] == kd.exit_evidence("yes", 4.0)
    assert kd.decide_exit("yes", 0.55, None, 0.52, 4.0)["evidence"] == ""


def test_decision_line_is_one_markdown_line_with_action_timing_grade_and_cost():
    d = kd.decide_action(None, 0.55, 0.45, 0.52, 10.0)
    q = kd.entry_quality(0.45, 0.52, None, 10.0)
    t = kd.entry_timing(q, 10.0)
    line = kd.decision_line(d, q, t)
    assert line == "\u26AA **NO TRADE** \u00b7 **WAIT** \u00b7 entry \U0001F7E1 **EXPENSIVE** (5.5\u00a2 to enter)"
    assert "\n" not in line
    held = kd.decision_line(d, q, t, {"action": "HOLD TO SETTLE"})
    assert held.endswith(" \u00b7 held: **HOLD TO SETTLE**")
