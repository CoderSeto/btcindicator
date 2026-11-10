"""
Tests for kalshi_strategy_study.py: the four features (each against an
independent calculation or a hand-worked answer, and each for look-ahead), the
eight rules, the profit arithmetic (against kalshi_study's and
kalshi_scalp_study's own), the family and its selection rule. Hand-built data
throughout; nothing touches the network. Expected values are worked out in the
comments.
"""
import math

import numpy as np
import pandas as pd
import pytest

import kalshi
import kalshi_scalp_study as sc
import kalshi_strategy_study as st
import kalshi_study as ks

T0 = 1_788_000_000 - (1_788_000_000 % 86400)           # a UTC midnight
MINUTES_PER_YEAR = 365.25 * 24 * 60


def grid_of(closes, t_start=T0):
    """A MinuteGrid whose candle i starts at t_start + 60 i with open = high = low = close = closes[i]."""
    closes = np.asarray(closes, dtype=float)
    ts = t_start + 60 * np.arange(len(closes))
    return ks.MinuteGrid(pd.DataFrame({"ts": ts, "open": closes, "high": closes, "low": closes, "close": closes}))


def walk(n, seed=0, start=100_000.0, sd=3e-4):
    return start * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, sd, n)))


def market(i, outcome=1, strike=100_000.0, first_open=T0 + 3600):
    return {"ticker": f"M{i:04d}", "open_ts": first_open + 900 * i, "close_ts": first_open + 900 * (i + 1),
            "strike": strike, "outcome": outcome}


def frame_of(**cols):
    n = len(next(iter(cols.values())))
    base = {"bid0": [0.49] * n, "ask0": [0.51] * n, "residual": [np.nan] * n, "streak": [np.nan] * n,
            "vol_ratio": [np.nan] * n, "basis_dev_bp": [np.nan] * n}
    base.update(cols)
    return pd.DataFrame(base)


# ---------- the family ----------

def test_the_family_is_twenty_cells_at_two_order_sizes():
    assert len(st.CELLS) == 20 and len(set(st.CELLS)) == 20
    assert st.STRATEGY_TESTS == 40
    assert sum(1 for r, _, _ in st.CELLS if r.startswith("residual")) == 8
    for prefix in ("streak", "vol_regime", "basis"):
        assert sum(1 for r, _, _ in st.CELLS if r.startswith(prefix)) == 4
    # only the residual rules have a timed sale; everything else is held to settlement
    assert {r for r, _, k in st.CELLS if k != st.SETTLE} == {"residual_fade", "residual_follow"}
    assert st.FRAME_HORIZONS == (13, 10, 5)
    assert ks.bonferroni_z(st.STRATEGY_TESTS) == pytest.approx(3.2272, abs=1e-3)


def test_costs_split_bootstrap_and_selection_are_the_edge_studys_own_objects():
    assert st.ORDER_SIZES is ks.ORDER_SIZES
    assert st.select_and_test is sc.select_and_test and st.with_split is sc.with_split


# ---------- the rules ----------

@pytest.mark.parametrize("resid, fade", [(0.03, -1), (0.08, -1), (0.0299, 0), (0.0, 0), (-0.0299, 0), (-0.03, 1), (-0.2, 1)])
def test_residual_rules_fade_or_follow_a_three_cent_residual_inclusively(resid, fade):
    f = frame_of(residual=[resid])
    assert st.rule_sides(f, "residual_fade")[0] == fade          # the contract rose too far: buy NO
    assert st.rule_sides(f, "residual_follow")[0] == -fade


def test_a_float_residue_residual_of_three_cents_still_counts():
    assert 0.435 - 0.405 < 0.03                                  # the trap: 0.02999999999999997
    assert st.rule_sides(frame_of(residual=[0.435 - 0.405]), "residual_fade")[0] == -1
    assert st.rule_sides(frame_of(residual=[0.405 - 0.435]), "residual_fade")[0] == 1


@pytest.mark.parametrize("streak, follow", [(1.0, 1), (-1.0, -1), (0.0, 0), (np.nan, 0)])
def test_streak_rules_buy_with_or_against_the_run(streak, follow):
    f = frame_of(streak=[streak])
    assert st.rule_sides(f, "streak_follow")[0] == follow
    assert st.rule_sides(f, "streak_fade")[0] == -follow


@pytest.mark.parametrize("dev, fade", [(1.5, -1), (4.0, -1), (1.49, 0), (-1.49, 0), (-1.5, 1)])
def test_basis_rules_use_the_one_and_a_half_bp_threshold_inclusively(dev, fade):
    f = frame_of(basis_dev_bp=[dev])
    assert st.rule_sides(f, "basis_fade")[0] == fade             # strike set high against Coinbase: buy NO
    assert st.rule_sides(f, "basis_follow")[0] == -fade


@pytest.mark.parametrize("bid, ask, ratio, side", [
    (0.69, 0.71, 1.5, -1),        # YES favorite at 0.70, vol spiked: buy the underdog (NO)
    (0.69, 0.71, 2.4, -1),
    (0.29, 0.31, 1.5, 1),         # NO favorite, vol spiked: the underdog is YES
    (0.69, 0.71, 1 / 1.5, 1),     # calm: buy the favorite (YES)
    (0.29, 0.31, 0.5, -1),        # calm, NO favorite
    (0.69, 0.71, 1.49, 0),        # neither regime
    (0.69, 0.71, 0.67, 0),        # 0.67 > 1/1.5: not calm enough
    (0.69, 0.71, np.nan, 0),
    (0.54, 0.56, 2.0, 0),         # mid 0.55: no real favorite (under the 10c band)
    (0.59, 0.61, 2.0, -1),        # mid 0.60: exactly on the band's inner edge
    (0.89, 0.91, 2.0, -1),        # mid 0.90: exactly on its outer edge
    (0.90, 0.92, 2.0, 0),         # mid 0.91: too near certain
    (0.49, 0.51, 2.0, 0),         # mid 0.50: nobody is the favorite
])
def test_vol_regime_buys_the_underdog_after_a_spike_and_the_favorite_in_a_calm(bid, ask, ratio, side):
    f = frame_of(bid0=[bid], ask0=[ask], vol_ratio=[ratio])
    assert st.rule_sides(f, "vol_regime")[0] == side
    assert st.rule_sides(f, "vol_regime_inverse")[0] == -side


def test_every_rule_is_the_exact_mirror_of_its_pair_and_missing_features_never_trade():
    rng = np.random.default_rng(5)
    n = 400
    bid = np.round(rng.uniform(0.05, 0.9, n), 2)
    f = frame_of(bid0=bid, ask0=bid + 0.02, residual=rng.normal(0, 0.05, n), streak=rng.choice([1.0, -1.0, 0.0, np.nan], n),
                 vol_ratio=rng.uniform(0.3, 3, n), basis_dev_bp=rng.normal(0, 2, n))
    for a, b in st.PAIRS:
        sa, sb = st.rule_sides(f, a), st.rule_sides(f, b)
        assert (sa == -sb).all() and set(np.unique(sa)) == {-1, 0, 1}
    empty = frame_of(bid0=[0.69], ask0=[0.71])
    for rule in st.RULES:
        assert st.rule_sides(empty, rule)[0] == 0


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        st.rule_sides(frame_of(residual=[0.1]), "residual_astrology")


# ---------- the streak feature ----------

def test_streak_by_hand():
    outcomes = [1, 1, 0, 0, 0, 1, 1]
    s = st.streak_features([market(i, o) for i, o in enumerate(outcomes)])
    assert math.isnan(s["M0000"]) and math.isnan(s["M0001"])     # fewer than two contracts before them
    assert s["M0002"] == 1.0                                     # before it: 1, 1
    assert s["M0003"] == 0.0                                     # 0, 1
    assert s["M0004"] == -1.0                                    # 0, 0
    assert s["M0005"] == -1.0                                    # 0, 0
    assert s["M0006"] == 0.0                                     # 1, 0


def test_streak_never_reads_the_contract_itself_or_anything_later():
    a = [market(i, o) for i, o in enumerate([1, 1, 0, 1, 0])]
    b = [market(i, o) for i, o in enumerate([1, 1, 1, 0, 1])]    # contracts 2, 3 and 4 settle differently
    assert st.streak_features(a)["M0002"] == st.streak_features(b)["M0002"] == 1.0


def test_a_gap_in_the_chain_is_no_streak_not_a_guess():
    ms = [market(i, 1) for i in range(6)]
    del ms[2]                                                    # contract 2 is missing
    s = st.streak_features(ms)
    assert math.isnan(s["M0003"]) and math.isnan(s["M0004"])     # each needs contract 2
    assert s["M0005"] == 1.0                                     # 4 and 3 are both there


def test_two_contracts_sharing_a_close_break_the_chain():
    ms = [market(i, 1) for i in range(5)]
    ms.append({**market(1, 0), "ticker": "DUP"})                 # a second contract closing when contract 2 opens
    s = st.streak_features(ms)
    assert math.isnan(s["M0002"]) and math.isnan(s["M0003"])     # both chains pass through that close
    assert s["M0004"] == 1.0


# ---------- the basis feature ----------

def basis_markets(bps):
    return [market(i, strike=100_000.0 * (1 + b / 1e4)) for i, b in enumerate(bps)]


def flat_grid(n_contracts):
    return grid_of(np.full(60 + 15 * n_contracts + 5, 100_000.0))


def test_basis_deviation_by_hand():
    # Coinbase flat at 100,000, so a contract's basis is exactly the bp built into its strike.
    bps = [5.0] * 54 + [1.0] * 96 + [2.0]
    dev = st.basis_features(basis_markets(bps), flat_grid(len(bps)))
    assert math.isnan(dev["M0047"])                              # only 47 contracts before it
    assert dev["M0048"] == pytest.approx(0.0)                    # 5 - median(48 fives)
    assert dev["M0054"] == pytest.approx(-4.0)                   # 1 - median(54 fives)
    assert dev["M0100"] == pytest.approx(-4.0)                   # window 4..99: 50 fives, 46 ones -> median 5
    assert dev["M0104"] == pytest.approx(0.0)                    # window 8..103: 46 fives, 50 ones -> median 1
    assert dev["M0150"] == pytest.approx(1.0)                    # window 54..149 is all ones: the fives have aged out


def test_the_median_excludes_the_contract_itself_and_everything_after_it():
    base = [1.0] * 60
    changed = [1.0] * 50 + [9.0] * 10                            # contracts 50.. differ
    g = flat_grid(60)
    a, b = st.basis_features(basis_markets(base), g), st.basis_features(basis_markets(changed), g)
    assert a["M0049"] == b["M0049"] == pytest.approx(0.0)        # nothing at or after 50 reaches contract 49
    assert b["M0050"] == pytest.approx(8.0)                      # its own 9 is not in its median


def test_a_basis_beyond_twenty_bp_is_missing_for_itself_and_for_later_medians():
    bps = [1.0] * 47 + [25.0] + [3.0, 3.0]
    dev = st.basis_features(basis_markets(bps), flat_grid(len(bps)))
    assert math.isnan(dev["M0047"])                              # a data error, not a 25bp signal
    assert math.isnan(dev["M0048"])                              # 48 before it, but only 47 usable
    assert dev["M0049"] == pytest.approx(2.0)                    # usable: 47 ones and a three -> median 1


def test_a_strike_scaled_by_ten_thousand_is_missing_not_a_huge_signal():
    ms = basis_markets([1.0] * 49)
    ms[48]["strike"] = 7.2437                                    # Kalshi's real error (REVIEW_FINDINGS)
    assert math.isnan(st.basis_features(ms, flat_grid(49))["M0048"])


def test_basis_uses_the_minute_before_open_and_no_later_candle():
    closes = np.full(60 + 15 * 50 + 5, 100_000.0)
    open_i = 60 + 15 * 49                                        # contract 49 opens at this minute index
    closes[open_i - 1] = 100_020.0                               # the minute BEFORE open: basis = 100010/100020 - 1 = -1.0bp
    ms = basis_markets([1.0] * 50)
    dev = st.basis_features(ms, grid_of(closes))["M0049"]
    assert dev == pytest.approx((100_010.0 / 100_020.0 - 1) * 1e4 - 1.0)
    closes[open_i:] = 150_000.0                                  # everything from the open on changes
    assert st.basis_features(ms, grid_of(closes))["M0049"] == pytest.approx(dev)


# ---------- the residual and vol-ratio features ----------

CLOSES = walk(200, seed=7)
ENTRY_I = 180                                                    # entry at the start of minute 180
T_ENTRY = T0 + 60 * ENTRY_I


def numpy_vol(closes):
    return float(np.std(np.diff(np.log(closes)), ddof=1) * math.sqrt(MINUTES_PER_YEAR))


def test_residual_matches_an_independent_calculation():
    g, strike, h = grid_of(CLOSES), float(CLOSES[176]) * 1.0002, 10
    vol = numpy_vol(CLOSES[59:180])                              # the 121 closes ending just before entry
    p_now = kalshi.settlement_probability(float(CLOSES[179]), strike, 10, vol)     # last trade before entry
    p_prev = kalshi.settlement_probability(float(CLOSES[176]), strike, 13, vol)    # 3 minutes earlier, 13 left
    got = st.residual_feature(g, strike, T_ENTRY, h, 0.60, 0.50)
    assert got == pytest.approx((0.60 - 0.50) - (p_now - p_prev), abs=1e-12)
    assert abs(p_now - p_prev) > 1e-4                            # the model term is really in there


def test_residual_is_zero_when_the_contract_moves_exactly_as_the_model_does():
    g, strike = grid_of(CLOSES), float(CLOSES[176])
    vol = ks.model_inputs(g, T_ENTRY)[1]
    p_now = st.model_probability(g, T_ENTRY, strike, 10, vol)
    p_prev = st.model_probability(g, T_ENTRY - 180, strike, 13, vol)
    assert st.residual_feature(g, strike, T_ENTRY, 10, p_now, p_prev) == pytest.approx(0.0, abs=1e-12)


def test_residual_reads_nothing_from_the_entry_minute_on():
    strike = float(CLOSES[176])
    before = st.residual_feature(grid_of(CLOSES), strike, T_ENTRY, 10, 0.6, 0.5)
    later = CLOSES.copy()
    later[ENTRY_I:] *= 1.01                                      # the candle starting at entry, and all after
    assert st.residual_feature(grid_of(later), strike, T_ENTRY, 10, 0.6, 0.5) == before
    earlier = CLOSES.copy()
    earlier[ENTRY_I - 1] *= 1.001                                # the last candle before entry IS an input
    assert st.residual_feature(grid_of(earlier), strike, T_ENTRY, 10, 0.6, 0.5) != before


def test_residual_is_missing_without_both_mids_a_vol_or_a_plausible_strike():
    g, strike = grid_of(CLOSES), float(CLOSES[176])
    assert math.isnan(st.residual_feature(g, strike, T_ENTRY, 10, np.nan, 0.5))
    assert math.isnan(st.residual_feature(g, strike, T_ENTRY, 10, 0.6, np.nan))
    assert math.isnan(st.residual_feature(g, strike / 1e4, T_ENTRY, 10, 0.6, 0.5))          # Kalshi's 10^-4 strike error
    assert math.isnan(st.residual_feature(g, strike, T0 + 60 * 50, 10, 0.6, 0.5))           # under 2 hours of history: no vol
    holed = CLOSES.copy()
    holed[176] = np.nan                                                                     # the earlier spot minute is missing
    assert math.isnan(st.residual_feature(grid_of(holed), strike, T_ENTRY, 10, 0.6, 0.5))


def test_vol_ratio_matches_numpy():
    g = grid_of(CLOSES)
    want = numpy_vol(CLOSES[164:180]) / numpy_vol(CLOSES[59:180])      # 15 returns over 120 returns, both ending at entry
    assert st.vol_ratio_feature(g, T_ENTRY) == pytest.approx(want, rel=1e-12)


def test_vol_ratio_sees_a_spike_and_a_calm():
    rng = np.random.default_rng(1)
    spike = 100_000.0 * np.exp(np.cumsum(np.concatenate([rng.normal(0, 2e-4, 165), rng.normal(0, 8e-4, 35)])))
    calm = 100_000.0 * np.exp(np.cumsum(np.concatenate([rng.normal(0, 8e-4, 165), rng.normal(0, 1e-4, 35)])))
    assert st.vol_ratio_feature(grid_of(spike), T_ENTRY) > st.VOL_RATIO
    assert st.vol_ratio_feature(grid_of(calm), T_ENTRY) < 1 / st.VOL_RATIO


def test_vol_ratio_reads_nothing_from_the_entry_minute_on_and_needs_the_whole_short_window():
    before = st.vol_ratio_feature(grid_of(CLOSES), T_ENTRY)
    later = CLOSES.copy()
    later[ENTRY_I:] *= 1.05
    assert st.vol_ratio_feature(grid_of(later), T_ENTRY) == before
    holed = CLOSES.copy()
    holed[170] = np.nan                                          # one minute of the 15 missing
    assert math.isnan(st.vol_ratio_feature(grid_of(holed), T_ENTRY))
    flat = CLOSES.copy()
    flat[164:180] = flat[164]                                    # the short window never moved: no vol to compare
    assert math.isnan(st.vol_ratio_feature(grid_of(flat), T_ENTRY))


# ---------- the profit arithmetic ----------

def pnl(side, hold, b0=0.43, a0=0.44, b1=0.50, a1=0.51, outcome=1, n=1):
    p, f = st.trade_pnl([side], [b0], [a0], [b1], [a1], [outcome], hold, n)
    return float(p[0]), bool(f[0])


def test_held_to_settlement_pays_the_ask_and_one_fee():
    # YES at 0.44: fee 0.07 * 0.44 * 0.56 = 0.017248 -> 2c. Wins: 1 - 0.44 - 0.02. Loses: -0.46.
    assert pnl(+1, st.SETTLE, outcome=1) == (pytest.approx(0.54), False)
    assert pnl(+1, st.SETTLE, outcome=0) == (pytest.approx(-0.46), False)
    # NO at 1 - 0.43 = 0.57: fee 0.07 * 0.57 * 0.43 = 0.017157 -> 2c. Wins on NO: 1 - 0.57 - 0.02.
    assert pnl(-1, st.SETTLE, outcome=0) == (pytest.approx(0.41), False)
    assert pnl(-1, st.SETTLE, outcome=1) == (pytest.approx(-0.59), False)
    # 100 contracts: 0.07 * 100 * 0.44 * 0.56 = 1.7248 -> $1.73, 1.73c each
    assert pnl(+1, st.SETTLE, outcome=1, n=100)[0] == pytest.approx(1 - 0.44 - 0.0173)


def test_never_the_mid():
    # at the mid (0.435) a winning YES would make 0.565 before fees; the ask makes it 0.56, less the fee
    assert pnl(+1, st.SETTLE, outcome=1)[0] < 1 - 0.435 - 0.02 + 1e-12


@pytest.mark.parametrize("outcome", [0, 1])
@pytest.mark.parametrize("n", [1, 100])
def test_settlement_profit_is_the_edge_studys_and_the_timed_sale_is_the_scalp_studys(outcome, n):
    rng = np.random.default_rng(outcome * 7 + n)
    k = 200
    b0 = np.round(rng.uniform(0.05, 0.9, k), 2)
    a0 = b0 + np.round(rng.uniform(0.01, 0.05, k), 2)
    b1 = np.round(rng.uniform(0.05, 0.9, k), 2)
    a1 = b1 + np.round(rng.uniform(0.01, 0.3, k), 2)             # some exits wider than the 20c cap: forced holds
    side = rng.choice([1, -1], k)
    out = np.full(k, outcome)
    got, forced = st.trade_pnl(side, b0, a0, b1, a1, out, st.SETTLE, n)
    want = np.where(side == 1, ks.ev_buy_yes(out, a0, n), ks.ev_buy_no(out, b0, n))
    assert np.allclose(got, want) and not forced.any()
    got, forced = st.trade_pnl(side, b0, a0, b1, a1, out, st.EXIT_HOLD_MIN, n)
    want, want_forced = sc.scalp_pnl(side, b0, a0, b1, a1, out, n)
    assert np.allclose(got, want) and (forced == want_forced).all() and forced.any() and not forced.all()


def test_a_timed_sale_pays_two_fees_and_an_unsaleable_exit_is_held_and_flagged():
    # buy YES 0.44 (2c), sell at the 0.50 bid (2c): 0.50 - 0.44 - 0.04
    assert pnl(+1, st.EXIT_HOLD_MIN) == (pytest.approx(0.02), False)
    # no book at the exit: held to settlement, entry fee only, flagged -- never dropped
    assert pnl(+1, st.EXIT_HOLD_MIN, b1=np.nan, a1=np.nan, outcome=0) == (pytest.approx(-0.46), True)
    assert pnl(-1, st.EXIT_HOLD_MIN, b1=0.0, a1=0.55, outcome=0) == (pytest.approx(0.41), True)


def test_win_rate_and_break_even_by_hand_and_their_difference_is_the_mean_profit():
    d = pd.DataFrame({"bid0": [0.43, 0.43, 0.60], "ask0": [0.44, 0.44, 0.62], "outcome": [1, 0, 0]})
    side = np.array([1, 1, -1])
    # trades: YES at 0.44 won; YES at 0.44 lost; NO at 0.40 (fee 0.07*0.4*0.6 = 0.0168 -> 2c) won
    win, be = st.win_and_breakeven(d, side, 1)
    assert win == pytest.approx(2 / 3)
    assert be == pytest.approx((0.46 + 0.46 + 0.42) / 3)
    p, _ = st.trade_pnl(side, d.bid0, d.ask0, [np.nan] * 3, [np.nan] * 3, d.outcome, st.SETTLE, 1)
    assert float(np.mean(p)) == pytest.approx(win - be)
    assert all(math.isnan(x) for x in st.win_and_breakeven(d.iloc[:0], side[:0], 1))


# ---------- the trade frame ----------

def quote(bid, ask):
    return (bid, ask, (bid + ask) / 2)


def one_contract():
    """A contract closing at minute 195 of CLOSES' grid, with a quote at every minute it is open."""
    close = T0 + 60 * 195
    m = {"ticker": "A", "open_ts": close - 900, "close_ts": close, "strike": float(CLOSES[176]), "outcome": 1}
    q = {close - 60 * k: quote(round(0.30 + 0.02 * (15 - k), 2), round(0.32 + 0.02 * (15 - k), 2)) for k in range(15, 0, -1)}
    return m, q


def test_trade_frame_reads_each_quote_at_its_exact_minute():
    m, q = one_contract()
    f = st.trade_frame([m], {"A": q}, grid_of(CLOSES))
    assert list(f.horizon) == [13, 10, 5] and len(f) == 3
    r = f[f.horizon == 10].iloc[0]
    assert (r.bid0, r.ask0) == pytest.approx((0.40, 0.42))       # the quote 10 minutes before close
    assert (r.bid1, r.ask1) == pytest.approx((0.46, 0.48))       # the quote 7 minutes before close: 3 minutes on
    # residual: mid 0.41 now, 0.35 three minutes earlier, less the model's own change
    want = st.residual_feature(grid_of(CLOSES), m["strike"], m["close_ts"] - 600, 10, 0.41, 0.35)
    assert r.residual == pytest.approx(want) and not math.isnan(want)
    assert r.vol_ratio == pytest.approx(st.vol_ratio_feature(grid_of(CLOSES), m["close_ts"] - 600))
    assert math.isnan(r.streak) and math.isnan(r.basis_dev_bp)   # a lone contract has no history
    # 13 minutes before close the quote 3 minutes earlier predates the open: no residual
    assert math.isnan(f[f.horizon == 13].iloc[0].residual)


def test_no_rule_reads_the_exit_quote_or_the_outcome():
    m, q = one_contract()
    g = grid_of(CLOSES)
    a = st.with_split(st.trade_frame([m], {"A": q}, g), 0)
    q2 = {t: (quote(0.01, 0.99) if t > m["close_ts"] - 600 else v) for t, v in q.items()}       # every later quote changes
    b = st.with_split(st.trade_frame([{**m, "outcome": 0}], {"A": q2}, g), 0)
    ten_a, ten_b = a[a.horizon == 10], b[b.horizon == 10]
    for rule in st.RULES:
        assert (st.rule_sides(ten_a, rule) == st.rule_sides(ten_b, rule)).all()


def test_a_missing_or_placeholder_entry_quote_is_kept_as_a_row_but_never_traded():
    m, q = one_contract()
    q[m["close_ts"] - 600] = quote(0.001, 0.55)                  # a placeholder book at the 10-minute entry
    del q[m["close_ts"] - 300]                                   # no candle at the 5-minute entry
    f = st.with_split(st.trade_frame([m], {"A": q}, grid_of(CLOSES)), 0)
    assert len(f) == 3 and math.isnan(f[f.horizon == 5].iloc[0].bid0)
    forced_yes = f.assign(residual=-1.0)                         # a residual that would buy YES everywhere
    for h in (10, 5):
        d, side, p, forced = st.cell_trades(forced_yes, "residual_fade", h, st.SETTLE, 1)
        assert len(d) == 0 and len(p) == 0


def test_the_five_minute_residual_sale_is_at_the_quote_two_minutes_before_close():
    m, q = one_contract()
    f = st.with_split(st.trade_frame([m], {"A": q}, grid_of(CLOSES)), 0).assign(residual=-1.0)   # buy YES
    d, side, p, forced = st.cell_trades(f, "residual_fade", 5, st.EXIT_HOLD_MIN, 1)
    # entry 5 min out: 0.50/0.52 -> buy at 0.52 (fee 2c). Exit 2 min out: 0.56/0.58 -> sell at 0.56 (fee 2c).
    assert list(side) == [1] and float(p[0]) == pytest.approx(0.56 - 0.52 - 0.04) and not forced[0]


# ---------- the family, on synthetic data with a known answer ----------

def synthetic(n_days=30, per_day=16, persistence=0.5, seed=0, t0=T0, bid=0.49, ask=0.51):
    """
    Contiguous 15-minute contracts from 06:00 UTC each day, all quoted bid/ask at every
    minute. A contract settles the way the one before it did with probability
    `persistence` (0.5: a coin; above: a planted streak that streak_follow can trade).
    Coinbase is a random walk; each strike is the last Coinbase close before its open.
    Returns (markets, candles, coinbase DataFrame).
    """
    rng = np.random.default_rng(seed)
    n_min = 1440 * n_days
    closes = walk(n_min, seed=seed + 1, start=75_000.0, sd=2e-4)
    cb = pd.DataFrame({"ts": t0 + 60 * np.arange(n_min), "open": closes, "high": closes, "low": closes,
                       "close": closes, "volume": 1.0})
    markets, candles = [], {}
    prev = 1
    for d in range(n_days):
        for j in range(per_day):
            open_ts = t0 + 86400 * d + 6 * 3600 + 900 * j
            outcome = prev if rng.random() < persistence else 1 - prev
            prev = outcome
            tk = f"S{d:03d}-{j:02d}"
            strike = float(closes[(open_ts - t0) // 60 - 1]) * (1 + rng.normal(0, 2e-4))
            markets.append({"ticker": tk, "open_ts": open_ts, "close_ts": open_ts + 900, "strike": strike, "outcome": outcome})
            candles[tk] = {open_ts + 60 * k: quote(bid, ask) for k in range(1, 15)}
    return markets, candles, cb


def build(markets, candles, cb):
    f = st.trade_frame(markets, candles, ks.MinuteGrid(cb))
    return st.with_split(f, ks.split_cutoff([m["close_ts"] for m in markets]))


@pytest.fixture(scope="module")
def planted():
    frame = build(*synthetic(persistence=0.9, seed=3))
    z = ks.bonferroni_z(st.STRATEGY_TESTS)
    return frame, st.family_results(frame, z), z


def test_family_results_has_every_cell_on_both_splits(planted):
    _, res, _ = planted
    assert len(res) == 2 * st.STRATEGY_TESTS
    assert set(res.split) == {"train", "test"} and set(res.rule) == set(st.RULES)
    assert not res.duplicated(["split", "rule", "horizon", "hold", "contracts"]).any()


def test_a_planted_streak_is_selected_on_train_and_confirmed_held_out(planted):
    _, res, z = planted
    out = st.select_and_test(res, z)
    assert set(out["selected"].rule) == {"streak_follow"}
    assert len(out["selected"]) == 4 and out["test"]["confirmed"].all()         # 2 entries x 2 sizes
    r = res[(res.split == "train") & (res.rule == "streak_follow") & (res.horizon == 13) & (res.contracts == 1)].iloc[0]
    # after two same-way settlements the next repeats 90% of the time; buying at 0.51 costs 0.53 with the fee
    assert r.breakeven == pytest.approx(0.53) and r.win_rate == pytest.approx(0.9, abs=0.06)
    assert r["mean"] == pytest.approx(r.win_rate - r.breakeven)
    fade = res[(res.split == "train") & (res.rule == "streak_fade") & (res.horizon == 13) & (res.contracts == 1)].iloc[0]
    assert fade["mean"] < -0.3 and fade.hi < 0 and fade.n == r.n               # the mirror loses on the same trades


def test_the_planted_streak_shows_as_direction_information(planted):
    _, res, _ = planted
    info = st.direction_information(res)
    assert len(info) == st.STRATEGY_TESTS                        # one per mirrored pair of cells, per split
    s = info[(info.pair == "streak_follow vs streak_fade") & (info.split == "train") & (info.horizon == 13) & (info.contracts == 1)]
    # follow wins 0.9 - 0.53, fade 0.1 - 0.53: half the difference is 0.4
    assert float(s.information.iloc[0]) == pytest.approx(0.4, abs=0.06)


def test_direction_information_by_hand():
    rows = [{"split": "train", "rule": a, "horizon": 10, "hold": 0, "contracts": 1, "n": 10, "mean": m}
            for a, m in (("basis_fade", -0.01), ("basis_follow", -0.05))]
    info = st.direction_information(pd.DataFrame(rows))
    assert len(info) == 1 and float(info.information.iloc[0]) == pytest.approx(0.02)       # (-0.01 - -0.05) / 2


def test_coin_flip_outcomes_select_nothing():
    frame = build(*synthetic(persistence=0.5, seed=8))
    z = ks.bonferroni_z(st.STRATEGY_TESTS)
    res = st.family_results(frame, z)
    assert st.select_and_test(res, z)["selected"].empty
    # every cell that trades pays the 2c spread and the fee: the settlement cells' break-even is 53c on a coin
    settle = res[(res.hold == st.SETTLE) & (res.contracts == 1) & (res.n > 0)]
    assert np.allclose(settle.breakeven, 0.53)


def test_small_samples_are_flagged_and_an_empty_cell_does_not_crash(planted):
    frame, res, _ = planted
    assert res[res.n_eff < st.MIN_N_EFF].small.all() and not res[res.n_eff >= st.MIN_N_EFF].small.any()
    vol = res[res.rule == "vol_regime"]                          # every mid is 0.50: nobody is the favorite, no trades
    assert (vol.n == 0).all() and vol.small.all() and vol["mean"].isna().all()


def test_one_trade_per_contract_per_cell(planted):
    frame, _, _ = planted
    for rule, h, k in st.CELLS:
        d, side, p, forced = st.cell_trades(frame, rule, h, k, 1)
        assert not d.ticker.duplicated().any() and len(d) == len(side) == len(p) == len(forced)
