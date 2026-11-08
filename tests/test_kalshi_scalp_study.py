"""
Tests for kalshi_scalp_study.py: the round-trip arithmetic, the entry rules, the
trade frame (no look-ahead), the family and its selection rule, and the cost
floor / oracle. Hand-built data throughout; nothing touches the network. Expected
values are worked out in the comments, and the fee is the edge study's own
(kalshi_study.order_fee), so the arithmetic is cross-checked against it.
"""
import numpy as np
import pandas as pd
import pytest

import kalshi_scalp_study as sc
import kalshi_study as ks

T0 = 1_788_000_000 - (1_788_000_000 % 86400)           # a UTC midnight
CLOSE = T0 + 3600                                      # a contract closing an hour in


def pnl(side, b0, a0, b1, a1, outcome=1, n=1):
    p, f = sc.scalp_pnl([side], [b0], [a0], [b1], [a1], [outcome], n)
    return float(p[0]), bool(f[0])


# ---------- the arithmetic ----------

def test_yes_scalp_buys_the_ask_and_sells_the_bid_paying_a_fee_each_way():
    # buy at 0.44 (fee 0.0172 -> 2c), sell at 0.50 (0.0175 -> 2c): 0.50 - 0.44 - 0.02 - 0.02
    assert pnl(+1, 0.43, 0.44, 0.50, 0.51) == (pytest.approx(0.02), False)


def test_no_scalp_buys_one_minus_the_bid_and_sells_one_minus_the_ask():
    # buy NO at 1 - 0.43 = 0.57 (fee 2c); sell at 1 - 0.51 = 0.49 (fee 0.0175 -> 2c): 0.49 - 0.57 - 0.04
    assert pnl(-1, 0.43, 0.44, 0.50, 0.51) == (pytest.approx(-0.12), False)


def test_the_fee_rounds_up_per_order_not_per_contract():
    # 100 contracts: entry fee 0.07*100*0.44*0.56 = 1.7248 -> 1.73; exit 0.07*100*0.25 = 1.75
    got, _ = pnl(+1, 0.43, 0.44, 0.50, 0.51, n=100)
    assert got == pytest.approx(0.50 - 0.44 - 0.0173 - 0.0175)


def test_an_unsaleable_exit_is_held_to_settlement_and_flagged():
    # exit book one-sided (bid 0): hold. YES won: 1 - 0.44 - 0.02 entry fee. YES lost: -0.44 - 0.02.
    assert pnl(+1, 0.43, 0.44, 0.0, 0.55, outcome=1) == (pytest.approx(0.54), True)
    assert pnl(+1, 0.43, 0.44, 0.0, 0.55, outcome=0) == (pytest.approx(-0.46), True)
    assert pnl(-1, 0.43, 0.44, np.nan, np.nan, outcome=0) == (pytest.approx(1 - 0.57 - 0.02), True)   # no quote at all
    assert pnl(+1, 0.43, 0.44, 0.40, 0.70, outcome=1)[1] is True                                      # spread over the 20c cap


@pytest.mark.parametrize("outcome", [0, 1])
def test_a_held_position_matches_the_edge_studys_hold_to_settlement_profit(outcome):
    for n in (1, 100):
        yes, _ = pnl(+1, 0.43, 0.44, np.nan, np.nan, outcome, n)
        no, _ = pnl(-1, 0.43, 0.44, np.nan, np.nan, outcome, n)
        assert yes == pytest.approx(float(ks.ev_buy_yes(outcome, 0.44, n)))
        assert no == pytest.approx(float(ks.ev_buy_no(outcome, 0.43, n)))


def test_buying_no_is_buying_yes_on_the_mirrored_book():
    rng = np.random.default_rng(3)
    for _ in range(300):
        b0 = round(float(rng.uniform(0.05, 0.9)), 2)
        a0 = round(min(0.97, b0 + float(rng.uniform(0.01, 0.1))), 2)
        b1 = round(float(rng.uniform(0.05, 0.9)), 2)
        a1 = round(min(0.97, b1 + float(rng.uniform(0.01, 0.1))), 2)
        outcome, n = int(rng.integers(0, 2)), int(rng.choice([1, 5, 100]))
        # the NO book is (1 - ask, 1 - bid); buying NO on this book = buying YES on that one
        no_on_book = pnl(-1, b0, a0, b1, a1, outcome, n)
        yes_on_mirror = pnl(+1, 1 - a0, 1 - b0, 1 - a1, 1 - b1, 1 - outcome, n)
        assert no_on_book[0] == pytest.approx(yes_on_mirror[0], abs=1e-9)
        assert no_on_book[1] == yes_on_mirror[1]


def test_an_unchanged_book_always_loses_the_spread_and_two_fees():
    for bid, ask in [(0.43, 0.44), (0.10, 0.15), (0.80, 0.85), (0.50, 0.52)]:
        for side in (+1, -1):
            got, forced = pnl(side, bid, ask, bid, ask)
            assert not forced and got < -(ask - bid) + 1e-12          # at least the spread, plus the fees


# ---------- the entry rules ----------

def frame_of(**cols):
    n = len(next(iter(cols.values())))
    base = {"contract_move": [np.nan] * n, "btc_move_bp": [np.nan] * n}
    base.update(cols)
    return pd.DataFrame(base)


@pytest.mark.parametrize("move, mom", [(0.02, 1), (0.05, 1), (0.0199, 0), (0.0, 0), (-0.0199, 0), (-0.02, -1), (-0.1, -1)])
def test_contract_rules_use_the_two_cent_threshold_inclusively(move, mom):
    f = frame_of(contract_move=[move])
    assert sc.rule_sides(f, "contract_momentum")[0] == mom
    assert sc.rule_sides(f, "contract_reversion")[0] == -mom


@pytest.mark.parametrize("bp, mom", [(2.0, 1), (7.5, 1), (1.99, 0), (-1.99, 0), (-2.0, -1)])
def test_btc_rules_use_the_two_basis_point_threshold_inclusively(bp, mom):
    f = frame_of(btc_move_bp=[bp])
    assert sc.rule_sides(f, "btc_momentum")[0] == mom
    assert sc.rule_sides(f, "btc_reversion")[0] == -mom


def test_a_float_residue_difference_of_two_cents_still_counts_as_two_cents():
    assert 0.425 - 0.405 < 0.02                                       # the trap: 0.01999999999999999
    assert sc.rule_sides(frame_of(contract_move=[0.425 - 0.405]), "contract_momentum")[0] == 1


def test_a_missing_feature_means_no_trade_not_a_bullish_one():
    f = frame_of(contract_move=[np.nan], btc_move_bp=[np.nan])
    for rule in sc.RULES:
        assert sc.rule_sides(f, rule)[0] == 0


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        sc.rule_sides(frame_of(contract_move=[0.1]), "contract_astrology")


# ---------- the trade frame ----------

def quote(bid, ask):
    return (bid, ask, (bid + ask) / 2)


def market(ticker="T1", close=CLOSE, outcome=1):
    return {"ticker": ticker, "open_ts": close - 900, "close_ts": close, "strike": 75_000.0, "outcome": outcome}


def coinbase_grid(closes, start=T0):
    c = np.asarray(closes, dtype=float)
    ts = start + 60 * np.arange(len(c))
    return ks.MinuteGrid(pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c, "close": c, "volume": 1.0}))


def hand_candles():
    """Quotes at every minute from 15 to 1 minute before close: bid 0.40 + 0.01 per minute elapsed, spread 1c."""
    out = {}
    for m in range(15, 0, -1):
        t = CLOSE - 60 * m
        bid = round(0.40 + 0.01 * (15 - m), 2)
        out[t] = quote(bid, round(bid + 0.01, 2))
    return out


def test_one_row_per_contract_horizon_and_hold_with_exact_quotes():
    f = sc.trade_frame([market()], {"T1": hand_candles()})
    assert len(f) == len(sc.ENTRY_HORIZONS_MIN) * len(sc.HOLDS_MIN)
    r = f[(f.horizon == 10) & (f.hold == 2)].iloc[0]
    # 10 min before close = 5 minutes after the first quote: bid 0.45. Two minutes later: bid 0.47.
    assert (r.bid0, r.ask0, r.bid1, r.ask1) == pytest.approx((0.45, 0.46, 0.47, 0.48))
    # two minutes earlier the mid was 0.43 + 0.005... mid0 = 0.455, mid(t0-2m) = 0.435: a rise of 0.02
    assert r.contract_move == pytest.approx(0.02)
    assert np.isnan(r.btc_move_bp)                                     # no Coinbase grid given
    assert r.day == pd.to_datetime(CLOSE, unit="s", utc=True).strftime("%Y-%m-%d")


def test_the_btc_feature_is_the_move_over_the_lookback_ending_at_the_entry_minute():
    px = np.full(120, 75_000.0)
    t0 = CLOSE - 600
    i = (t0 - T0) // 60
    px[i - 1] = 75_015.0                      # the candle covering [t0-60, t0): the last trade before entry
    px[i - 3] = 75_000.0                      # the candle covering [t0-180, t0-120): the last trade before the lookback start
    f = sc.trade_frame([market()], {"T1": hand_candles()}, grid=coinbase_grid(px))
    r = f[(f.horizon == 10) & (f.hold == 1)].iloc[0]
    assert r.btc_move_bp == pytest.approx((75_015 / 75_000 - 1) * 1e4)


def test_nothing_after_entry_reaches_the_entry_features():
    base = hand_candles()
    shifted = dict(base)
    for t in list(shifted):
        if t > CLOSE - 600:                                            # every quote after the 10-minute entry
            shifted[t] = quote(0.01, 0.99)
    a = sc.trade_frame([market()], {"T1": base})
    b = sc.trade_frame([market()], {"T1": shifted})
    for col in ("bid0", "ask0", "contract_move"):
        x = a[a.horizon == 10][col].to_numpy()
        y = b[b.horizon == 10][col].to_numpy()
        assert np.allclose(x, y, equal_nan=True)
    # ...but the exit quotes do follow the later data
    assert not np.allclose(a[a.horizon == 10]["bid1"], b[b.horizon == 10]["bid1"])


def test_the_exit_is_the_quote_at_exactly_entry_plus_hold_never_a_neighbour():
    c = hand_candles()
    t1 = CLOSE - 600 + 120
    c[t1 + 60] = quote(0.99, 0.999)                                    # a different minute must not leak in
    del c[t1]                                                          # the exact minute has no candle
    r = sc.trade_frame([market()], {"T1": c})
    r = r[(r.horizon == 10) & (r.hold == 2)].iloc[0]
    assert np.isnan(r.bid1) and np.isnan(r.ask1)                       # missing stays missing, becomes a forced hold


def test_a_contract_with_no_candles_gives_nan_rows_not_missing_rows():
    f = sc.trade_frame([market()], {})
    assert len(f) == 6 and f[["bid0", "ask0", "bid1", "ask1"]].isna().all().all()


def test_the_signal_needs_a_real_book_at_both_ends_of_the_lookback():
    c = hand_candles()
    c[CLOSE - 600 - 120] = quote(0.0, 0.6)                             # a placeholder book two minutes before entry
    r = sc.trade_frame([market()], {"T1": c})
    assert np.isnan(r[(r.horizon == 10)]["contract_move"]).all()


# ---------- the family ----------

def synthetic_markets_and_candles(n_days=40, per_day=12, momentum=0.0, seed=0, t0=T0):
    """
    Random-walk contract quotes (1c half-spread). `momentum` > 0 makes each minute's
    change continue the previous one (a planted, tradeable pattern); 0 is a pure walk.
    """
    rng = np.random.default_rng(seed)
    markets, candles = [], {}
    for d in range(n_days):
        for j in range(per_day):
            close = t0 + 86400 * d + 3600 + 900 * j
            mid, step = 0.5, 0.0
            q = {}
            for m in range(15, 0, -1):
                step = momentum * step + rng.normal(0, 0.02)
                mid = float(np.clip(mid + step, 0.12, 0.88))
                bid = round(mid - 0.01, 2)
                q[close - 60 * m] = quote(bid, round(bid + 0.02, 2))
            tk = f"M{d:03d}-{j:02d}"
            markets.append({"ticker": tk, "open_ts": close - 900, "close_ts": close, "strike": 75_000.0,
                            "outcome": int(rng.random() < mid)})
            candles[tk] = q
    return markets, candles


def build(markets, candles):
    f = sc.trade_frame(markets, candles)
    return sc.with_split(f, ks.split_cutoff([m["close_ts"] for m in markets]))


def test_the_family_has_forty_eight_cells_per_split():
    assert sc.SCALP_TESTS == 4 * 2 * 3 * 2 == 48
    markets, candles = synthetic_markets_and_candles(n_days=12)
    res = sc.family_results(build(markets, candles), z=3.0)
    assert len(res) == 96 and set(res.split) == {"train", "test"}


def test_a_pure_random_walk_loses_the_round_trip_cost_in_every_cell_and_selects_nothing():
    markets, candles = synthetic_markets_and_candles(n_days=40, momentum=0.0, seed=5)
    z = ks.bonferroni_z(sc.SCALP_TESTS)
    res = sc.family_results(build(markets, candles), z)
    train = res[(res.split == "train") & (res.contracts == 1) & (res.n > 50)]
    assert (train["mean"] < 0).all()                                   # every cell loses, as the cost floor says it must
    assert train["mean"].mean() < -0.03                                # roughly spread + two fees, not a rounding error
    out = sc.select_and_test(res, z)
    assert out["selected"].empty and out["test"].empty and out["z_test"] is None


def test_a_planted_momentum_pattern_is_found_on_train_and_confirmed_on_test():
    markets, candles = synthetic_markets_and_candles(n_days=60, momentum=0.9, seed=11)
    z = ks.bonferroni_z(sc.SCALP_TESTS)
    res = sc.family_results(build(markets, candles), z)
    out = sc.select_and_test(res, z)
    assert not out["selected"].empty
    assert set(out["selected"]["rule"]) == {"contract_momentum"}      # the planted rule, not its mirror image
    assert (out["selected"]["mean"] > 0).all()
    assert out["z_test"] == pytest.approx(ks.bonferroni_z(len(out["selected"])))
    assert out["test"]["confirmed"].any()
    mirror = res[(res.split == "train") & (res.rule == "contract_reversion") & (res.contracts == 1) & (res.n > 50)]
    assert (mirror["mean"] < 0).all()                                  # fading a trend loses


def test_selection_scores_only_the_selected_cells_on_test():
    res = pd.DataFrame([
        {"split": "train", "rule": "a", "horizon": 10, "hold": 1, "contracts": 1, "n": 50, "mean": 0.05, "se": 0.01, "lo": 0.02, "hi": 0.08},
        {"split": "train", "rule": "b", "horizon": 10, "hold": 1, "contracts": 1, "n": 50, "mean": 0.05, "se": 0.04, "lo": -0.07, "hi": 0.17},
        {"split": "test", "rule": "a", "horizon": 10, "hold": 1, "contracts": 1, "n": 20, "mean": 0.04, "se": 0.01, "lo": 0.0, "hi": 0.0},
        {"split": "test", "rule": "b", "horizon": 10, "hold": 1, "contracts": 1, "n": 20, "mean": 0.09, "se": 0.01, "lo": 0.0, "hi": 0.0},
    ])
    out = sc.select_and_test(res, z=3.0)
    assert list(out["selected"]["rule"]) == ["a"]                      # b's point estimate is higher, but its interval spans zero
    assert list(out["test"]["rule"]) == ["a"]
    assert out["z_test"] == pytest.approx(ks.bonferroni_z(1))
    assert bool(out["test"]["confirmed"].iloc[0]) is True              # 0.04 - 1.96 * 0.01 > 0


def test_a_cell_selected_on_train_is_not_confirmed_when_test_falls_back():
    res = pd.DataFrame([
        {"split": "train", "rule": "a", "horizon": 5, "hold": 2, "contracts": 1, "n": 50, "mean": 0.05, "se": 0.01, "lo": 0.02, "hi": 0.08},
        {"split": "test", "rule": "a", "horizon": 5, "hold": 2, "contracts": 1, "n": 20, "mean": 0.01, "se": 0.02, "lo": 0.0, "hi": 0.0},
    ])
    assert bool(sc.select_and_test(res, z=3.0)["test"]["confirmed"].iloc[0]) is False


# ---------- the cost floor and the oracle ----------

def test_cost_floor_by_hand_for_one_contract_that_moves_five_cents():
    # entry 0.43/0.45 (spread 2c, mid 0.44), exit 0.48/0.50 (spread 2c, mid 0.49). Fees at the mids:
    # fee(0.44) = 0.07*0.44*0.56 = 0.017248 -> 0.02 ; fee(0.49) = 0.07*0.49*0.51 = 0.017493 -> 0.02. cost = 0.02 + 0.04 = 0.06.
    f = pd.DataFrame([{"horizon": 10, "hold": 1, "bid0": 0.43, "ask0": 0.45, "bid1": 0.48, "ask1": 0.50, "outcome": 1}])
    r = sc.cost_floor(f).iloc[0]
    assert (r.spread_in, r.spread_out, r.move) == pytest.approx((0.02, 0.02, 0.05))
    assert r.fee1 == pytest.approx(0.04) and r.cost1 == pytest.approx(0.06)
    assert r.share_moves_cost1 == 0.0                                  # a 5c move does not cover a 6c round trip
    # oracle: YES scalp = 0.48 - 0.45 - 0.04 = -0.01; NO scalp = (1-0.50) - (1-0.43) - 0.04 = -0.11; skip -> 0
    assert r.oracle1 == 0.0


def test_the_oracle_takes_the_better_side_and_skips_losing_trades():
    f = pd.DataFrame([
        # a 10c rise: YES scalp = 0.53 - 0.45 - 0.04 = +0.04  (entry fee 0.02 at 0.45, exit fee 0.02 at 0.53)
        {"horizon": 10, "hold": 1, "bid0": 0.43, "ask0": 0.45, "bid1": 0.53, "ask1": 0.55, "outcome": 1},
        # no move: both sides lose, the oracle skips
        {"horizon": 10, "hold": 1, "bid0": 0.43, "ask0": 0.45, "bid1": 0.43, "ask1": 0.45, "outcome": 1},
    ])
    r = sc.cost_floor(f).iloc[0]
    assert r.n == 2 and r.oracle1 == pytest.approx(0.02)               # (0.04 + 0) / 2


def test_cost_floor_leaves_out_rows_that_cannot_be_exited():
    f = pd.DataFrame([
        {"horizon": 10, "hold": 1, "bid0": 0.43, "ask0": 0.45, "bid1": np.nan, "ask1": np.nan, "outcome": 1},
        {"horizon": 10, "hold": 1, "bid0": 0.43, "ask0": 0.45, "bid1": 0.48, "ask1": 0.50, "outcome": 1},
    ])
    assert sc.cost_floor(f).iloc[0].n == 1


def test_the_oracle_ceiling_on_a_random_walk_is_small_next_to_a_trending_one():
    flat = sc.cost_floor(build(*synthetic_markets_and_candles(n_days=30, momentum=0.0, seed=2)))
    trend = sc.cost_floor(build(*synthetic_markets_and_candles(n_days=30, momentum=0.9, seed=2)))
    # more movement means a higher ceiling; the oracle never goes negative
    assert (flat.oracle1 >= 0).all() and (trend.oracle1 >= 0).all()
    assert trend.oracle1.mean() > flat.oracle1.mean()
