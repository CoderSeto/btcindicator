"""
Is there a scalping edge in Kalshi's KXBTC15M contracts? Reusable study code.

The edge study (kalshi_study.py, REVIEW_FINDINGS.md "Is there an edge?") asked
whether buying a side and HOLDING TO SETTLEMENT beats costs, and found nothing.
It never tested the other way to trade the same contracts: buy a side, then sell
it back a few minutes later at the new quote, before the window closes.

  Scalp question -- does any simple entry rule, followed by a short fixed hold
  and a sell at the then-standing bid, earn more than the round trip costs?

Scalping pays the spread and Kalshi's taker fee TWICE (in and out), where
holding pays them once (there is no settlement fee). That is why this is tested
before anything is built: the contract has to move in your favour by roughly
the whole round-trip cost before the first cent of profit.

This module holds the arithmetic, all tested on hand-built data.
`review/kalshi_scalp_backtest.py` runs it on the cached study data.

PRE-REGISTERED DESIGN. Fixed, and committed, before any scalp result had been
computed or looked at (no study_data existed in the environment it was written
in, so none could have been). Changing a constant after seeing results is
tuning on the held-out data; add a variant instead, count it, and label it as
an amendment below.
  - Data, split, bootstrap, costs: exactly kalshi_study's. Contracts sorted by
    close time, the last TEST_FRACTION (30%) held out; confidence intervals are
    a block bootstrap by UTC day, mean +/- z * SE; YES is bought at the ask and
    NO at 1 - bid; every order pays kalshi_study.order_fee (rounded up to the
    cent PER ORDER); a real book is kalshi_study.valid_book (two-sided and no
    wider than MAX_SPREAD). Fills at the quoted top of book with zero latency
    are assumed, in AND out -- candlesticks carry no depth, so the size really
    available may be smaller, and a scalp is most exposed to it.
  - Entry: one trade per contract per cell, at the quote standing
    ENTRY_HORIZONS_MIN minutes before close, only if that is a real book.
    A rule may use only information available at that quote: the contract's own
    mid and Coinbase's last trade at that moment, and the same two
    SIGNAL_LOOKBACK_MIN minutes earlier. Nothing after the entry is ever read
    by the entry rule.
  - Exit: HOLDS_MIN minutes after entry, sell at the quote standing then (YES
    at the bid, NO at 1 - ask), paying the fee on that price. If that is not a
    real book the position cannot be sold and is HELD TO SETTLEMENT instead (no
    exit fee, no second spread), and counted: forced holds are reported, never
    dropped, so an untradeable exit cannot flatter the result by disappearing.
  - Entry rules (RULES), each fixed and with fixed thresholds:
      contract_momentum  buy YES if the contract's mid rose by at least
                         CONTRACT_MOVE over the lookback, NO if it fell by that
                         much, otherwise no trade;
      contract_reversion the opposite side of contract_momentum;
      btc_momentum       buy YES if Coinbase BTC rose by at least BTC_MOVE_BP
                         basis points over the lookback, NO if it fell by that
                         much, otherwise no trade;
      btc_reversion      the opposite side of btc_momentum.
    The BTC rules are the "the contract lags the underlying" hypothesis; the
    contract rules are plain momentum and mean reversion in the contract's own
    price.
  - The family: RULES x ENTRY_HORIZONS_MIN x HOLDS_MIN x ORDER_SIZES = 48
    tests. This study's own family, not kalshi_study's 168: it asks a different
    question of different trades. Every interval uses one Bonferroni z over 48.
  - Selection: as in kalshi_study. A cell is "selected" on train only if its
    Bonferroni interval on mean profit per contract lies entirely above zero;
    only selected cells are then scored on the held-out set, with a second
    Bonferroni correction over the number selected. The held-out numbers for
    the other cells are printed but are descriptive, and nothing is claimed
    from them.
  - Descriptive, not counted: the cost floor (what the round trip costs on
    average and how often the contract moves that far) and the ORACLE ceiling
    (profit if the direction were known in advance and the trade skipped when
    it would lose). The oracle bounds every possible direction rule at this
    hold; if it is small, no entry rule can rescue the scalp.
  - Not tested, so not ruled out: exits that react inside a minute, take-profit
    or stop-loss rules, entries at other times, and anything needing quotes
    finer than the one-minute candlestick. Each would be a counted amendment.
"""
import numpy as np
import pandas as pd

import kalshi_study as ks

ENTRY_HORIZONS_MIN = (10, 5)
HOLDS_MIN = (1, 2, 3)
SIGNAL_LOOKBACK_MIN = 2
CONTRACT_MOVE = 0.02                # dollars of mid
BTC_MOVE_BP = 2.0
RULES = ("contract_momentum", "contract_reversion", "btc_momentum", "btc_reversion")
ORDER_SIZES = ks.ORDER_SIZES
SCALP_TESTS = len(RULES) * len(ENTRY_HORIZONS_MIN) * len(HOLDS_MIN) * len(ORDER_SIZES)

_EPS = 1e-9


# --- arithmetic ----------------------------------------------------------------

def scalp_pnl(side, bid0, ask0, bid1, ask1, outcome, contracts: int):
    """
    Profit per contract of buying at the entry quote (bid0, ask0) and selling at
    the exit quote (bid1, ask1). `side` is +1 for YES and -1 for NO, elementwise.

      YES: buy at ask0, sell at bid1.        NO: buy at 1 - bid0, sell at 1 - ask1.

    Both legs pay the per-order fee on their own price. Where the exit quote is
    not a real book the position is held to settlement instead: it pays the
    outcome and the entry fee only. Returns (pnl, forced_hold).
    """
    yes = np.asarray(side) == 1
    bid0, ask0, bid1, ask1 = (np.asarray(x, dtype=float) for x in (bid0, ask0, bid1, ask1))
    outcome = np.asarray(outcome, dtype=float)
    entry = np.where(yes, ask0, 1.0 - bid0)
    exit_ = np.where(yes, bid1, 1.0 - ask1)
    can_exit = ks.valid_book(bid1, ask1)
    payoff = np.where(yes, outcome, 1.0 - outcome)
    fee_in = ks.fee_per_contract(entry, contracts)
    with np.errstate(invalid="ignore"):
        fee_out = ks.fee_per_contract(exit_, contracts)
        sold = exit_ - entry - fee_in - fee_out
    held = payoff - entry - fee_in
    return np.where(can_exit, sold, held), ~can_exit


def rule_sides(frame: pd.DataFrame, rule: str) -> np.ndarray:
    """+1 buy YES, -1 buy NO, 0 no trade, for each row of `frame`, from its entry-time features only."""
    if rule.startswith("contract"):
        move, thresh = frame["contract_move"].to_numpy(dtype=float), CONTRACT_MOVE
    elif rule.startswith("btc"):
        move, thresh = frame["btc_move_bp"].to_numpy(dtype=float), BTC_MOVE_BP
    else:
        raise ValueError(f"unknown rule {rule!r}")
    with np.errstate(invalid="ignore"):
        momentum = np.where(move >= thresh - _EPS, 1, np.where(move <= -thresh + _EPS, -1, 0))
    if rule.endswith("momentum"):
        return momentum
    if rule.endswith("reversion"):
        return -momentum
    raise ValueError(f"unknown rule {rule!r}")


# --- building the trade frame ----------------------------------------------------

def _nan(x):
    return float("nan") if x is None else float(x)


def trade_frame(markets: list, candles_by_ticker: dict, grid=None,
                horizons=ENTRY_HORIZONS_MIN, holds=HOLDS_MIN) -> pd.DataFrame:
    """
    One row per (contract, entry horizon, hold): the entry quote, the exit
    quote, the outcome, and the two entry-rule features. `candles_by_ticker`
    maps ticker -> kalshi_study.parse_candles(...); `grid` is a
    kalshi_study.MinuteGrid of Coinbase 1-minute candles (None: no BTC features).

    The features use only quotes and Coinbase trades at or before the entry
    time. Rows with a missing quote or candle keep NaN, so the caller can count
    what it dropped. A quote is read only at its exact minute (quote_at), never
    the next or an earlier one.
    """
    lb = 60 * SIGNAL_LOOKBACK_MIN
    rows = []
    for m in markets:
        c = candles_by_ticker.get(m["ticker"], {})
        for h in horizons:
            t0 = m["close_ts"] - 60 * h
            q0 = ks.quote_at(c, t0) or (None, None, None)
            qp = ks.quote_at(c, t0 - lb) or (None, None, None)
            ok0 = bool(ks.valid_book(_nan(q0[0]), _nan(q0[1])))
            okp = bool(ks.valid_book(_nan(qp[0]), _nan(qp[1])))
            contract_move = ((q0[0] + q0[1]) / 2 - (qp[0] + qp[1]) / 2) if ok0 and okp else float("nan")
            btc_move = float("nan")
            if grid is not None:
                now_px, prev_px = grid.last_close_before(t0), grid.last_close_before(t0 - lb)
                if now_px and prev_px:
                    btc_move = (now_px / prev_px - 1) * 1e4
            for k in holds:
                q1 = ks.quote_at(c, t0 + 60 * k) or (None, None, None)
                rows.append({
                    "ticker": m["ticker"], "close_ts": m["close_ts"], "outcome": m["outcome"],
                    "horizon": h, "hold": k,
                    "bid0": _nan(q0[0]), "ask0": _nan(q0[1]), "bid1": _nan(q1[0]), "ask1": _nan(q1[1]),
                    "contract_move": contract_move, "btc_move_bp": btc_move,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["close_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


def with_split(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = np.where(out["close_ts"] < cutoff, "train", "test")
    return out


# --- the family ------------------------------------------------------------------

def cell_trades(frame: pd.DataFrame, rule: str, horizon: int, hold: int, contracts: int, split=None):
    """(rows traded, sides, pnl per contract, forced-hold flags) for one cell of the family."""
    d = frame[(frame["horizon"] == horizon) & (frame["hold"] == hold)]
    if split is not None:
        d = d[d["split"] == split]
    d = d[ks.valid_book(d["bid0"].to_numpy(), d["ask0"].to_numpy())]
    side = rule_sides(d, rule)
    take = side != 0
    d, side = d[take], side[take]
    pnl, forced = scalp_pnl(side, d["bid0"], d["ask0"], d["bid1"], d["ask1"], d["outcome"], contracts)
    return d, side, pnl, forced


def family_results(frame: pd.DataFrame, z: float) -> pd.DataFrame:
    """Every cell of the family, on train and on test, with a +/- z block-bootstrap interval."""
    rows = []
    for split in ("train", "test"):
        for rule in RULES:
            for h in ENTRY_HORIZONS_MIN:
                for k in HOLDS_MIN:
                    for n in ORDER_SIZES:
                        d, side, pnl, forced = cell_trades(frame, rule, h, k, n, split)
                        s = ks.block_bootstrap(pnl, d["day"].to_numpy())
                        lo, hi = ks.interval(s, z) if s["n"] > 1 else (np.nan, np.nan)
                        rows.append({
                            "split": split, "rule": rule, "horizon": h, "hold": k, "contracts": n,
                            "n": s["n"], "yes_share": float(np.mean(side == 1)) if len(side) else float("nan"),
                            "forced_rate": float(forced.mean()) if len(forced) else float("nan"),
                            "mean": s["mean"], "se": s["se"], "lo": lo, "hi": hi,
                            "mde": z * s["se"] if s["n"] > 1 else float("nan"), "n_eff": s["n_eff"],
                        })
    return pd.DataFrame(rows)


KEY = ["rule", "horizon", "hold", "contracts"]


def select_and_test(results: pd.DataFrame, z: float) -> dict:
    """
    The pre-registered selection rule. `selected`: train cells whose interval
    (already at the family z) lies entirely above zero. `test`: those cells on
    the held-out set at a second Bonferroni z over the number selected, with
    `confirmed` True only if that interval is also entirely above zero.
    """
    train = results[results["split"] == "train"]
    selected = train[(train["n"] > 1) & (train["lo"] > 0)]
    out = {"selected": selected, "z_test": None, "test": pd.DataFrame()}
    if selected.empty:
        return out
    z_t = ks.bonferroni_z(len(selected))
    test = results[results["split"] == "test"].merge(selected[KEY], on=KEY)
    test = test.assign(lo_t=test["mean"] - z_t * test["se"], hi_t=test["mean"] + z_t * test["se"])
    test["confirmed"] = test["lo_t"] > 0
    out["z_test"], out["test"] = z_t, test
    return out


# --- descriptive: what the round trip costs, and the ceiling --------------------------

def cost_floor(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per (entry horizon, hold), over every contract with a real book at BOTH
    ends (no rule, no direction):
      spread_in / spread_out   mean ask - bid at entry and at exit
      fee1 / fee100            mean fee per contract in plus out, each priced at its mid
      cost1 / cost100          the move in the mid a position needs just to break even:
                               (spread_in + spread_out) / 2 + fees
      move                     mean |change in mid| over the hold
      share_moves_cost1        how often |change in mid| exceeds cost1 (in EITHER direction)
      oracle1 / oracle100      profit per contract if the winning side were known in
                               advance and the trade skipped when both lose -- the
                               ceiling on any direction rule at this hold (it IS
                               look-ahead; it measures the opportunity, not a strategy)
    """
    rows = []
    for (h, k), g in frame.groupby(["horizon", "hold"]):
        g = g[ks.valid_book(g["bid0"].to_numpy(), g["ask0"].to_numpy())
              & ks.valid_book(g["bid1"].to_numpy(), g["ask1"].to_numpy())]
        if g.empty:
            continue
        mid0, mid1 = (g["bid0"] + g["ask0"]) / 2, (g["bid1"] + g["ask1"]) / 2
        spread_in, spread_out = g["ask0"] - g["bid0"], g["ask1"] - g["bid1"]
        row = {"horizon": h, "hold": k, "n": len(g),
               "spread_in": float(spread_in.mean()), "spread_out": float(spread_out.mean()),
               "move": float((mid1 - mid0).abs().mean())}
        for n in ORDER_SIZES:
            fees = ks.fee_per_contract(mid0.to_numpy(), n) + ks.fee_per_contract(mid1.to_numpy(), n)
            cost = (spread_in + spread_out) / 2 + fees
            yes_pnl, _ = scalp_pnl(np.ones(len(g)), g["bid0"], g["ask0"], g["bid1"], g["ask1"], g["outcome"], n)
            no_pnl, _ = scalp_pnl(-np.ones(len(g)), g["bid0"], g["ask0"], g["bid1"], g["ask1"], g["outcome"], n)
            row[f"fee{n}"] = float(np.mean(fees))
            row[f"cost{n}"] = float(cost.mean())
            row[f"share_moves_cost{n}"] = float(((mid1 - mid0).abs() > cost).mean())
            row[f"oracle{n}"] = float(np.maximum(np.maximum(yes_pnl, no_pnl), 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)
