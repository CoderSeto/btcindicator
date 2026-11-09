"""
Trade decision for KXBTC15M, and the walk-forward backtest that picks its rule.

One vectorized core (`rule_side`) serves both the live dashboard (`decide_action`,
one row) and the backtest (`walk_forward`, thousands of rows), so what is
backtested is what runs. Costs are the study's: buy YES at the ask or NO at
1 - bid, per-order fee from kalshi_study.order_fee, never the mid.

A Rule trades only when (a) the book is real (kalshi_study.valid_book), (b) the
minutes left sit within `tolerance` of one of the rule's horizons, (c) the mid
is inside the rule's price band, and (d) the rule's probability beats the price
paid by more than `margin` after the fee. The probability is the market mid
nudged toward the Kalshi model by `shrink`:

    p_hat = mid + shrink * (p_model - mid)

shrink = 1 is the model itself (the rule the edge study tested and found
losing); shrink = 0 is the market (never trades). The only inputs are the
quote and kalshi.settlement_probability -- nothing is blended with the bot's
directional signals (CLAUDE.md principle 1), and `shrink` is NOT assumed: it is
one of the things the walk-forward picks, or rejects.

Walk-forward: days are split into contiguous folds; for each fold the rule is
chosen using only earlier days, then applied to the fold. A rule is chosen only
if its train mean net P&L beats zero by `SELECT_Z` standard errors on at least
`MIN_TRAIN_TRADES` trades; otherwise that fold trades nothing. The out-of-sample
P&L of those choices is the honest number the dashboard shows next to the
action. Days within the cache have been seen by the earlier studies, so this is
a backtest, not a pre-registered test: the 144-rule grid is counted when the
selection bar is set, and the result says "no edge" whenever it finds none.
"""
import itertools
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
import kalshi_study as ks

HORIZONS = ks.HORIZONS_MIN                  # (10, 5, 2): the snapshots the cache has
TOLERANCE_MIN = 1.0                          # live: minutes left within this of a rule horizon
MARGINS = (0.0, 0.02, 0.05, 0.08)
SHRINKS = (0.25, 0.5, 1.0)
HORIZON_SETS = ((10,), (5,), (2,), (10, 5), (5, 2), (10, 5, 2))
BANDS = (None, 0.25)                         # None = any mid; 0.25 = mid within 25-75c
MIN_TRAIN_TRADES = 200
SELECT_Z = 2.0                               # train mean must exceed zero by this many SE
N_FOLDS = 6
FIRST_TRAIN_FRACTION = 0.4


@dataclass(frozen=True)
class Rule:
    margin: float
    shrink: float
    horizons: tuple
    band: object = None                      # max |mid - 0.5|, or None

    def label(self) -> str:
        band = "any mid" if self.band is None else f"mid within {0.5 - self.band:.0%}-{0.5 + self.band:.0%}"
        return (f"shrink {self.shrink:g}, margin {self.margin * 100:g}c, "
                f"at {'/'.join(str(h) for h in self.horizons)} min, {band}")


GRID = tuple(Rule(m, k, h, b) for m, k, h, b in itertools.product(MARGINS, SHRINKS, HORIZON_SETS, BANDS))


# --- the shared core ---------------------------------------------------------

def rule_side(rule: Rule, bid, ask, p_model, horizon, contracts: int = 1):
    """
    Side per row: +1 buy YES, -1 buy NO, 0 no trade (elementwise numpy).
    `horizon` is the minutes left (nominal in the backtest, live in the
    dashboard); a row is eligible within TOLERANCE_MIN of one of the rule's.
    Also returns the net edge per contract of the chosen side (NaN if none).
    """
    bid, ask, p, h = (np.asarray(x, dtype=float) for x in (bid, ask, p_model, horizon))
    ok = ks.valid_book(bid, ask) & np.isfinite(p)
    in_window = np.zeros(ok.shape, dtype=bool)
    for hz in rule.horizons:
        in_window |= np.abs(h - hz) <= TOLERANCE_MIN
    ok &= in_window
    mid = (bid + ask) / 2
    if rule.band is not None:
        with np.errstate(invalid="ignore"):
            ok &= np.abs(mid - 0.5) <= rule.band + 1e-9
    p_hat = mid + rule.shrink * (p - mid)
    with np.errstate(invalid="ignore"):
        edge_yes = p_hat - ask - ks.fee_per_contract(ask, contracts)
        edge_no = (1 - p_hat) - (1 - bid) - ks.fee_per_contract(1 - bid, contracts)
        side = np.where(ok & (edge_yes > rule.margin), 1, np.where(ok & (edge_no > rule.margin), -1, 0))
    edge = np.where(side == 1, edge_yes, np.where(side == -1, edge_no, np.nan))
    return side, edge


def rule_pnl(rule: Rule, df: pd.DataFrame, contracts: int = 1):
    """
    (side, realized net P&L per contract, NaN where no trade) for each row of an
    observation frame. A market is entered at most once, at the earliest horizon
    where the rule fires, like the live engine -- otherwise one outcome would be
    counted at every horizon the rule allows.
    """
    side, _ = rule_side(rule, df["bid"], df["ask"], df["p_model"], df["horizon"], contracts)
    fired = pd.DataFrame({"k": df["close_ts"].to_numpy(), "h": df["horizon"].to_numpy()})[side != 0]
    first = fired.sort_values("h", ascending=False).drop_duplicates("k").index
    keep = np.zeros(len(df), dtype=bool)
    keep[first] = True
    side = np.where(keep, side, 0)
    pnl = np.where(side == 1, ks.ev_buy_yes(df["outcome"], df["ask"], contracts),
                   np.where(side == -1, ks.ev_buy_no(df["outcome"], df["bid"], contracts), np.nan))
    return side, pnl


# --- walk-forward ------------------------------------------------------------

def _stats(pnl, days):
    v = np.asarray(pnl, dtype=float)
    v = v[np.isfinite(v)]
    d = np.asarray(days)[np.isfinite(np.asarray(pnl, dtype=float))]
    s = ks.block_bootstrap(v, d) if len(v) else {"n": 0, "mean": float("nan"), "se": float("nan"), "n_eff": 0.0}
    return s


def select_rule(train: pd.DataFrame, grid=GRID, contracts: int = 1):
    """
    The rule with the best train mean net P&L among those clearing the bar
    (>= MIN_TRAIN_TRADES trades and mean > SELECT_Z * SE), or (None, table).
    """
    rows = []
    for r in grid:
        _, pnl = rule_pnl(r, train, contracts)
        s = _stats(pnl, train["day"])
        rows.append({"rule": r, **s})
    table = pd.DataFrame(rows)
    ok = table[(table.n >= MIN_TRAIN_TRADES) & (table["mean"] - SELECT_Z * table.se > 0)]
    if ok.empty:
        return None, table
    return ok.sort_values("mean", ascending=False).iloc[0]["rule"], table


def fold_days(days, n_folds: int = N_FOLDS, first_train: float = FIRST_TRAIN_FRACTION):
    """(train_days, test_days) pairs, expanding train window, contiguous test blocks."""
    d = sorted(set(days))
    start = int(len(d) * first_train)
    edges = np.linspace(start, len(d), n_folds + 1).astype(int)
    return [(d[:a], d[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def max_drawdown(pnl_in_time_order) -> float:
    c = np.cumsum(np.asarray(pnl_in_time_order, dtype=float))
    return float((np.maximum.accumulate(np.r_[0.0, c])[1:] - c).max()) if len(c) else 0.0


def walk_forward(obs: pd.DataFrame, contracts: int = 1, grid=GRID, n_folds: int = N_FOLDS) -> dict:
    """
    obs: rows with bid, ask, p_model, horizon, outcome (1 = YES), close_ts, day.
    Returns the out-of-sample record of the chosen-rule strategy, per fold, and
    the rule that would be chosen using ALL the data (what the dashboard runs).
    """
    d = obs[obs["horizon"].isin(HORIZONS) & obs["p_model"].notna()].sort_values("close_ts").reset_index(drop=True)
    folds, oos = [], []
    for train_days, test_days in fold_days(d["day"], n_folds):
        train, test = d[d.day.isin(train_days)], d[d.day.isin(test_days)]
        rule, _ = select_rule(train, grid, contracts)
        if rule is None:
            folds.append({"test_days": [test_days[0], test_days[-1]], "rule": None, "n": 0, "mean": None})
            continue
        side, pnl = rule_pnl(rule, test, contracts)
        t = test.assign(pnl=pnl).dropna(subset=["pnl"])
        oos.append(t[["close_ts", "day", "pnl"]])
        s = _stats(t["pnl"], t["day"])
        folds.append({"test_days": [test_days[0], test_days[-1]], "rule": asdict(rule), "n": s["n"],
                      "mean": s["mean"] if s["n"] else None})
    final_rule, final_table = select_rule(d, grid, contracts)
    oos_df = pd.concat(oos) if oos else pd.DataFrame(columns=["close_ts", "day", "pnl"])
    s = _stats(oos_df["pnl"], oos_df["day"])
    lo, hi = ks.interval(s, 1.96) if s["n"] > 1 and s["se"] == s["se"] else (float("nan"), float("nan"))
    oos_df = oos_df.sort_values("close_ts")
    return {
        "contracts": contracts,
        "folds": folds,
        "oos": {"n": int(s["n"]), "n_eff": s["n_eff"], "mean": s["mean"] if s["n"] else None,
                "ci95": [lo, hi], "total": float(oos_df["pnl"].sum()),
                "max_drawdown": max_drawdown(oos_df["pnl"]),
                "folds_traded": int(sum(1 for f in folds if f["rule"] is not None)), "folds": len(folds)},
        "rule": asdict(final_rule) if final_rule is not None else None,
        "n_rules_searched": len(grid),
        "table": final_table,
    }


# --- persistence for the dashboard -------------------------------------------

def save_result(result: dict, path: str, asof=None) -> None:
    out = {k: v for k, v in result.items() if k != "table"}
    out["asof"] = asof
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)


def load_active(path: str):
    """(Rule or None, result dict or None). None rule = the backtest found nothing worth trading, or hasn't run."""
    try:
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError):
        return None, None
    r = res.get("rule")
    if not r:
        return None, res
    return Rule(float(r["margin"]), float(r["shrink"]), tuple(r["horizons"]),
                None if r.get("band") is None else float(r["band"])), res


# --- the live decision ---------------------------------------------------------

NO_EDGE_LINE = ("Backtested, not proven: the Kalshi model has not been shown to beat the market's own "
                "price after fees (REVIEW_FINDINGS.md). NO TRADE is the default.")


def oos_line(res) -> str:
    if not res or not res.get("oos") or not res["oos"].get("n"):
        return "No out-of-sample trades in the backtest."
    o = res["oos"]
    lo, hi = o["ci95"]
    return (f"Walk-forward out of sample: {o['n']} trades, {o['mean'] * 100:+.2f}c each "
            f"[{lo * 100:+.2f}, {hi * 100:+.2f}] net of fees, traded in {o['folds_traded']}/{o['folds']} folds, "
            f"max drawdown ${o['max_drawdown']:.2f}/contract.")


# --- entry quality -----------------------------------------------------------
#
# Measured on the PRE-CUTOFF study rows only (edge_hunt.load_exploration, 246 days to
# 2026-08-17; review/kalshi_entry_quality.py prints every number below). None of it
# claims a direction: it says what an entry costs and how the market's own mid has
# been calibrated in the same state, with its n_eff and day-block 95% interval.

# (mid band low, high, horizon) -> n, n_eff, outcome - mid in cents, CI low, CI high
MID_CALIBRATION = {
    (0.00, 0.10, 10): (835, 877, -1.31, -2.84, +0.21),
    (0.00, 0.10, 5): (4820, 5061, -0.94, -1.45, -0.44),
    (0.00, 0.10, 2): (6430, 6136, -0.54, -0.89, -0.20),
    (0.10, 0.25, 10): (3490, 4336, -2.06, -3.15, -0.98),
    (0.10, 0.25, 5): (3105, 3297, -0.25, -1.51, +1.01),
    (0.10, 0.25, 2): (1478, 1515, -1.51, -3.29, +0.27),
    (0.25, 0.75, 10): (14178, 15888, +0.27, -0.48, +1.01),
    (0.25, 0.75, 5): (6542, 6645, -0.68, -1.83, +0.46),
    (0.25, 0.75, 2): (2897, 3122, +0.06, -1.59, +1.71),
    (0.75, 0.90, 10): (3321, 3505, +1.13, -0.09, +2.35),
    (0.75, 0.90, 5): (2959, 2929, +1.08, -0.22, +2.37),
    (0.75, 0.90, 2): (1455, 1399, +0.20, -1.73, +2.12),
    (0.90, 1.00, 10): (862, 940, +1.46, +0.03, +2.89),
    (0.90, 1.00, 5): (4853, 6541, +0.89, +0.44, +1.34),
    (0.90, 1.00, 2): (6349, 6476, +0.21, -0.15, +0.57),
}

# (horizon now, spread band) -> markets, share still tradable at the next snapshot,
# median spread change in cents, share narrower. Books under 2c did not change (median 0).
WAIT_TABLE = {
    (10, "2-5c"): (5508, 0.97, -1.0, 0.74),
    (10, "5c+"): (967, 0.89, -3.0, 0.76),
    (5, "2-5c"): (4656, 0.78, -1.0, 0.74),
    (5, "5c+"): (791, 0.68, -3.0, 0.71),
}


def nearest_horizon(minutes_left, tolerance: float = TOLERANCE_MIN):
    """The cached snapshot horizon within `tolerance` of the clock, else None."""
    for hz in HORIZONS:
        if abs(minutes_left - hz) <= tolerance:
            return hz
    return None


def entry_cost(yes_bid, yes_ask, contracts: int = 1) -> dict:
    """
    What entering at the touch costs against a fair mid, per contract, as fractions:
    half the spread plus the per-order fee for each side, plus each side's breakeven
    probability (price paid + fee).
    """
    mid = (yes_bid + yes_ask) / 2
    fee_yes = float(ks.fee_per_contract(yes_ask, contracts))
    fee_no = float(ks.fee_per_contract(1 - yes_bid, contracts))
    half = (yes_ask - yes_bid) / 2
    return {"mid": mid, "half_spread": half, "fee_yes": fee_yes, "fee_no": fee_no,
            "cost_yes": half + fee_yes, "cost_no": half + fee_no,
            "breakeven_yes": yes_ask + fee_yes, "breakeven_no": (1 - yes_bid) + fee_no}


def mid_calibration(mid, minutes_left):
    """
    How the market mid has been calibrated in this price band at this horizon:
    {"n", "n_eff", "gap_c", "lo_c", "hi_c", "band", "horizon"} (gap = outcome - mid,
    cents, day-block 95% CI), or None when the clock is outside every cached window.
    """
    hz = nearest_horizon(minutes_left)
    if hz is None or mid is None or not (0 <= mid <= 1):
        return None
    for (lo, hi, h), (n, n_eff, gap, a, b) in MID_CALIBRATION.items():
        if h == hz and (lo < mid <= hi or (lo == 0.0 and mid == 0.0)):
            return {"n": n, "n_eff": n_eff, "gap_c": gap, "lo_c": a, "hi_c": b, "band": (lo, hi), "horizon": hz}
    return None


def wait_hint(spread, minutes_left, min_spread: float = None) -> str:
    """
    "Cheaper to wait?" -- only what the cache supports: books 2c and wider have usually
    tightened by the next snapshot, at the price of some becoming untradable. Empty for
    tight books (they did not change) and at the last snapshot (nothing later to wait for).
    """
    min_spread = config.DECISION_WAIT_MIN_SPREAD if min_spread is None else min_spread
    hz = nearest_horizon(minutes_left)
    if hz is None or hz == HORIZONS[-1] or spread < min_spread - 1e-9:
        return ""
    band = "5c+" if spread >= config.DECISION_SPREAD_EXPENSIVE - 1e-9 else "2-5c"
    n, tradable, med, narrower = WAIT_TABLE[(hz, band)]
    nxt = HORIZONS[HORIZONS.index(hz) + 1]
    return (f"Waiting: books this wide were narrower at {nxt} min in {narrower:.0%} of {n} markets "
            f"(median {med:+.0f}c), and {tradable:.0%} were still tradable then.")


def entry_quality(yes_bid, yes_ask, last_price, minutes_left, contracts: int = 1) -> dict:
    """
    GOOD / EXPENSIVE / AVOID for entering this book now, with the measured reasons, the
    cost in cents (average of the two sides' half-spread + fee), the market's calibration
    in this state and a wait hint. Says nothing about which side.
    """
    out = {"grade": "AVOID", "cost_c": None, "reasons": [], "calibration": None, "wait": "", "cost": None}
    if yes_bid is None or yes_ask is None or not ks.valid_book(yes_bid, yes_ask):
        out["reasons"].append("no tradable two-sided book")
        return out
    c = entry_cost(yes_bid, yes_ask, contracts)
    spread = yes_ask - yes_bid
    out["cost"] = c
    out["cost_c"] = 100 * (c["cost_yes"] + c["cost_no"]) / 2
    grade = "GOOD"
    if spread >= config.DECISION_SPREAD_AVOID - 1e-9:
        grade = "AVOID"
        out["reasons"].append(f"spread {spread * 100:.0f}c: entries this wide have cost ~8c and the mid says little")
    elif spread >= config.DECISION_SPREAD_EXPENSIVE - 1e-9:
        grade = "EXPENSIVE"
        out["reasons"].append(f"spread {spread * 100:.0f}c: entries 5-10c wide have cost ~5c")
    if last_price is not None and abs(last_price - c["mid"]) > config.DECISION_STALE_PRINT:
        grade = "AVOID" if grade == "AVOID" else "EXPENSIVE"
        out["reasons"].append(f"last print {last_price * 100:.0f}c is {abs(last_price - c['mid']) * 100:.0f}c "
                              f"off the mid: stale or placeholder quote")
    if c["mid"] <= config.DECISION_EXTREME_MID or c["mid"] >= 1 - config.DECISION_EXTREME_MID:
        fav = "YES" if c["mid"] >= 0.5 else "NO"
        px = yes_ask if fav == "YES" else 1 - yes_bid
        fee = c["fee_yes"] if fav == "YES" else c["fee_no"]
        out["reasons"].append(f"payout asymmetry: {fav} at {px * 100:.0f}c wins {(1 - px) * 100:.0f}c, "
                              f"risks {px * 100:.0f}c + {fee * 100:.0f}c fee")
    out["grade"] = grade
    out["calibration"] = mid_calibration(c["mid"], minutes_left)
    out["wait"] = wait_hint(spread, minutes_left)
    return out


def calibration_line(cal) -> str:
    if cal is None:
        return ""
    lo, hi = cal["band"]
    return (f"Market mid at {lo * 100:.0f}-{hi * 100:.0f}c, {cal['horizon']} min: outcomes ran "
            f"{cal['gap_c']:+.2f}c vs the mid [{cal['lo_c']:+.2f}, {cal['hi_c']:+.2f}] "
            f"(n={cal['n']}, n_eff={cal['n_eff']}).")


def decide_action(rule, p_model, yes_bid, yes_ask, minutes_left, contracts: int = 1, result=None) -> dict:
    """
    One live decision. Returns {"action": "BUY YES" | "BUY NO" | "NO TRADE", "side", "price", "fee",
    "breakeven", "edge", "reasons": [...], "flip": str, "evidence": str}. `rule` None (no backtest, or it
    found nothing selectable) is always NO TRADE, with the reason.
    """
    out = {"action": "NO TRADE", "side": None, "price": None, "fee": None, "breakeven": None, "edge": None,
           "reasons": [], "flip": "", "evidence": oos_line(result) if result else "", "caveat": NO_EDGE_LINE}
    if yes_bid is None or yes_ask is None:
        out["reasons"].append("no two-sided Kalshi quote")
        return out
    if not ks.valid_book(yes_bid, yes_ask):
        out["reasons"].append(f"book not tradable (bid {yes_bid:.2f}, ask {yes_ask:.2f}; spread over "
                              f"{ks.MAX_SPREAD * 100:.0f}c or one-sided)")
        return out
    if p_model is None:
        out["reasons"].append("no model probability for this strike")
        return out
    if rule is None:
        out["reasons"].append("backtest has not selected a rule that beats zero out of sample "
                              "(run review/kalshi_decision_backtest.py)" if result is None or result.get("rule") is None
                              else "no active rule")
        return out
    if not any(abs(minutes_left - hz) <= TOLERANCE_MIN for hz in rule.horizons):
        out["reasons"].append(f"{minutes_left:.1f} min left is outside the windows the rule was backtested at "
                              f"({', '.join(str(h) for h in rule.horizons)} min, \u00b1{TOLERANCE_MIN:g})")
        out["flip"] = "the rule only speaks inside those windows"
        return out
    side, edge = rule_side(rule, [yes_bid], [yes_ask], [p_model], [minutes_left], contracts)
    s, e = int(side[0]), float(edge[0])
    if s == 0:
        mid = (yes_bid + yes_ask) / 2
        p_hat = mid + rule.shrink * (p_model - mid)
        out["reasons"].append(f"rule ({rule.label()}) sees no net edge: p {p_hat * 100:.0f}% vs "
                              f"YES ask {yes_ask * 100:.0f}c / NO cost {(1 - yes_bid) * 100:.0f}c, "
                              f"{minutes_left:.1f} min left")
        out["flip"] = f"needs more than {rule.margin * 100:g}c of edge after the fee, inside the rule's window"
        return out
    price = yes_ask if s == 1 else 1 - yes_bid
    fee = float(ks.fee_per_contract(price, contracts))
    out.update(action="BUY YES" if s == 1 else "BUY NO", side="yes" if s == 1 else "no", price=price, fee=fee,
               breakeven=price + fee, edge=e)
    out["reasons"].append(f"rule ({rule.label()}): net edge {e * 100:+.1f}c per contract at "
                          f"{price * 100:.0f}c + {fee * 100:.1f}c fee (breakeven {(price + fee) * 100:.1f}%)")
    out["flip"] = "edge falls under the margin if the quote moves toward the model"
    return out


def decide_exit(side: str, p_model, yes_bid, yes_ask, minutes_left, rule=None, contracts: int = 1,
                exit_margin: float = 0.03) -> dict:
    """
    HOLD TO SETTLE vs EXIT NOW for an open position. Exit sells back at the bid
    (YES) or at 1 - ask (NO) and pays the fee on the sale; holding pays nothing
    more. The hold value is the rule's p_hat for the position's side, or the
    market mid with no rule. EXIT only when the net sale beats the hold value
    by `exit_margin`: the scalping study found a round trip costs 4-5c and
    nothing beat it, so this rarely fires.
    """
    if side not in ("yes", "no") or yes_bid is None or yes_ask is None or not ks.valid_book(yes_bid, yes_ask):
        return {"action": "HOLD TO SETTLE", "reason": "no tradable exit price"}
    mid = (yes_bid + yes_ask) / 2
    shrink = rule.shrink if rule is not None else 0.0
    p_yes = mid + shrink * (p_model - mid) if p_model is not None else mid
    sale = yes_bid if side == "yes" else 1 - yes_ask
    fee = float(ks.fee_per_contract(sale, contracts)) if sale > 0 else 0.0
    hold = p_yes if side == "yes" else 1 - p_yes
    net_sale = sale - fee
    if net_sale - hold > exit_margin:
        return {"action": "EXIT NOW", "reason": f"sale nets {net_sale * 100:.0f}c vs {hold * 100:.0f}c expected from holding"}
    return {"action": "HOLD TO SETTLE",
            "reason": f"sale nets {net_sale * 100:.0f}c vs {hold * 100:.0f}c expected from holding"}
