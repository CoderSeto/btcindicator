"""
Four further KXBTC15M strategy hypotheses, tested the way the edge and scalp
studies were. Reusable study code.

The edge study (kalshi_study.py) found no edge in buying by price bucket or
toward the GBM model and holding to settlement; the scalp study
(kalshi_scalp_study.py) found none in four short momentum / reversion scalps.
Both listed what they had NOT tested. This asks four of those questions, each
as a rule and its mirror, so the data can speak in either direction:

  residual    does the contract over- or under-react to a move in BTC?
  streak      does a run of same-way settlements carry into the next contract
              (or does the market over-price that it will)?
  vol_regime  does the price lag a change in short-term volatility?
  basis       is a strike set while the index sat unusually far from Coinbase
              mispriced once that gap closes?

This module holds the arithmetic, all tested on hand-built data.
`review/kalshi_strategy_backtest.py` runs it on the cached study data.

PRE-REGISTERED DESIGN. Fixed, and committed, before any of these rules had been
run on any outcome: no study_data existed in the environment this was written
in, and none was fetched until after the commit. Changing a constant after
seeing results is tuning on the held-out data; add a variant instead, count it
in the family, and label it as an amendment below.

What the designer HAD seen, stated because it is the honest limit of "blind":
REVIEW_FINDINGS.md, which prints train AND descriptive held-out numbers for the
two earlier studies (price buckets, the GBM model, four scalp rules, the
Coinbase-vs-index basis distribution). No number for any rule below was in it.
The one held-out pattern it flags as a hypothesis (the 2-minute calibration
reversal) is deliberately NOT tested here: it was found on the held-out set, so
that set cannot test it.

  - Data, split, bootstrap, costs: exactly kalshi_study's, not reimplemented.
    Contracts sorted by close time, the last TEST_FRACTION (30%) held out
    (ks.split_cutoff over every parsed contract, so the held-out set is the
    same contracts as in the two earlier studies); intervals are a block
    bootstrap by UTC day (ks.block_bootstrap), mean +/- z * SE; YES is bought
    at the ask and NO at 1 - bid; every order pays ks.order_fee (rounded up to
    the cent PER ORDER), on every leg; a real book is ks.valid_book. Fills at
    the quoted top of book with zero latency are assumed -- candlesticks carry
    no depth, so the size really available may be smaller. Never the mid.
  - Entry: at most one trade per contract per cell, at the quote standing a
    fixed number of minutes before close, only if that is a real book. A rule
    reads only what was known then: quotes at or before the entry minute,
    Coinbase 1-minute candles that ENDED at or before it, strikes (published at
    open) and the results of contracts that closed at or before this one
    opened. Nothing after the entry is read by any rule.
  - Exit: hold to settlement (one fee, no second spread), except the two
    residual `sell` cells, which sell EXIT_HOLD_MIN minutes after entry at the
    quote standing then (YES at the bid, NO at 1 - ask, a second fee), using
    kalshi_scalp_study.scalp_pnl: if that is not a real book the position is
    HELD TO SETTLEMENT instead and counted (`forced`), never dropped.
  - Rules. Every threshold is inclusive and fixed.
      residual_fade / residual_follow    entry RESIDUAL_HORIZONS (10, 5) min
          residual = (mid now - mid RESIDUAL_LOOKBACK_MIN (3) minutes ago)
                   - (p now - p then), p = kalshi.settlement_probability on the
          last Coinbase trade before each of the two times, the minutes left at
          each, and ONE vol for both (ks.model_inputs at entry: the 120-minute
          realized vol), so a change in vol cannot pose as a residual. The
          difference removes the model's level error (the basis), which is what
          sank it in the edge study. Both quotes must be real books; no model
          value when ks.model_inputs has none or when the strike is more than
          kalshi.MAX_STRIKE_DISTANCE from spot (Kalshi's 10^-4 strike errors).
          fade: residual >= +RESIDUAL_MOVE (0.03) buys NO, <= -0.03 buys YES
          (the contract moved further than BTC explains, and comes back).
          follow: the opposite side (the contract knows something the Coinbase
          minute does not). Each with two exits: settlement, and selling 3
          minutes later.
      streak_follow / streak_fade        entry EARLY_HORIZONS (13, 10) min
          The STREAK_LENGTH (2) contracts before this one -- the one that
          closed exactly when this one opened, and the one before that --
          settled the same way. follow buys that side, fade the other. No
          trade if either is missing from the parsed contracts. Assumes those
          results are public by 2 minutes after the close they settle on.
      vol_regime / vol_regime_inverse    entry RESIDUAL_HORIZONS (10, 5) min
          ratio = realized vol of the last VOL_SHORT_MIN (15) one-minute
          Coinbase returns / that of the last 120 (ks.model_inputs), both
          ending at entry; the short window needs all 16 closes. Only when
          there is a favorite and it is not near-certain: VOL_BAND_LO (0.10)
          <= |mid - 0.5| <= VOL_BAND_HI (0.40). vol_regime: ratio >= VOL_RATIO
          (1.5) buys the UNDERDOG (vol has jumped; if the price lags, the
          favorite is over-priced), ratio <= 1 / 1.5 buys the FAVORITE.
          vol_regime_inverse: the opposite side in both regimes.
      basis_fade / basis_follow          entry EARLY_HORIZONS (13, 10) min
          basis = strike / (Coinbase OHLC mean of the minute before open) - 1,
          in bp: where the index's strike-setting average sat against Coinbase.
          dev = basis - median basis of the previous BASIS_WINDOW (96)
          contracts by open time, needing BASIS_MIN_HISTORY (48) of them; a
          basis beyond +/- BASIS_MAX_ABS_BP (20) is a data error or a minute
          the proxy cannot follow and counts as missing, here and in the
          median. fade: dev >= +BASIS_DEV_BP (1.5) buys NO (the strike was set
          high against Coinbase; if the gap closes, the bar is harder to
          clear), dev <= -1.5 buys YES. follow: the opposite side.
  - The family: CELLS (20: residual 2 rules x 2 entries x 2 exits, the other
    three 2 rules x 2 entries each) x ORDER_SIZES (1 and 100 contracts) = 40
    tests. Its own family, as the scalp study's was: different trades, a
    different question. Every interval uses one Bonferroni z over 40.
  - Metric: mean profit per contract, net of every cost above.
  - Selection: kalshi_scalp_study.select_and_test, unchanged. A cell is
    "selected" on train only if its family interval lies entirely above zero;
    only selected cells are then scored on the held-out set as a claim, with a
    second Bonferroni correction over the number selected. The held-out set is
    run ONCE. Its numbers for the other cells are printed but are descriptive,
    and nothing is claimed from them.
  - Small samples: a cell with train n_eff below MIN_N_EFF (200) is flagged in
    the output. The flag does not change the selection rule.
  - Descriptive, not counted: for settlement cells the win rate next to the
    break-even rate (the mean price paid plus fee: what the win rate must
    exceed); and the direction information in each mirrored pair (half the
    difference of the two means: what the signal is worth before costs).
  - Not tested, so not ruled out: maker (resting) orders, anything finer than a
    one-minute candle, other venues' prices (Binance.US and Bybit history was
    unreachable where this was written), the live order-book signals (no
    history exists), and every threshold other than the ones fixed above.
RUN NOTE, added after the single run (2026-11-10); nothing above was changed
and no amendment was made. The cache was built that day and holds 30,950
contracts, 262 more than the earlier studies' 30,688, so ks.split_cutoff put
the held-out boundary two days later (2026-08-04): overlapping, but not "the
same contracts" as stated above. Result: no cell selected on train
(REVIEW_FINDINGS.md, "Four further strategy hypotheses").
"""
import numpy as np
import pandas as pd

import kalshi
import kalshi_scalp_study as sc
import kalshi_study as ks

RESIDUAL_HORIZONS = (10, 5)
EARLY_HORIZONS = (13, 10)
RESIDUAL_LOOKBACK_MIN = 3
RESIDUAL_MOVE = 0.03                # dollars of (mid change - model change)
EXIT_HOLD_MIN = 3                   # the residual `sell` exit
SETTLE = 0                          # `hold` value meaning: held to settlement
STREAK_LENGTH = 2
VOL_SHORT_MIN = 15
VOL_RATIO = 1.5
VOL_BAND_LO, VOL_BAND_HI = 0.10, 0.40
BASIS_WINDOW = 96                   # contracts: 24 hours
BASIS_MIN_HISTORY = 48
BASIS_DEV_BP = 1.5
BASIS_MAX_ABS_BP = 20.0
MIN_N_EFF = 200
ORDER_SIZES = ks.ORDER_SIZES

PAIRS = (("residual_fade", "residual_follow"), ("streak_follow", "streak_fade"),
         ("vol_regime", "vol_regime_inverse"), ("basis_fade", "basis_follow"))
RULES = tuple(r for pair in PAIRS for r in pair)

# (rule, entry minutes before close, hold: SETTLE or minutes until the sale)
CELLS = tuple(
    [(r, h, k) for r in PAIRS[0] for h in RESIDUAL_HORIZONS for k in (SETTLE, EXIT_HOLD_MIN)]
    + [(r, h, SETTLE) for r in PAIRS[1] for h in EARLY_HORIZONS]
    + [(r, h, SETTLE) for r in PAIRS[2] for h in RESIDUAL_HORIZONS]
    + [(r, h, SETTLE) for r in PAIRS[3] for h in EARLY_HORIZONS])
STRATEGY_TESTS = len(CELLS) * len(ORDER_SIZES)
FRAME_HORIZONS = tuple(sorted({h for _, h, _ in CELLS}, reverse=True))

_EPS = 1e-9


# --- features (each uses only what was known at entry) ---------------------------

def model_probability(grid, t: int, strike: float, minutes: float, vol: float) -> float:
    """
    kalshi.settlement_probability on the last Coinbase trade before `t`, or NaN
    when that minute is missing or the strike fails kalshi.strike_check (the
    10^-4 strike errors price at exactly 0 or 1 and would make every model
    change zero).
    """
    spot = grid.last_close_before(t)
    if spot is None or not strike > 0 or kalshi.strike_check(strike, spot) is not None:
        return float("nan")
    return float(kalshi.settlement_probability(spot, strike, minutes, vol))


def residual_feature(grid, strike: float, t0: int, horizon: int, mid_now: float, mid_prev: float) -> float:
    """
    (mid now - mid RESIDUAL_LOOKBACK_MIN ago) - (model now - model then), with
    one vol -- the one known at entry -- for both model values. NaN if either
    mid, the vol or either model value is missing.
    """
    if not (mid_now == mid_now and mid_prev == mid_prev):
        return float("nan")
    mi = ks.model_inputs(grid, t0)
    if mi is None:
        return float("nan")
    vol = mi[1]
    p_now = model_probability(grid, t0, strike, horizon, vol)
    p_prev = model_probability(grid, t0 - 60 * RESIDUAL_LOOKBACK_MIN, strike, horizon + RESIDUAL_LOOKBACK_MIN, vol)
    return (mid_now - mid_prev) - (p_now - p_prev)


def vol_ratio_feature(grid, t0: int) -> float:
    """
    Realized vol of the VOL_SHORT_MIN one-minute returns ending at t0 over that
    of the 120 ending at t0 (ks.model_inputs). Both go through
    kalshi.realized_volatility, so the annualization cancels. NaN unless every
    close of the short window exists and it moved at all.
    """
    mi = ks.model_inputs(grid, t0)
    closes = grid.closes_before(t0, VOL_SHORT_MIN + 1)
    if mi is None or closes is None or np.isnan(closes).any():
        return float("nan")
    short = kalshi.realized_volatility(pd.DataFrame({"close": closes}), VOL_SHORT_MIN, 60)["annual_vol"]
    if short is None:
        return float("nan")
    return float(short / mi[1])


def streak_features(markets: list) -> dict:
    """
    {ticker: +1 / -1 / 0 / NaN}: +1 if the STREAK_LENGTH contracts before it
    all settled YES, -1 if all NO, 0 if mixed, NaN if the chain is broken. The
    contract "before" one is the one whose close is that one's open. Two
    contracts sharing a close make the chain ambiguous, so it counts as broken.
    """
    by_close, clash = {}, set()
    for m in markets:
        if m["close_ts"] in by_close:
            clash.add(m["close_ts"])
        by_close[m["close_ts"]] = m
    out = {}
    for m in markets:
        outcomes, cur = [], m
        for _ in range(STREAK_LENGTH):
            prev = by_close.get(cur["open_ts"])
            if prev is None or cur["open_ts"] in clash:
                outcomes = None
                break
            outcomes.append(prev["outcome"])
            cur = prev
        if outcomes is None:
            out[m["ticker"]] = float("nan")
        else:
            out[m["ticker"]] = 1.0 if all(o == 1 for o in outcomes) else -1.0 if all(o == 0 for o in outcomes) else 0.0
    return out


def basis_features(markets: list, grid) -> dict:
    """
    {ticker: dev}: the contract's basis at open, in bp, minus the median basis
    of the BASIS_WINDOW contracts that opened before it (strictly earlier open,
    by open order), NaN without BASIS_MIN_HISTORY usable ones. The basis is
    strike / (Coinbase OHLC mean of the minute before open) - 1; beyond
    +/- BASIS_MAX_ABS_BP it is treated as missing everywhere.
    """
    order = sorted(markets, key=lambda m: (m["open_ts"], m["ticker"]))
    basis = []
    for m in order:
        proxy = grid.minute_average_before(m["open_ts"])
        b = (m["strike"] / proxy - 1) * 1e4 if proxy else float("nan")
        basis.append(b if abs(b) <= BASIS_MAX_ABS_BP + _EPS else float("nan"))
    basis = np.array(basis, dtype=float)
    opens = np.array([m["open_ts"] for m in order], dtype=np.int64)
    out = {}
    for i, m in enumerate(order):
        j = int(np.searchsorted(opens, opens[i], side="left"))     # first contract sharing this open: not "before"
        past = basis[max(0, j - BASIS_WINDOW): j]
        past = past[~np.isnan(past)]
        if np.isnan(basis[i]) or len(past) < BASIS_MIN_HISTORY:
            out[m["ticker"]] = float("nan")
        else:
            out[m["ticker"]] = float(basis[i] - np.median(past))
    return out


# --- the rules -------------------------------------------------------------------

def _threshold_sides(x, thresh: float) -> np.ndarray:
    """+1 where x >= thresh, -1 where x <= -thresh, else 0 (NaN -> 0). Inclusive despite float residue."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(x >= thresh - _EPS, 1, np.where(x <= -thresh + _EPS, -1, 0))


def rule_sides(frame: pd.DataFrame, rule: str) -> np.ndarray:
    """+1 buy YES, -1 buy NO, 0 no trade, for each row of `frame`, from its entry-time features only."""
    if rule in PAIRS[0]:
        fade = -_threshold_sides(frame["residual"], RESIDUAL_MOVE)
        return fade if rule == "residual_fade" else -fade
    if rule in PAIRS[1]:
        s = frame["streak"].to_numpy(dtype=float)
        follow = np.where(s == 1, 1, np.where(s == -1, -1, 0))
        return follow if rule == "streak_follow" else -follow
    if rule in PAIRS[2]:
        mid = ((frame["bid0"] + frame["ask0"]) / 2).to_numpy(dtype=float)
        ratio = frame["vol_ratio"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            dist = np.abs(mid - 0.5)
            in_band = (dist >= VOL_BAND_LO - _EPS) & (dist <= VOL_BAND_HI + _EPS)
            favorite = np.where(mid > 0.5, 1, -1)
            regime = np.where(ratio >= VOL_RATIO - _EPS, -1, np.where(ratio <= 1 / VOL_RATIO + _EPS, 1, 0))
        side = np.where(in_band, regime * favorite, 0)           # spike: the underdog; calm: the favorite
        return side if rule == "vol_regime" else -side
    if rule in PAIRS[3]:
        fade = -_threshold_sides(frame["basis_dev_bp"], BASIS_DEV_BP)
        return fade if rule == "basis_fade" else -fade
    raise ValueError(f"unknown rule {rule!r}")


# --- building the trade frame ----------------------------------------------------

def _nan(x):
    return float("nan") if x is None else float(x)


def _mid(q) -> float:
    """Mid of a quote if it is a real book, else NaN."""
    b, a = _nan(q[0]), _nan(q[1])
    return (b + a) / 2 if bool(ks.valid_book(b, a)) else float("nan")


def trade_frame(markets: list, candles_by_ticker: dict, grid, horizons=FRAME_HORIZONS) -> pd.DataFrame:
    """
    One row per (contract, entry horizon): the entry quote, the quote
    EXIT_HOLD_MIN minutes later (read by the `sell` exit only, never by a
    rule), the outcome, and the four features. `candles_by_ticker` maps ticker
    -> ks.parse_candles(...); `grid` is a ks.MinuteGrid of Coinbase 1-minute
    candles. Rows with a missing quote or feature keep NaN, so the caller can
    count what it dropped. A quote is read only at its exact minute.
    """
    streak = streak_features(markets)
    basis = basis_features(markets, grid)
    rows = []
    for m in markets:
        c = candles_by_ticker.get(m["ticker"], {})
        for h in horizons:
            t0 = m["close_ts"] - 60 * h
            q0 = ks.quote_at(c, t0) or (None, None, None)
            qp = ks.quote_at(c, t0 - 60 * RESIDUAL_LOOKBACK_MIN) or (None, None, None)
            q1 = ks.quote_at(c, t0 + 60 * EXIT_HOLD_MIN) or (None, None, None)
            rows.append({
                "ticker": m["ticker"], "close_ts": m["close_ts"], "outcome": m["outcome"], "horizon": h,
                "bid0": _nan(q0[0]), "ask0": _nan(q0[1]), "bid1": _nan(q1[0]), "ask1": _nan(q1[1]),
                "residual": residual_feature(grid, m["strike"], t0, h, _mid(q0), _mid(qp)),
                "streak": streak[m["ticker"]],
                "vol_ratio": vol_ratio_feature(grid, t0),
                "basis_dev_bp": basis[m["ticker"]],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["close_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


with_split = sc.with_split


# --- the family ------------------------------------------------------------------

def trade_pnl(side, bid0, ask0, bid1, ask1, outcome, hold: int, contracts: int):
    """
    Profit per contract, net of every fee. hold == SETTLE: buy and hold to
    settlement (ks.ev_buy_yes / ks.ev_buy_no). Otherwise: buy and sell at the
    exit quote, held to settlement where that is not a real book
    (sc.scalp_pnl). Returns (pnl, forced_hold); forced is all False for SETTLE.
    """
    side = np.asarray(side)
    if hold == SETTLE:
        pnl = np.where(side == 1, ks.ev_buy_yes(outcome, ask0, contracts), ks.ev_buy_no(outcome, bid0, contracts))
        return pnl, np.zeros(len(side), dtype=bool)
    return sc.scalp_pnl(side, bid0, ask0, bid1, ask1, outcome, contracts)


def cell_trades(frame: pd.DataFrame, rule: str, horizon: int, hold: int, contracts: int, split=None):
    """(rows traded, sides, pnl per contract, forced-hold flags) for one cell of the family."""
    d = frame[frame["horizon"] == horizon]
    if split is not None:
        d = d[d["split"] == split]
    d = d[ks.valid_book(d["bid0"].to_numpy(), d["ask0"].to_numpy())]
    side = rule_sides(d, rule)
    take = side != 0
    d, side = d[take], side[take]
    pnl, forced = trade_pnl(side, d["bid0"], d["ask0"], d["bid1"], d["ask1"], d["outcome"], hold, contracts)
    return d, side, pnl, forced


def win_and_breakeven(d: pd.DataFrame, side, contracts: int):
    """
    For hold-to-settlement trades: (win rate, break-even rate). A trade wins
    $1 or nothing, so its break-even win probability is what it paid: the
    price plus the fee per contract. mean profit = win rate - break-even rate.
    """
    side = np.asarray(side)
    if len(side) == 0:
        return float("nan"), float("nan")
    price = np.where(side == 1, d["ask0"].to_numpy(dtype=float), 1.0 - d["bid0"].to_numpy(dtype=float))
    won = np.where(side == 1, d["outcome"].to_numpy(dtype=float), 1.0 - d["outcome"].to_numpy(dtype=float))
    return float(won.mean()), float((price + ks.fee_per_contract(price, contracts)).mean())


def family_results(frame: pd.DataFrame, z: float) -> pd.DataFrame:
    """Every cell of the family, on train and on test, with a +/- z block-bootstrap interval."""
    rows = []
    for split in ("train", "test"):
        for rule, h, k in CELLS:
            for n in ORDER_SIZES:
                d, side, pnl, forced = cell_trades(frame, rule, h, k, n, split)
                s = ks.block_bootstrap(pnl, d["day"].to_numpy())
                lo, hi = ks.interval(s, z) if s["n"] > 1 else (np.nan, np.nan)
                win, breakeven = win_and_breakeven(d, side, n) if k == SETTLE else (float("nan"), float("nan"))
                rows.append({
                    "split": split, "rule": rule, "horizon": h, "hold": k, "contracts": n,
                    "n": s["n"], "days": int(d["day"].nunique()),
                    "yes_share": float(np.mean(side == 1)) if len(side) else float("nan"),
                    "forced_rate": float(forced.mean()) if len(forced) else float("nan"),
                    "win_rate": win, "breakeven": breakeven,
                    "mean": s["mean"], "se": s["se"], "lo": lo, "hi": hi,
                    "mde": z * s["se"] if s["n"] > 1 else float("nan"), "n_eff": s["n_eff"],
                    "small": not (s["n_eff"] >= MIN_N_EFF),
                })
    return pd.DataFrame(rows)


select_and_test = sc.select_and_test            # the pre-registered selection rule, unchanged


def direction_information(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per mirrored pair, entry, hold, size and split: half the difference of the
    two rules' mean profits. The two trade the same contracts on opposite
    sides, so this is what the signal's direction is worth per trade before
    costs (positive: the first-named rule's side is the better one).
    Descriptive, not a test.
    """
    rows = []
    for a, b in PAIRS:
        ra = results[results["rule"] == a]
        rb = results[results["rule"] == b]
        m = ra.merge(rb, on=["split", "horizon", "hold", "contracts"], suffixes=("_a", "_b"))
        for r in m.itertuples():
            rows.append({"split": r.split, "pair": f"{a} vs {b}", "horizon": r.horizon, "hold": r.hold,
                         "contracts": r.contracts, "n": r.n_a, "information": (r.mean_a - r.mean_b) / 2})
    return pd.DataFrame(rows)
