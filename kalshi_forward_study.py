"""
Evaluator for the pre-registered forward edge hunt in KXBTC15M.

The design is EDGE_HYPOTHESES.md, committed before any forward row existed.
Every constant below is a transcription of that file; changing one after
data has been looked at is a labelled amendment there first, counted in the
family, and never a quiet edit here.

Input: edge_record.csv from edge_recorder.py (tick and gap rows), the Kalshi
prints log is not needed, and settled outcomes from
kalshi_tracking.resolve_outcomes. Output: for each hypothesis its trades,
mean net P&L per contract, day-block interval, n, n_eff, the verdict under
the fixed stopping and futility rules, and the mandatory sanity checks.
`review/kalshi_forward_edge.py` runs it.

Confirmatory rows are those stamped at or after CONFIRMATORY_FROM; earlier
rows are refused (counted and dropped), so nothing recorded before the
pre-registration can enter a verdict.

Not modelled (EDGE_HYPOTHESES.md, "Common rules"): queue position, impact,
size behind the touch, latency beyond the recorder's own round trip. Every
one flatters the result; a negative here is therefore an upper bound.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import kalshi
import kalshi_study as ks

# --- pre-registered constants (EDGE_HYPOTHESES.md) -------------------------
CONFIRMATORY_FROM = datetime(2026, 11, 8, 23, 15, tzinfo=timezone.utc)
STOP_DATE = datetime(2027, 5, 8, tzinfo=timezone.utc)
STOP_N_EFF = 2630                      # 80% power at a 3c edge
FUTILITY_HALF_WIDTH = 0.02             # z*SE above this -> inconclusive
FAMILY = 5
Z = ks.bonferroni_z(FAMILY)            # 2.576
CONTRACTS = 1
WINDOW_S = (60.0, 300.0)
MID_GATE = (0.40, 0.60)
H1_LOOKBACK_S, H1_MOVE_BP, H1_MARGIN, H1_MAX_NEXT_S = 10.0, 5.0, 0.01, 3.0
H2_WINDOW_S, H2_MARGIN = (60.0, 180.0), 0.02
H3_IMB, H3_PERSIST_S = 0.18, 30.0
H4_DELTA = 0.30
H5_LIQ_USD = 1_000_000.0
PERMUTATIONS, PERMUTATION_SEED = 2000, 20261108
HOUR_BINS = ((0, 6), (6, 12), (12, 18), (18, 24))
HYPOTHESES = ("H1", "H2", "H3", "H4", "H5")
TITLES = {"H1": "stale Kalshi quote after a fast Coinbase move",
          "H2": "final-minutes quote prices the point rule, not the average",
          "H3": "persistent aggregated depth imbalance",
          "H4": "aggressive trade-flow imbalance",
          "H5": "liquidation cascade momentum"}

NUMERIC = ["gap_seconds", "strike", "secs_to_close", "k_yes_bid", "k_yes_ask", "k_yes_bid_sz", "k_yes_ask_sz",
           "k_last", "k_quote_age_s", "k_rtt_ms", "k_up", "cb_price", "cb_up", "book_imb", "book_imb_near",
           "book_microprice_bps", "book_flow", "book_persist_s", "book_venues_up", "book_up", "agg_delta",
           "agg_buy_notional", "agg_sell_notional", "agg_venues_up", "agg_up", "liq_long_notional",
           "liq_short_notional", "liq_up", "vol_ann", "vol_samples", "p_settle", "p_point"]


# --- loading -----------------------------------------------------------------

def load_record(path_or_df, confirmatory_from: datetime = CONFIRMATORY_FROM) -> dict:
    """
    Reads the recorder's CSV (or takes a DataFrame of it). Returns
    {"ticks", "gaps", "refused", "first_ts", "last_ts"}; `refused` is the
    number of rows stamped before `confirmatory_from`, which are dropped.
    Blank cells become NaN, never zero.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, dtype=str, keep_default_na=False)
    df = df.copy()
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    early = df["ts"] < confirmatory_from
    refused = int(early.sum())
    df = df[~early].sort_values("ts").reset_index(drop=True)
    ticks = df[df["row_kind"] == "tick"].reset_index(drop=True)
    gaps = df[df["row_kind"] == "gap"].reset_index(drop=True)
    return {"ticks": ticks, "gaps": gaps, "refused": refused,
            "first_ts": df["ts"].min() if len(df) else None, "last_ts": df["ts"].max() if len(df) else None}


# --- fills -------------------------------------------------------------------

def _book_ok(row) -> bool:
    return bool(row["k_up"] == 1 and ks.valid_book(row["k_yes_bid"], row["k_yes_ask"]))


def _spot_ok(row) -> bool:
    return bool(row["cb_up"] == 1 and row["cb_source"] == "ticker" and np.isfinite(row["cb_price"]))


def _mid(row) -> float:
    return 0.5 * (row["k_yes_bid"] + row["k_yes_ask"])


def fill(side: str, row) -> dict:
    """
    The entry a 1-lot taker gets on this tick: YES at the ask, NO at 1 - bid,
    only if the book is real and at least one contract rests at the touch.
    Returns {"price", "fee"} or None.
    """
    if not _book_ok(row):
        return None
    if side == "yes":
        price, size = row["k_yes_ask"], row["k_yes_ask_sz"]
    else:
        price, size = 1.0 - row["k_yes_bid"], row["k_yes_bid_sz"]
    if not (np.isfinite(size) and size >= CONTRACTS):
        return None
    return {"price": float(price), "fee": float(ks.order_fee(price, CONTRACTS))}


def model_side(p, row, margin: float):
    """Side the model's P(YES) favours net of price and fee by `margin`, or None (both -> None)."""
    if not (np.isfinite(p) and _book_ok(row)):
        return None
    yes = p - row["k_yes_ask"] - float(ks.order_fee(row["k_yes_ask"], CONTRACTS))
    no = (1 - p) - (1 - row["k_yes_bid"]) - float(ks.order_fee(1 - row["k_yes_bid"], CONTRACTS))
    if yes >= margin and no >= margin:
        return None
    if yes >= margin:
        return "yes"
    if no >= margin:
        return "no"
    return None


# --- the five rules -----------------------------------------------------------
# Each takes the market's tick frame (sorted by ts) and the index of the tick
# being considered, and returns (side, fill_row_index) or None. The driver
# applies the window, the "sources up" rule and "first firing only".

def _in_window(secs, window) -> bool:
    return np.isfinite(secs) and window[0] <= secs <= window[1]


def _mid_gated(row) -> bool:
    return _book_ok(row) and MID_GATE[0] <= _mid(row) <= MID_GATE[1]


def rule_h1(m: pd.DataFrame, i: int, gap_ts=None):
    row = m.iloc[i]
    if not (_book_ok(row) and _spot_ok(row) and np.isfinite(row["p_settle"])):
        return None
    t = row["ts"]
    past = m[(m["ts"] <= t - pd.Timedelta(seconds=H1_LOOKBACK_S))]
    if past.empty:
        return None
    p = past.iloc[-1]
    if (t - p["ts"]).total_seconds() > H1_LOOKBACK_S + 2.0:     # the lookback tick must really be ~10s back
        return None
    if gap_ts is not None and ((gap_ts > p["ts"]) & (gap_ts <= t)).any():
        return None
    if not (_book_ok(p) and _spot_ok(p)):
        return None
    move_bp = (row["cb_price"] / p["cb_price"] - 1.0) * 1e4
    if abs(move_bp) < H1_MOVE_BP:
        return None
    if not (row["k_yes_bid"] == p["k_yes_bid"] and row["k_yes_ask"] == p["k_yes_ask"]):
        return None
    side = "yes" if move_bp > 0 else "no"
    if model_side(row["p_settle"], row, H1_MARGIN) != side:
        return None
    if i + 1 >= len(m):
        return None
    nxt = m.iloc[i + 1]
    if (nxt["ts"] - t).total_seconds() > H1_MAX_NEXT_S:
        return None
    return side, i + 1                                           # filled on the NEXT tick


def rule_h2(m, i, gap_ts=None):
    row = m.iloc[i]
    if not (_spot_ok(row) and np.isfinite(row["vol_ann"])):
        return None
    side = model_side(row["p_settle"], row, H2_MARGIN)
    return (side, i) if side else None


def rule_h3(m, i, gap_ts=None):
    row = m.iloc[i]
    if not (_spot_ok(row) and row["book_up"] == 1 and row["book_venues_up"] >= 2 and _mid_gated(row)):
        return None
    if not (np.isfinite(row["book_imb"]) and abs(row["book_imb"]) >= H3_IMB and row["book_persist_s"] >= H3_PERSIST_S):
        return None
    d = row["book_persist_dir"]
    if d == "bullish":
        return "yes", i
    if d == "bearish":
        return "no", i
    return None


def rule_h4(m, i, gap_ts=None):
    row = m.iloc[i]
    if not (_spot_ok(row) and row["agg_up"] == 1 and row["agg_venues_up"] >= 2 and _mid_gated(row)):
        return None
    if row["agg_delta"] >= H4_DELTA:
        return "yes", i
    if row["agg_delta"] <= -H4_DELTA:
        return "no", i
    return None


def rule_h5(m, i, gap_ts=None):
    row = m.iloc[i]
    if not (_spot_ok(row) and row["liq_up"] == 1 and _mid_gated(row)):
        return None
    long_hit = row["liq_long_notional"] >= H5_LIQ_USD
    short_hit = row["liq_short_notional"] >= H5_LIQ_USD
    if long_hit and short_hit:
        return None
    if long_hit:
        return "no", i
    if short_hit:
        return "yes", i
    return None


RULES = {"H1": (rule_h1, WINDOW_S), "H2": (rule_h2, H2_WINDOW_S), "H3": (rule_h3, WINDOW_S),
         "H4": (rule_h4, WINDOW_S), "H5": (rule_h5, WINDOW_S)}


def _hour_bin(ts) -> str:
    h = ts.hour
    for lo, hi in HOUR_BINS:
        if lo <= h < hi:
            return f"{lo:02d}-{hi:02d}"
    return "?"


def trades_for(hyp: str, record: dict) -> pd.DataFrame:
    """One entry per market: the first in-window tick where the rule fires and the fill is possible."""
    rule, window = RULES[hyp]
    ticks, gap_ts = record["ticks"], record["gaps"]["ts"] if len(record["gaps"]) else None
    out = []
    for ticker, m in ticks.groupby("ticker", sort=False):
        if not ticker:
            continue
        m = m.sort_values("ts").reset_index(drop=True)
        for i in range(len(m)):
            row = m.iloc[i]
            if not _in_window(row["secs_to_close"], window):
                continue
            hit = rule(m, i, gap_ts)
            if hit is None:
                continue
            side, j = hit
            f = fill(side, m.iloc[j])
            if f is None:
                continue                     # the rule fired but there was nothing to buy; keep looking
            fr = m.iloc[j]
            out.append({"hyp": hyp, "ticker": ticker, "ts": fr["ts"], "side": side, "price": f["price"],
                        "fee": f["fee"], "secs_to_close": fr["secs_to_close"], "vol_ann": row["vol_ann"],
                        "day": fr["ts"].strftime("%Y-%m-%d"), "hour_bin": _hour_bin(fr["ts"]),
                        "p_settle": row["p_settle"], "mid": _mid(fr), "bid": fr["k_yes_bid"], "ask": fr["k_yes_ask"]})
            break
    cols = ["hyp", "ticker", "ts", "side", "price", "fee", "secs_to_close", "vol_ann", "day", "hour_bin",
            "p_settle", "mid", "bid", "ask"]
    return pd.DataFrame(out, columns=cols)


# --- settlement and statistics -------------------------------------------------

def settle(trades: pd.DataFrame, outcomes: dict) -> dict:
    """
    Joins Kalshi's recorded result. P&L per contract = won (0/1) - price - fee.
    Markets with no result yet are returned as `pending`, never counted.
    """
    t = trades.copy()
    t["result"] = t["ticker"].map(lambda k: outcomes.get(k))
    pending = t[~t["result"].isin(["yes", "no"])]
    t = t[t["result"].isin(["yes", "no"])].copy()
    t["won"] = (t["result"] == t["side"]).astype(float)
    t["pnl"] = t["won"] - t["price"] - t["fee"]
    t["pnl_worse_1c"] = t["won"] - (t["price"] + 0.01) - np.asarray(ks.order_fee(t["price"] + 0.01, CONTRACTS))
    return {"settled": t.reset_index(drop=True), "pending": pending.reset_index(drop=True)}


def permutation_p(won, side, bid, ask, resamples: int = PERMUTATIONS, seed: int = PERMUTATION_SEED) -> float:
    """
    Side-flip null: does the rule's DIRECTION beat a coin flip at the same
    books? Each resample flips every trade's side with probability 1/2; a
    flipped trade pays the other side's price (YES at the ask, NO at 1 - bid)
    and its outcome inverts. Returns the share of resamples whose mean P&L is
    at least the observed one.

    (Shuffling outcomes across trades, the first idea, leaves the mean P&L
    unchanged for flat 1-lot sizing: sum of wins minus sum of costs does not
    depend on which trade won. Amendment 1 in EDGE_HYPOTHESES.md.)
    """
    won = np.asarray(won, dtype=float)
    is_yes = np.asarray(side) == "yes"
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    n = len(won)
    if n == 0:
        return float("nan")
    price_yes, price_no = ask, 1.0 - bid
    fee_yes = np.asarray(ks.order_fee(price_yes, CONTRACTS), dtype=float)
    fee_no = np.asarray(ks.order_fee(price_no, CONTRACTS), dtype=float)
    pnl_as_is = won - np.where(is_yes, price_yes + fee_yes, price_no + fee_no)
    pnl_flipped = (1.0 - won) - np.where(is_yes, price_no + fee_no, price_yes + fee_yes)
    observed = float(pnl_as_is.mean())
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(resamples, n)).astype(bool)
    means = np.where(flips, pnl_flipped, pnl_as_is).mean(axis=1)
    return float(np.mean(means >= observed - 1e-12))


def verdict(stats: dict, z: float = Z, futility: float = FUTILITY_HALF_WIDTH) -> str:
    """The fixed order: inconclusive, passes, rejected, not detected."""
    if stats["n"] == 0 or not np.isfinite(stats["se"]):
        return "inconclusive (no interval)"
    half = z * stats["se"]
    if half > futility:
        return "inconclusive"
    lo, hi = stats["mean"] - half, stats["mean"] + half
    if lo > 0:
        return "passes confirmation"
    if hi < 0:
        return "no edge found (rejected)"
    return "no edge found (not detected)"


def stop_reached(stats: dict, now: datetime, stop_date: datetime = STOP_DATE, stop_n_eff: float = STOP_N_EFF) -> bool:
    return now >= stop_date or (np.isfinite(stats.get("n_eff", np.nan)) and stats["n_eff"] >= stop_n_eff)


def strata(t: pd.DataFrame, vol_edges) -> dict:
    """Mean P&L and n by UTC day, by 6-hour bin, and by vol tercile (edges fixed from all ticks)."""
    out = {}
    out["day"] = t.groupby("day")["pnl"].agg(["count", "mean"]) if len(t) else pd.DataFrame()
    out["hour_bin"] = t.groupby("hour_bin")["pnl"].agg(["count", "mean"]) if len(t) else pd.DataFrame()
    if len(t) and vol_edges is not None and np.all(np.isfinite(vol_edges)) and vol_edges[0] < vol_edges[1]:
        terc = pd.cut(t["vol_ann"], [-np.inf, vol_edges[0], vol_edges[1], np.inf], labels=["low", "mid", "high"])
        out["vol_tercile"] = t.groupby(terc, observed=False)["pnl"].agg(["count", "mean"])
    else:
        out["vol_tercile"] = pd.DataFrame()
    return out


def concentration(t: pd.DataFrame, top: int = 5) -> dict:
    if len(t) == 0:
        return {"top_share": float("nan"), "mean_without_top": float("nan"), "top": top}
    by_market = t.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    total = by_market.sum()
    top_sum = by_market.head(top).sum()
    rest = t[~t["ticker"].isin(by_market.head(top).index)]
    return {"top_share": float(top_sum / total) if total > 0 else float("nan"),
            "mean_without_top": float(rest["pnl"].mean()) if len(rest) else float("nan"), "top": top}


def vol_tercile_edges(ticks: pd.DataFrame):
    v = ticks["vol_ann"].dropna()
    if len(v) < 3:
        return None
    return np.quantile(v, [1 / 3, 2 / 3])


def model_drift(ticks: pd.DataFrame, model_version: str) -> dict:
    """Rows logged under another model version, and the largest gap between logged and recomputed p_settle."""
    other = int((ticks["model_version"] != model_version).sum()) if "model_version" in ticks else 0
    ok = ticks.dropna(subset=["p_settle", "cb_price", "strike", "secs_to_close", "vol_ann"])
    ok = ok[ok["secs_to_close"] > 0]
    if len(ok) == 0:
        return {"other_version_rows": other, "max_abs_gap": float("nan"), "checked": 0}
    recomputed = [kalshi.settlement_probability(r.cb_price, r.strike, r.secs_to_close / 60.0, r.vol_ann)
                  for r in ok.itertuples()]
    gap = float(np.max(np.abs(np.asarray(recomputed) - ok["p_settle"].to_numpy())))
    return {"other_version_rows": other, "max_abs_gap": gap, "checked": int(len(ok))}


def overlap(trades_by_hyp: dict) -> pd.DataFrame:
    """Share of each hypothesis's markets also traded, same side, by each other hypothesis."""
    keys = {h: set(zip(t["ticker"], t["side"])) for h, t in trades_by_hyp.items()}
    rows = []
    for a in trades_by_hyp:
        for b in trades_by_hyp:
            if a < b:
                both = len(keys[a] & keys[b])
                rows.append({"a": a, "b": b, "same_side_markets": both,
                             "share_of_a": both / len(keys[a]) if keys[a] else float("nan"),
                             "share_of_b": both / len(keys[b]) if keys[b] else float("nan")})
    return pd.DataFrame(rows, columns=["a", "b", "same_side_markets", "share_of_a", "share_of_b"])


def h2_scoring(record: dict, outcomes: dict) -> dict:
    """
    Descriptive, not counted: on H2's eligible ticks (one per market, the
    first in-window tick with mid, p_settle and p_point), paired day-block
    differences of Brier score: mid - p_settle, mid - p_point, p_point - p_settle.
    """
    rows = []
    for ticker, m in record["ticks"].groupby("ticker", sort=False):
        res = outcomes.get(ticker)
        if res not in ("yes", "no") or not ticker:
            continue
        m = m.sort_values("ts")
        for r in m.itertuples():
            if _in_window(r.secs_to_close, H2_WINDOW_S) and _book_ok(m.loc[r.Index]) and np.isfinite(r.p_settle) and np.isfinite(r.p_point):
                mid = 0.5 * (r.k_yes_bid + r.k_yes_ask)
                rows.append({"day": r.ts.strftime("%Y-%m-%d"), "y": 1.0 if res == "yes" else 0.0,
                             "mid": mid, "p_settle": r.p_settle, "p_point": r.p_point})
                break
    df = pd.DataFrame(rows, columns=["day", "y", "mid", "p_settle", "p_point"])
    out = {"n": len(df)}
    if len(df) == 0:
        return out
    b = {k: (df[k] - df["y"]) ** 2 for k in ("mid", "p_settle", "p_point")}
    for name, diff in (("mid_minus_settle", b["mid"] - b["p_settle"]), ("mid_minus_point", b["mid"] - b["p_point"]),
                       ("point_minus_settle", b["p_point"] - b["p_settle"])):
        out[name] = ks.block_bootstrap(diff.to_numpy(), df["day"].to_numpy())
    return out


# --- the whole evaluation --------------------------------------------------------

def evaluate(record: dict, outcomes: dict, now: datetime, model_version: str) -> dict:
    """
    Runs every hypothesis once. `now` decides, with each hypothesis's n_eff,
    whether its stopping rule has been reached; before that the verdict is
    labelled interim and is not a result.
    """
    ticks = record["ticks"]
    vol_edges = vol_tercile_edges(ticks)
    results = {}
    trades_by_hyp = {}
    for hyp in HYPOTHESES:
        trades = trades_for(hyp, record)
        s = settle(trades, outcomes)
        t = s["settled"]
        trades_by_hyp[hyp] = t
        stats = ks.block_bootstrap(t["pnl"].to_numpy(), t["day"].to_numpy()) if len(t) else \
            {"n": 0, "mean": float("nan"), "se": float("nan"), "n_eff": 0.0}
        worse = ks.block_bootstrap(t["pnl_worse_1c"].to_numpy(), t["day"].to_numpy()) if len(t) else None
        results[hyp] = {
            "title": TITLES[hyp], "trades": t, "pending": len(s["pending"]), "stats": stats,
            "interval": (stats["mean"] - Z * stats["se"], stats["mean"] + Z * stats["se"]) if stats["n"] else (np.nan, np.nan),
            "verdict": verdict(stats), "stop_reached": stop_reached(stats, now),
            "permutation_p": permutation_p(t["won"], t["side"], t["bid"], t["ask"]) if len(t) else float("nan"),
            "worse_1c": worse, "strata": strata(t, vol_edges), "concentration": concentration(t),
            "sides": t["side"].value_counts().to_dict() if len(t) else {},
        }
    return {"results": results, "overlap": overlap(trades_by_hyp), "drift": model_drift(ticks, model_version),
            "h2_scoring": h2_scoring(record, outcomes), "z": Z, "refused": record["refused"],
            "first_ts": record["first_ts"], "last_ts": record["last_ts"], "ticks": len(ticks),
            "gaps": len(record["gaps"]), "markets": int(ticks["ticker"].replace("", np.nan).nunique()),
            "vol_edges": vol_edges, "now": now}


def _c(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}c"


def report(ev: dict) -> str:
    lines = ["KXBTC15M forward edge hunt (EDGE_HYPOTHESES.md); hold to settlement, 1 contract, taker fills,",
             "fee rounded up per order; family of 5, z = %.3f. Not modelled: queue, impact, size behind the"
             % ev["z"], "touch, latency beyond the recorder's round trip. Every number below is an upper bound.",
             f"Rows: {ev['ticks']} ticks, {ev['gaps']} gaps, {ev['markets']} markets, "
             f"{ev['first_ts']} -> {ev['last_ts']}; {ev['refused']} pre-registration rows refused.", ""]
    for hyp, r in ev["results"].items():
        s = r["stats"]
        lo, hi = r["interval"]
        label = r["verdict"] if r["stop_reached"] else f"INTERIM, NOT A VERDICT (would read: {r['verdict']})"
        lines.append(f"{hyp}: {r['title']}")
        lines.append(f"  n = {s['n']}, n_eff = {s['n_eff']:.0f}, pending = {r['pending']}, sides = {r['sides']}")
        lines.append(f"  mean net P&L/contract = {_c(s['mean'])}, interval [{_c(lo)}, {_c(hi)}], "
                     f"half-width {_c(ev['z'] * s['se'] if s['n'] else np.nan)}")
        lines.append(f"  verdict: {label}")
        w = r["worse_1c"]
        lines.append(f"  side-flip permutation p = {r['permutation_p']:.3f}; +1c worse fills: mean {_c(w['mean']) if w else 'n/a'}; "
                     f"top-{r['concentration']['top']} markets hold {r['concentration']['top_share']:.0%} of P&L, "
                     f"mean without them {_c(r['concentration']['mean_without_top'])}"
                     if s["n"] else "  (no trades)")
        for name in ("hour_bin", "vol_tercile"):
            tbl = r["strata"][name]
            if len(tbl):
                lines.append(f"  by {name}: " + "; ".join(f"{k}: n={int(v['count'])} {_c(v['mean'])}" for k, v in tbl.iterrows()))
        lines.append("")
    if len(ev["overlap"]):
        lines.append("Overlap (markets traded by both, same side):")
        for o in ev["overlap"].itertuples():
            lines.append(f"  {o.a}&{o.b}: {o.same_side_markets} ({o.share_of_a:.0%} of {o.a}, {o.share_of_b:.0%} of {o.b})")
    d = ev["drift"]
    lines.append(f"Model drift: {d['other_version_rows']} rows under another model version; max |logged - recomputed p_settle| "
                 f"= {d['max_abs_gap']:.2e} over {d['checked']} rows.")
    h2 = ev["h2_scoring"]
    if h2.get("n"):
        lines.append(f"H2 scoring (descriptive, not counted), n = {h2['n']}: Brier mid - p_settle = "
                     f"{h2['mid_minus_settle']['mean']:+.4f} (SE {h2['mid_minus_settle']['se']:.4f}); mid - p_point = "
                     f"{h2['mid_minus_point']['mean']:+.4f}; p_point - p_settle = {h2['point_minus_settle']['mean']:+.4f}. "
                     "Positive = the first is worse.")
    return "\n".join(lines)
