"""
Does Binance.US trade flow improve a calibrated P(YES) forecast for KXBTC15M?
A forecasting study, NOT a trading rule. Reusable study code.

The forecasting study (kalshi_forecast_study.py) found that nothing built from
Coinbase minutes, the dashboard's model or the bot's votes beats Kalshi's mid
(held-out Brier 0.13346), and that a fitted combiner was worse than the mid
alone. It had no trade-flow input. This asks one narrow question:

  Flow question -- does adding Binance.US signed trade flow to a forecast make
  it better than THE SAME forecast without flow, and is the result at least
  not worse than the market mid?

No fee, fill or profit appears anywhere here. This module holds the trade
cache, the flow features, the models and the verdict rule, all tested on
hand-built data. `review/kalshi_flow_backtest.py` fetches and runs it.

PRE-REGISTERED DESIGN. Fixed, and committed, before any Binance.US trade had
been fetched (not even a probe of how many there are, which is why the design
has to tolerate a thin venue) and before any model here had been fitted.
Changing anything after seeing held-out results is tuning on them; add a
model instead, count it in the family, and label it as an amendment below.

What the designer HAD seen: REVIEW_FINDINGS.md and the forecasting study's
full held-out output on this same cache, which the same designer ran. So the
baselines' held-out scores, and the fact that train-fitted recalibration
over-sharpened on the held-out months, were known. No flow number was.

  - Limits of the data, stated in the output too. ONE venue: Binance.US
    BTCUSDT, not the three-venue flow the live bot aggregates (Bybit is
    geo-blocked where this was run; Coinbase has no time-ranged trade
    history). Binance.US is a thin venue. It quotes USDT, not USD: flow is a
    dimensionless ratio, so the USDT/USD basis does not enter the features,
    but the venue's own price is not used at all for that reason. No
    order-book data exists historically. Kalshi quotes are one-minute closes,
    so flow is sampled at whole minutes even though trades are timed to the
    millisecond.
  - Data: the study_data/ cache as it stands (30,950 settled contracts closing
    2025-12-15 .. 2026-11-10, their 1-minute Kalshi quotes, Coinbase 1-minute
    candles), plus Binance.US aggTrades for the same span, fetched after this
    commit into study_data/binance_us_trades/, one file per UTC day,
    paginated by aggregate-trade id. An aggTrade is one taker order's fills
    at one price, so "trade count" means aggregate trades.
  - Held-out set. The existing one (the last ks.TEST_FRACTION of contracts,
    closing from 2026-08-04) has been scored for two earlier families and is
    NOT pristine. A pristine set is preferred if one exists: after this
    commit the runner counts settled contracts closing after the cache's last
    close. If there are at least PRISTINE_MIN_CONTRACTS (1,000 -- three rows
    each, so an n_eff above 1,000 is attainable) the cache is extended, every
    cached contract becomes train, and the new contracts are the held-out
    set. Otherwise the existing held-out set is used, and the output says it
    is not pristine and how many new contracts there were. (The cache's last
    close is 2026-11-10 17:30 UTC and this was written the same evening, so
    the count is expected to be a handful.) Either way the set is scored ONCE.
  - Train is everything before the held-out set, split by ks.split_cutoff
    with fc.VALIDATION_FRACTION (25%) into `fit` and `validation`. All tuning
    is fit-on-`fit`, compare-on-`validation` by pooled log loss; the chosen
    setting is refitted on all of train.
  - Rows: kalshi_forecast_study.build_dataset's (one per contract per horizon
    in ks.HORIZONS_MIN, 10 / 5 / 2 minutes before close; its `usable` rule),
    further restricted to rows with a flow reading (`flow_ok`, below). Every
    model and every baseline is fitted and scored on the SAME rows.
  - No look-ahead. Flow at quote time t uses trades with timestamp T <= t
    only (T in milliseconds; t is a whole minute). Side is the TAKER side,
    from exchange_feeds.parse_binance_trade: isBuyerMaker true means the
    taker SOLD (gotcha 17). Trade timestamps are checked for plausibility on
    the way in (gotcha 2): a seconds/milliseconds mix-up raises.
  - Holes are excluded, not imputed (gotcha 10). A row's long window has a
    hole if it reaches outside the cached span, overlaps a missing day, or
    overlaps a break in the aggregate-trade id sequence (ids are consecutive,
    so a jump means trades are missing between the two timestamps). Such rows
    are dropped and counted (`hole`). A window with fewer than
    MIN_TRADES_LONG (10) trades has no usable reading and is dropped and
    counted (`thin`): absent, not neutral. A quiet stretch with consecutive
    ids is real data, not a hole.
  - Windows: SHORT = 5 minutes, LONG = 30 minutes, both [t - w, t] inclusive
    (the live TradeAggregator's rule). Notional = price * quantity.
  - Flow features (FLOW_FEATURES, 7):
      imb_s, imb_l   signed notional imbalance (buy - sell) / (buy + sell) in
                     the short / long window.
      cnt_imb_l      trade-count imbalance (buys - sells) / trades, long.
      large_imb_l    (large buy - large sell notional) / all notional, long;
                     large = a trade of at least LARGE_USD ($10,000).
      net_surge      short-window net notional over what the long window's
                     activity implies for a window that long:
                     (buy_s - sell_s) / (total_l * SHORT / LONG), clipped to
                     +/- SURGE_CLIP (5). Flow normalised by activity.
      div_l          flow-versus-return divergence: imb_l - clip(r / 2, -1, 1)
                     with r the 30-minute Coinbase log return in units of the
                     120-minute realized per-minute sd times sqrt(30). Flow
                     normalised by volatility.
      short_empty    1 if the short window holds no trade (imb_s and
                     net_surge are then 0, and this column tells the model
                     that 0 means "no reading", not "balanced").
    Regime indicators (flow-free): high_vol (120-minute realized vol above
    the median of usable `fit` rows) and us_hours (quote hour 13..20 UTC).
  - Model class: fc.fit_logistic (L2-penalized logistic regression on
    standardized inputs, Newton's method), one fit per horizon, outputs
    clipped as there. L2 chosen on validation from fc.L2_GRID for every model
    but market_base (one input, no penalty).
  - The family: 8 fitted models -- 4 flow models, each with the flow-free
    counterpart that is THE SAME MODEL WITHOUT THE FLOW COLUMNS:
      market_base         logit(mid)
      market_flow         market_base + the 7 flow features
      market_regime_base  logit(mid), high_vol, us_hours
      market_flow_regime  market_regime_base + the 7 flow features + imb_s
                          and imb_l each times high_vol and times us_hours
      model_base          logit(p_asis), mom5, log_vol_ratio
      model_flow          model_base + the 7 flow features
      both_base           logit(mid), logit(p_asis), mom5, log_vol_ratio
      both_flow           both_base + the 7 flow features
    (p_asis, mom5, log_vol_ratio as defined in kalshi_forecast_study.)
    Combining market, model and flow happens only inside these fitted models
    (principle 1), and both_flow is compared held-out with each component.
  - Baselines, not counted as models: `market` (the mid), `asis`
    (settlement_probability as the dashboard computes it), `base_rate` (the
    YES rate of the train rows used, one constant).
  - Metrics, held-out only, pooled over the three horizons, all from existing
    code: ks.brier, ks.log_loss; fc.skill (1 - loss / reference loss, day-
    block bootstrap interval, n_eff); fc.reliability_table (flags buckets
    with n_eff under fc.MIN_BUCKET_N_EFF); sharpness (sd, mean |p - 0.5|);
    ks.calibration_fit / calibration_bootstrap (intercept, slope).
  - Multiple comparisons: FAMILY_TESTS = 60 skill intervals share one
    Bonferroni z (ks.bonferroni_z): each flow model vs market, asis,
    base_rate and its counterpart (4 x 4), each flow-free model vs the three
    baselines (4 x 3), both_flow vs market_flow and vs model_flow (2); 30
    pairs x 2 metrics.
  - Verdict rule. fc.skill calls a model BETTER / WORSE than a reference only
    when the interval excludes zero at that z. FLOW HELPS a model only if
    BOTH its Brier-skill and its log-loss-skill intervals against its own
    flow-free counterpart lie entirely above zero AND neither of its
    intervals against the market lies entirely below zero. Nothing else is a
    claim. A null is reported as a null.
  - Descriptive, not counted: per-horizon scores, the tuning table, fitted
    flow coefficients.
  - Not tested, so not ruled out: multi-venue or deeper-venue flow, order-book
    inputs, sub-minute sampling, windows other than 5 and 30 minutes, a
    large-trade threshold other than $10,000, non-linear models.
"""
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import exchange_feeds
import kalshi_forecast_study as fc
import kalshi_study as ks

# --- pre-registered design (see module docstring) ---------------------------
SYMBOL = "BTCUSDT"
SHORT_MIN = 5
LONG_MIN = 30
MIN_TRADES_LONG = 10
LARGE_USD = 10_000.0
SURGE_CLIP = 5.0
US_HOURS = (13, 21)                 # quote hour h with 13 <= h < 21, UTC
PRISTINE_MIN_CONTRACTS = 1000
FLOW_FEATURES = ("imb_s", "imb_l", "cnt_imb_l", "large_imb_l", "net_surge", "div_l", "short_empty")
REGIME_FEATURES = ("high_vol", "us_hours")
INTERACTIONS = (("imb_s", "high_vol"), ("imb_l", "high_vol"), ("imb_s", "us_hours"), ("imb_l", "us_hours"))

COUNTERPART = {"market_flow": "market_base", "market_flow_regime": "market_regime_base",
               "model_flow": "model_base", "both_flow": "both_base"}
FLOW_MODELS = tuple(COUNTERPART)
BASE_MODELS = tuple(dict.fromkeys(COUNTERPART.values()))
MODELS = tuple(m for pair in COUNTERPART.items() for m in pair[::-1])
BASELINES = fc.BASELINES
METRICS = fc.METRICS
COMPONENT_COMPARISONS = (("both_flow", "market_flow"), ("both_flow", "model_flow"))
COMPARISONS = tuple(
    [(m, b, "baseline") for m in MODELS for b in BASELINES]
    + [(m, COUNTERPART[m], "counterpart") for m in FLOW_MODELS]
    + [(m, c, "component") for m, c in COMPONENT_COMPARISONS])
FAMILY_TESTS = len(COMPARISONS) * len(METRICS)

TRADE_DIR = "binance_us_trades"
META_FILE = "flow_meta.json"
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
PAGE_LIMIT = 1000
_MIN_PLAUSIBLE_MS = 1_577_836_800_000        # 2020-01-01: anything earlier is a units bug, not a trade
WEIGHT_SOFT_LIMIT = 800                      # of Binance.US's 1200 request weight per minute


# --- time (gotcha 2) -------------------------------------------------------------

def epoch_ms(times) -> np.ndarray:
    """
    Epoch milliseconds from datetimes of ANY resolution: forced to
    datetime64[ns] first, because pandas does not reliably produce
    nanoseconds and dividing a seconds-resolution integer by 1e6 silently
    lands in 1970.
    """
    ns = np.asarray(pd.to_datetime(times).values).astype("datetime64[ns]").astype("int64")
    return ns // 1_000_000


def check_plausible_ms(ts_ms, lo_ms: int = None, hi_ms: int = None, what: str = "trade timestamps") -> None:
    """
    Fail loudly if any timestamp cannot be a millisecond epoch of a real trade
    (before 2020 -- which is what seconds or a 1970 date look like -- or more
    than a day ahead of now), or falls outside [lo_ms, hi_ms] when given.
    """
    t = np.asarray(ts_ms, dtype=np.int64)
    if len(t) == 0:
        return
    ceiling = int(time.time() * 1000) + DAY_MS
    if t.min() < _MIN_PLAUSIBLE_MS or t.max() > ceiling:
        raise ValueError(f"implausible {what}: {int(t.min())} .. {int(t.max())} is not a millisecond epoch "
                         "of a real trade (seconds passed as milliseconds, or a wrong datetime resolution?)")
    if (lo_ms is not None and t.min() < lo_ms) or (hi_ms is not None and t.max() > hi_ms):
        raise ValueError(f"{what} {int(t.min())} .. {int(t.max())} fall outside the requested range {lo_ms} .. {hi_ms}")


def day_start_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) * 1000


def day_of_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms // 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# --- the trade cache -------------------------------------------------------------

def binance_get(session, params: dict, retries: int = 8, state: dict = None):
    """
    One aggTrades page from Binance.US's public REST API. Backs off on 429 /
    418 / 5xx (honouring Retry-After) and slows down when the used-weight
    header nears the limit, so the fetch stays inside the published rate limit.
    """
    url = f"{exchange_feeds._BINANCE_US_REST_BASE}/api/v3/aggTrades"
    delay = 2.0
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code in (418, 429) or resp.status_code >= 500:
            if attempt == retries - 1:
                resp.raise_for_status()
            time.sleep(float(resp.headers.get("Retry-After", delay)))
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
        used = int(resp.headers.get("x-mbx-used-weight-1m", 0) or 0)
        if state is not None:
            state["used_weight"] = used
        if used > WEIGHT_SOFT_LIMIT:
            time.sleep(5.0)
        return resp.json()
    raise RuntimeError("unreachable")


def _empty_trades() -> dict:
    return {"id": np.zeros(0, np.int64), "ts": np.zeros(0, np.int64), "price": np.zeros(0), "qty": np.zeros(0),
            "buy": np.zeros(0, bool)}


def parse_page(page: list) -> dict:
    """
    Raw aggTrades -> arrays. Side goes through exchange_feeds.parse_binance_trade,
    the one place that knows `m` (isBuyerMaker) true means the TAKER sold.
    """
    out = {"id": [], "ts": [], "price": [], "qty": [], "buy": []}
    for raw in page:
        tr = exchange_feeds.parse_binance_trade(raw)
        if tr is None or "a" not in raw or "T" not in raw:
            raise ValueError(f"unparseable aggTrade: {raw!r}")
        out["id"].append(int(raw["a"]))
        out["ts"].append(int(raw["T"]))
        out["price"].append(tr["price"])
        out["qty"].append(tr["size"])
        out["buy"].append(tr["side"] == "buy")
    return {"id": np.array(out["id"], np.int64), "ts": np.array(out["ts"], np.int64),
            "price": np.array(out["price"], float), "qty": np.array(out["qty"], float),
            "buy": np.array(out["buy"], bool)}


def fetch_trades(get, start_ms: int, end_ms: int, symbol: str = SYMBOL) -> dict:
    """
    Every aggTrade with start_ms <= T <= end_ms. `get(params)` returns one raw
    page. The first trade is found by time (one hour at a time, the widest
    startTime/endTime span the API promises to accept); everything after is
    paged by `fromId`, so no trade is skipped when many share a millisecond at
    a page boundary -- which a time cursor would do.
    """
    page, h = [], start_ms
    while h <= end_ms and not page:
        page = get({"symbol": symbol, "startTime": h, "endTime": min(h + HOUR_MS - 1, end_ms), "limit": PAGE_LIMIT})
        h += HOUR_MS
    parts = []
    while page:
        p = parse_page(page)
        parts.append(p)
        if p["ts"][-1] > end_ms:
            break
        page = get({"symbol": symbol, "fromId": int(p["id"][-1]) + 1, "limit": PAGE_LIMIT})
    if not parts:
        return _empty_trades()
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    check_plausible_ms(out["ts"])                 # before the range filter, which would quietly drop a units bug
    keep = (out["ts"] >= start_ms) & (out["ts"] <= end_ms)
    out = {k: v[keep] for k, v in out.items()}
    if len(out["id"]) and (np.diff(out["id"]) <= 0).any():
        raise ValueError("aggTrade ids are not strictly increasing")
    check_plausible_ms(out["ts"], start_ms, end_ms)
    return out


def _day_path(cache_dir: str, day: str) -> str:
    return os.path.join(cache_dir, TRADE_DIR, f"{day}.npz")


def save_day(cache_dir: str, day: str, trades: dict, end_ms: int) -> None:
    """One UTC day's trades, written whole (tmp + rename), with the last millisecond it covers."""
    os.makedirs(os.path.join(cache_dir, TRADE_DIR), exist_ok=True)
    path = _day_path(cache_dir, day)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, end_ms=np.int64(end_ms), **trades)
    os.replace(tmp, path)


def load_day(cache_dir: str, day: str):
    """(trades, covered_end_ms) for a cached day, or None if absent or unreadable (a torn file is refetched)."""
    path = _day_path(cache_dir, day)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in ("id", "ts", "price", "qty", "buy")}, int(z["end_ms"])
    except Exception:
        return None


def days_between(start_ms: int, end_ms: int) -> list:
    first = start_ms - start_ms % DAY_MS
    return [day_of_ms(t) for t in range(first, end_ms + 1, DAY_MS)]


def fetch_missing_days(get, cache_dir: str, start_ms: int, end_ms: int, progress=None, workers: int = 1) -> int:
    """
    Fetches every UTC day touching [start_ms, end_ms] that is not cached
    through the end it needs (the day's last millisecond, or end_ms on the
    final day), one file per day, so an interrupted run resumes at the next
    missing day. Returns how many days were fetched.
    """
    from concurrent.futures import ThreadPoolExecutor
    todo = []
    for day in days_between(start_ms, end_ms):
        d0 = day_start_ms(day)
        hi = min(d0 + DAY_MS - 1, end_ms)
        have = load_day(cache_dir, day)
        if have is None or have[1] < hi:                         # absent, torn, or cached before the day was over
            todo.append((day, d0, hi))
    done = [0]

    def one(item):
        day, d0, hi = item
        save_day(cache_dir, day, fetch_trades(get, d0, hi), hi)
        done[0] += 1
        if progress:
            progress(done[0], len(todo), day)
    if workers <= 1:
        for item in todo:
            one(item)
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, todo))
    return len(todo)


def load_trades(cache_dir: str, start_ms: int, end_ms: int) -> dict:
    """
    Cached trades for [start_ms, end_ms], time-ordered, with `holes`: the
    closed millisecond intervals in which trades may be missing --
      - before the first and after the last cached millisecond of the span,
      - every missing or short day,
      - every break in the aggregate-trade id sequence, as the interval
        between the two trades either side of it (ids are consecutive, so a
        jump means trades are missing there).
    """
    parts, holes, covered_to = [], [], start_ms - 1
    for day in days_between(start_ms, end_ms):
        d0 = day_start_ms(day)
        have = load_day(cache_dir, day)
        if have is None:
            continue
        if d0 > covered_to + 1:
            holes.append((covered_to + 1, d0 - 1))
        parts.append(have[0])
        covered_to = max(covered_to, have[1])
    if covered_to < end_ms:
        holes.append((covered_to + 1, end_ms))
    t = {k: np.concatenate([p[k] for p in parts]) for k in _empty_trades()} if parts else _empty_trades()
    order = np.lexsort((t["id"], t["ts"]))
    t = {k: v[order] for k, v in t.items()}
    check_plausible_ms(t["ts"])
    by_id = np.argsort(t["id"], kind="stable")
    ids, ts = t["id"][by_id], t["ts"][by_id]
    jump = np.flatnonzero(np.diff(ids) > 1)
    id_gaps = [(int(min(ts[i], ts[i + 1])), int(max(ts[i], ts[i + 1]))) for i in jump]
    t["missing_ids"] = int((np.diff(ids)[jump] - 1).sum()) if len(jump) else 0
    t["duplicate_ids"] = int((np.diff(ids) == 0).sum())
    t["id_gaps"] = id_gaps
    t["holes"] = sorted(holes + id_gaps)
    return t


# --- flow features ---------------------------------------------------------------

def windows_with_hole(holes: list, lo_ms, hi_ms) -> np.ndarray:
    """True where the closed window [lo, hi] touches any closed hole interval."""
    lo_ms, hi_ms = np.asarray(lo_ms, dtype=np.int64), np.asarray(hi_ms, dtype=np.int64)
    if not holes:
        return np.zeros(len(lo_ms), dtype=bool)
    h = np.array(sorted(holes), dtype=np.int64)
    reach = np.maximum.accumulate(h[:, 1])                       # the furthest any hole starting this early extends
    k = np.searchsorted(h[:, 0], hi_ms, side="right")            # holes starting at or before the window's end
    return (k > 0) & (reach[np.maximum(k - 1, 0)] >= lo_ms)


def flow_sums(trades: dict, lo_ms, hi_ms) -> dict:
    """
    Per closed window [lo, hi]: buy / sell notional, buy / sell trade counts,
    and buy / sell notional in trades of at least LARGE_USD. Prefix sums and a
    binary search; an empty window is exactly zero (its count decides, so
    float residue from the subtraction cannot pose as a reading).
    """
    ts = trades["ts"]
    notional = trades["price"] * trades["qty"]
    buy = trades["buy"]
    large = notional >= LARGE_USD - 1e-9
    i0 = np.searchsorted(ts, np.asarray(lo_ms, dtype=np.int64), side="left")
    i1 = np.searchsorted(ts, np.asarray(hi_ms, dtype=np.int64), side="right")

    def window(values):
        c = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
        return c[i1] - c[i0]
    n_buy = window(buy.astype(float)).round().astype(np.int64)
    n_sell = window((~buy).astype(float)).round().astype(np.int64)
    out = {"n_buy": n_buy, "n_sell": n_sell,
           "buy": np.where(n_buy > 0, window(np.where(buy, notional, 0.0)), 0.0),
           "sell": np.where(n_sell > 0, window(np.where(~buy, notional, 0.0)), 0.0),
           "large_buy": window(np.where(buy & large, notional, 0.0)),
           "large_sell": window(np.where(~buy & large, notional, 0.0))}
    n_large_buy, n_large_sell = window((buy & large).astype(float)).round(), window((~buy & large).astype(float)).round()
    out["large_buy"] = np.where(n_large_buy > 0, out["large_buy"], 0.0)
    out["large_sell"] = np.where(n_large_sell > 0, out["large_sell"], 0.0)
    return {k: np.maximum(v, 0) for k, v in out.items()}


def flow_features(trades: dict, quote_ts) -> pd.DataFrame:
    """
    The trade-only flow features at each quote time (epoch SECONDS), plus
    `n_long`, `hole` and `thin`. Only trades with T <= quote time are read.
    div_l needs prices and is added by add_flow.
    """
    t_ms = np.asarray(quote_ts, dtype=np.int64) * 1000
    check_plausible_ms(t_ms, what="quote times")
    s = flow_sums(trades, t_ms - SHORT_MIN * 60_000, t_ms)
    lg = flow_sums(trades, t_ms - LONG_MIN * 60_000, t_ms)
    n_s, n_l = s["n_buy"] + s["n_sell"], lg["n_buy"] + lg["n_sell"]
    tot_s, tot_l = s["buy"] + s["sell"], lg["buy"] + lg["sell"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "imb_s": np.where(n_s > 0, (s["buy"] - s["sell"]) / tot_s, 0.0),
            "imb_l": np.where(n_l > 0, (lg["buy"] - lg["sell"]) / tot_l, np.nan),
            "cnt_imb_l": np.where(n_l > 0, (lg["n_buy"] - lg["n_sell"]) / n_l, np.nan),
            "large_imb_l": np.where(n_l > 0, (lg["large_buy"] - lg["large_sell"]) / tot_l, np.nan),
            "net_surge": np.where(n_l > 0, np.clip((s["buy"] - s["sell"]) / (tot_l * SHORT_MIN / LONG_MIN),
                                                   -SURGE_CLIP, SURGE_CLIP), np.nan),
            "short_empty": (n_s == 0).astype(float),
            "n_long": n_l,
        })
    out["hole"] = windows_with_hole(trades.get("holes", []), t_ms - LONG_MIN * 60_000, t_ms)
    out["thin"] = ~out["hole"] & (n_l < MIN_TRADES_LONG)
    return out


def add_flow(dataset: pd.DataFrame, trades: dict, grid) -> pd.DataFrame:
    """
    `dataset` (fc.build_dataset's) with the flow features, the two regime
    inputs' raw material, `flow_ok` and `use` (usable AND flow_ok: the rows
    every model and baseline is fitted and scored on).
    """
    d = dataset.reset_index(drop=True).copy()
    f = flow_features(trades, d["quote_ts"].to_numpy())
    for c in f.columns:
        d[c] = f[c].to_numpy()
    close = grid.cols["close"]
    idx = (d["quote_ts"].to_numpy(dtype=np.int64) - 60 - grid.t0) // 60
    ok = (idx >= LONG_MIN) & (idx < len(close))
    safe = np.where(ok, idx, LONG_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_minute = d["vol"].to_numpy(dtype=float) / math.sqrt(fc.MINUTES_PER_YEAR)
        r = np.where(ok, np.log(close[safe] / close[safe - LONG_MIN]), np.nan) / (per_minute * math.sqrt(LONG_MIN))
        d["div_l"] = d["imb_l"] - np.clip(r / 2.0, -1.0, 1.0)
    d["us_hours"] = (((d["quote_ts"] // 3600) % 24 >= US_HOURS[0]) & ((d["quote_ts"] // 3600) % 24 < US_HOURS[1])).astype(float)
    finite = np.isfinite(d[list(FLOW_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    d["flow_ok"] = ~d["hole"] & ~d["thin"] & finite
    d["use"] = d["usable"] & d["flow_ok"]
    return d


def assign_splits(dataset: pd.DataFrame, pristine_after: int = None) -> pd.DataFrame:
    """
    `split` = fit / validation / test. pristine_after None: the existing
    held-out set (fc.build_dataset's split, left as it is). Otherwise every
    contract closing after `pristine_after` is the held-out set and all earlier
    ones are train, the last fc.VALIDATION_FRACTION of them the validation slice.
    """
    if pristine_after is None:
        return dataset
    d = dataset.copy()
    closes = np.unique(d["close_ts"].to_numpy(dtype=np.int64))
    val_cut = ks.split_cutoff(closes[closes <= pristine_after], fc.VALIDATION_FRACTION)
    d["split"] = np.where(d["close_ts"] > pristine_after, "test", np.where(d["close_ts"] >= val_cut, "validation", "fit"))
    return d


# --- models --------------------------------------------------------------------------

def model_columns(model: str) -> list:
    """The inputs of each model, by name. A flow model's are its counterpart's plus flow columns, and nothing else."""
    base = {"market_base": ["logit_mid"],
            "market_regime_base": ["logit_mid", "high_vol", "us_hours"],
            "model_base": ["logit_asis", "mom5", "log_vol_ratio"],
            "both_base": ["logit_mid", "logit_asis", "mom5", "log_vol_ratio"]}
    if model in base:
        return list(base[model])
    if model not in COUNTERPART:
        raise ValueError(f"unknown model {model!r}")
    cols = base[COUNTERPART[model]] + list(FLOW_FEATURES)
    if model == "market_flow_regime":
        cols += [f"{a}_x_{b}" for a, b in INTERACTIONS]
    return cols


def design_frame(d: pd.DataFrame, vol_threshold: float) -> pd.DataFrame:
    """Every input any model uses, as named columns. `vol_threshold` comes from train (tune_and_fit)."""
    x = pd.DataFrame({"logit_mid": fc.logit(d["mid"]), "logit_asis": fc.logit(d["p_asis"])}, index=d.index)
    for c in ("mom5", "log_vol_ratio", "us_hours") + FLOW_FEATURES:
        x[c] = d[c].to_numpy(dtype=float)
    x["high_vol"] = (d["vol"].to_numpy(dtype=float) > vol_threshold).astype(float)
    for a, b in INTERACTIONS:
        x[f"{a}_x_{b}"] = x[a] * x[b]
    return x


def _fit_one(train: pd.DataFrame, model: str, vol_threshold: float, l2: float) -> dict:
    per_h = {}
    for h in ks.HORIZONS_MIN:
        g = train[train["horizon"] == h]
        per_h[h] = fc.fit_logistic(design_frame(g, vol_threshold)[model_columns(model)].to_numpy(), g["outcome"], l2)
    return {"model": model, "l2": l2, "vol_threshold": vol_threshold, "per_horizon": per_h}


def predict_one(fitted: dict, d: pd.DataFrame) -> np.ndarray:
    out = np.full(len(d), np.nan)
    for h, m in fitted["per_horizon"].items():
        rows = (d["horizon"] == h).to_numpy()
        if rows.any():
            x = design_frame(d[rows], fitted["vol_threshold"])[model_columns(fitted["model"])].to_numpy()
            out[rows] = fc.predict_logistic(m, x)
    return out


def tune_and_fit(dataset: pd.DataFrame) -> dict:
    """
    The whole train-side procedure on `use` rows; reads no row whose split is
    "test". Returns {"models", "l2", "vol_threshold", "base_rate", "tuning"}.
    """
    d = dataset[dataset["use"] & (dataset["split"] != "test")]
    fit, val = d[d["split"] == "fit"], d[d["split"] == "validation"]
    vol_threshold = float(fit["vol"].median())
    tuning, l2 = [], {}
    for model in MODELS:
        grid = (fc.PLATT_L2,) if model == "market_base" else fc.L2_GRID
        best = None
        for lam in grid:
            p = predict_one(_fit_one(fit, model, vol_threshold, lam), val)
            loss = float(ks.log_loss(p, val["outcome"]).mean())
            tuning.append({"model": model, "l2": lam, "val_log_loss": loss})
            if best is None or loss < best[0] - 1e-12:
                best = (loss, lam)
        l2[model] = best[1]
    models = {m: _fit_one(d, m, vol_threshold, l2[m]) for m in MODELS}
    return {"models": models, "l2": l2, "vol_threshold": vol_threshold, "base_rate": float(d["outcome"].mean()),
            "tuning": pd.DataFrame(tuning)}


def predictions(fitted: dict, d: pd.DataFrame) -> pd.DataFrame:
    """Every baseline's and model's P(YES) for the rows of `d`, side by side, never combined."""
    out = d[["ticker", "close_ts", "day", "horizon", "outcome"]].copy()
    out["market"] = d["mid"].to_numpy(dtype=float)
    out["asis"] = d["p_asis"].to_numpy(dtype=float)
    out["base_rate"] = fitted["base_rate"]
    for m in MODELS:
        out[m] = predict_one(fitted["models"][m], d)
    return out


def flow_coefficients(fitted: dict) -> pd.DataFrame:
    """Standardized coefficients of the flow columns in each flow model, per horizon. Descriptive."""
    rows = []
    for m in FLOW_MODELS:
        cols = model_columns(m)
        for h, fit in fitted["models"][m]["per_horizon"].items():
            for c, b in zip(cols, fit["beta"][1:]):
                if c in FLOW_FEATURES or "_x_" in c:
                    rows.append({"model": m, "horizon": h, "input": c, "beta_std": float(b)})
    return pd.DataFrame(rows)


# --- metrics (all existing code, composed) ---------------------------------------------

def scorecard(pred: pd.DataFrame, calibration_resamples: int = 300) -> pd.DataFrame:
    """Per forecast: Brier, log loss, sharpness and calibration, from ks.brier / log_loss / calibration_*."""
    y, days = pred["outcome"].to_numpy(dtype=float), pred["day"].to_numpy()
    rows = []
    for name in BASELINES + MODELS:
        p = pred[name].to_numpy(dtype=float)
        row = {"forecast": name, "kind": "baseline" if name in BASELINES else "flow" if name in FLOW_MODELS else "flow-free",
               "n": len(p), "brier": float(ks.brier(p, y).mean()), "log_loss": float(ks.log_loss(p, y).mean()),
               "sharp_sd": float(p.std()), "sharp_abs": float(np.abs(p - 0.5).mean()),
               "cal_a": float("nan"), "cal_b": float("nan"), "cal_a_se": float("nan"), "cal_b_se": float("nan")}
        if np.ptp(p) > 1e-9:
            row["cal_a"], row["cal_b"] = ks.calibration_fit(p, y)
            try:
                c = ks.calibration_bootstrap(p, y, days, resamples=calibration_resamples)
                row.update({"cal_a_se": c["a_se"], "cal_b_se": c["b_se"]})
            except np.linalg.LinAlgError:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def skill_table(pred: pd.DataFrame, z: float) -> pd.DataFrame:
    """Every counted skill interval (FAMILY_TESTS rows), each from fc.skill."""
    y, days = pred["outcome"].to_numpy(dtype=float), pred["day"].to_numpy()
    rows = []
    for model, ref, kind in COMPARISONS:
        for metric in METRICS:
            f = fc.LOSSES[metric]
            rows.append({"model": model, "reference": ref, "kind": kind, "metric": metric,
                         **fc.skill(f(pred[model], y), f(pred[ref], y), days, z)})
    return pd.DataFrame(rows)


def flow_verdicts(skills: pd.DataFrame) -> pd.DataFrame:
    """
    The pre-registered rule, per flow model: FLOW HELPS only if both metrics'
    intervals against the flow-free counterpart lie above zero AND neither
    interval against the market lies below zero.
    """
    rows = []
    for m in FLOW_MODELS:
        vs_cp = skills[(skills.model == m) & (skills.kind == "counterpart")]
        vs_mkt = skills[(skills.model == m) & (skills.reference == "market")]
        beats = bool((vs_cp.verdict == "BETTER").all()) and len(vs_cp) == len(METRICS)
        worse_than_market = bool((vs_mkt.verdict == "WORSE").any())
        hurts = bool((vs_cp.verdict == "WORSE").any())
        verdict = ("FLOW HELPS" if beats and not worse_than_market else
                   "better than its counterpart but worse than the market: does not count" if beats else
                   "flow makes it worse" if hurts else "no detectable effect")
        rows.append({"model": m, "counterpart": COUNTERPART[m], "beats_counterpart": beats,
                     "worse_than_market": worse_than_market, "helps": beats and not worse_than_market, "verdict": verdict})
    return pd.DataFrame(rows)


# --- the pristine-set check ------------------------------------------------------------

def count_new_settled(session, after_close_ts: int) -> int:
    """How many cleanly settled KXBTC15M contracts close after `after_close_ts` (Kalshi's public listing)."""
    n = 0
    for m in ks.iter_markets(session, "/markets", {"series_ticker": ks.SERIES, "status": "settled",
                                                    "min_close_ts": int(after_close_ts) + 1}):
        p = ks.parse_settled_market(m)
        if p is not None and p["close_ts"] > after_close_ts:
            n += 1
    return n


def read_meta(cache_dir: str) -> dict:
    path = os.path.join(cache_dir, META_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_meta(cache_dir: str, meta: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, META_FILE)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(path + ".tmp", path)
