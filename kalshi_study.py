"""
Is there a tradeable edge in Kalshi's KXBTC15M contracts? Reusable study code.

Two questions, answered on settled contracts:
  Step 2 -- is the market itself mispriced? Bucket contracts by the quoted
            mid a fixed number of minutes before close and compare it with
            how often YES actually settled, net of the spread and the fee.
  Step 3 -- does kalshi.settlement_probability, fed Coinbase prices, beat
            the market mid? (see model_observations / score_probabilities)

This module holds the fetching, caching and arithmetic, all tested.
`review/kalshi_market_study.py` is the one-off analysis that runs it and
prints the tables written up in REVIEW_FINDINGS.md.

PRE-REGISTERED DESIGN. These constants were fixed, and committed, before any
result was looked at. Changing one after seeing results is tuning on the
held-out data; add a variant instead and count it.
  - Split: contracts sorted by close time, the last TEST_FRACTION (30%) held
    out. The cutoff is the close time at the 70th percentile of all settled
    contracts in the cache.
  - One observation per contract per horizon (HORIZONS_MIN), from the quote
    at exactly that minute. Minutes inside a contract are correlated and are
    not used as extra samples.
  - A quote counts only with a real two-sided book: 0 < bid < ask < 1.
    Kalshi shows bid 0 / ask 0.999-1 before anyone has quoted.
  - Costs: YES is bought at the ask, NO at 1 - bid, and each pays Kalshi's
    taker fee (taker_fee). No resting-order fills are assumed.
  - Confidence intervals: a block bootstrap by UTC day (contracts on the
    same day share a volatility regime), as mean +/- z * bootstrap SE, with z
    Bonferroni-corrected for the number of cells tested.
  - Selection: a cell is "selected" on train only if its Bonferroni interval
    lies entirely above zero; only selected cells are then scored on test,
    with a second Bonferroni correction over the number selected.

Fees, verified 2026-11-07 against Kalshi's own fee schedule
(kalshi.com/docs/kalshi-fee-schedule.pdf, "Last updated and effective: July
7, 2026"): "fees = round up(M x 0.07 x C x P x (1-P))", rounded "such that
the fee + positionCost is rounded to a centicent"; maker fees apply only to
series listed with a maker multiplier; "There is no settlement fee."
KXBTC15M is not on the non-standard list, and GET /series/KXBTC15M reports
fee_type "quadratic" (no maker fee) with fee_multiplier 1.

Data: the live API lists settled markets from 2026-09-01; older ones are
under /historical/markets (cutoff from GET /historical/cutoff). The batch
endpoint GET /markets/candlesticks serves both, but caps one request at
10,000 candlesticks counted as markets x minutes in the requested range.
"""
import json
import math
import os
import time
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests

import kalshi
import kalshi_api

SERIES = kalshi_api.SERIES_TICKER

# --- pre-registered design (see module docstring) ---------------------------
HORIZONS_MIN = (10, 5, 2)
TEST_FRACTION = 0.30
BUCKET_EDGES = tuple(round(0.1 * i, 1) for i in range(11))     # mid deciles
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261107
TAKER_FEE_RATE = 0.07
FEE_MULTIPLIER = 1.0

# Step 3
VOL_LOOKBACK_MIN = 120              # same window as the live calculator
MIN_VOL_CANDLES = 110               # of the 121 one-minute candles the vol window needs
SPOT_SOURCES = ("coinbase", "basis_adjusted")
EDGE_MARGINS = (0.0, 0.02, 0.05)   # extra edge, beyond costs, required to trade
LOGLOSS_EPS = 1e-4

MAX_BATCH_CANDLES = 10_000          # Kalshi's per-request cap (its own error message)
CACHE_DIR = "study_data"


# --- costs -------------------------------------------------------------------

def taker_fee(price, contracts: float = 1.0, multiplier: float = FEE_MULTIPLIER, rate: float = TAKER_FEE_RATE):
    """
    Kalshi's taker fee in dollars for `contracts` contracts at `price`, before
    the centicent round-up (worth at most $0.0001 per order, so it is left out
    of per-contract expected values). 1.75 cents a contract at 50 cents, 0.63 at
    10 or 90 cents. Works elementwise on arrays.
    """
    p = np.asarray(price, dtype=float)
    return multiplier * rate * contracts * p * (1.0 - p)


def ev_buy_yes(outcome, ask):
    """Profit per contract of buying YES at the ask: pays 1 on YES."""
    ask = np.asarray(ask, dtype=float)
    return np.asarray(outcome, dtype=float) - ask - taker_fee(ask)


def ev_buy_no(outcome, bid):
    """Profit per contract of buying NO at 1 - bid (selling YES at the bid): pays 1 on NO."""
    cost = 1.0 - np.asarray(bid, dtype=float)
    return (1.0 - np.asarray(outcome, dtype=float)) - cost - taker_fee(cost)


def valid_book(bid, ask):
    """A real two-sided quote: 0 < bid < ask < 1 (elementwise)."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    with np.errstate(invalid="ignore"):
        return (bid > 0) & (ask < 1) & (bid < ask)


# --- parsing raw Kalshi payloads --------------------------------------------

def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_settled_market(m: dict):
    """
    The fields the study needs from a raw market record, or None if it didn't
    settle cleanly yes/no. `floor_strike` is the 60-second BRTI average before
    the window opened and `expiration_value` the one before it closed
    (rules_primary), so both double as index readings for Step 3.
    """
    result = m.get("result")
    if result not in ("yes", "no"):
        return None
    strike = _num(m.get("floor_strike"))
    if strike is None or not m.get("open_time") or not m.get("close_time"):
        return None
    return {
        "ticker": m["ticker"],
        "open_ts": _ts(m["open_time"]),
        "close_ts": _ts(m["close_time"]),
        "strike": strike,
        "expiration_value": _num(m.get("expiration_value")),
        "outcome": 1 if result == "yes" else 0,
    }


def _close_of(side: dict):
    if not side:
        return None
    return _num(side.get("close_dollars", side.get("close")))


def parse_candles(candlesticks: list) -> dict:
    """
    {end_period_ts: (yes_bid, yes_ask, last_price)} from a market's 1-minute
    candlesticks. A candle ending at t covers (t - 60, t], so its *closing*
    bid/ask is the quote standing at t. Accepts both the `close_dollars` keys
    the batch endpoint returns and the bare `close` of the per-market
    historical endpoint.
    """
    out = {}
    for c in candlesticks or []:
        out[int(c["end_period_ts"])] = (_close_of(c.get("yes_bid")), _close_of(c.get("yes_ask")),
                                         _close_of(c.get("price")))
    return out


def quote_at(candles: dict, ts: int):
    """The (bid, ask, last) standing at `ts`, or None when no candle ends exactly then."""
    return candles.get(int(ts))


# --- observations, split, statistics ----------------------------------------

def observations(markets: list, candles_by_ticker: dict, horizons=HORIZONS_MIN) -> pd.DataFrame:
    """
    One row per (contract, horizon): the quote `h` minutes before close and
    the settled outcome. `valid` marks a real two-sided book; rows without a
    candle at that exact minute are kept with NaN quotes (and valid False) so
    the drop count can be reported rather than hidden.
    """
    rows = []
    for m in markets:
        c = candles_by_ticker.get(m["ticker"], {})
        for h in horizons:
            t = m["close_ts"] - 60 * h
            q = quote_at(c, t)
            bid, ask, last = q if q else (None, None, None)
            rows.append({**m, "horizon": h, "quote_ts": t, "bid": bid, "ask": ask, "last": last})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("bid", "ask", "last"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid"] = valid_book(df["bid"].to_numpy(), df["ask"].to_numpy())
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["day"] = pd.to_datetime(df["close_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


def split_cutoff(close_ts, test_fraction: float = TEST_FRACTION) -> int:
    """
    Close time from which contracts are held out: the first contract in the
    last `test_fraction` of contracts sorted by close time.
    """
    s = np.sort(np.unique(np.asarray(close_ts, dtype=np.int64)))
    if len(s) == 0:
        raise ValueError("no contracts to split")
    k = int(math.floor(len(s) * (1 - test_fraction)))
    return int(s[min(k, len(s) - 1)])


def bonferroni_z(tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """Two-sided normal critical value with alpha split over `tests` tests."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(int(tests), 1)))


def block_bootstrap(values, blocks, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Mean of `values` and its block-bootstrap standard error, resampling whole
    blocks (days) with replacement. `n_eff` is n times the ratio of the iid
    variance of the mean to the bootstrap one: how many independent
    observations the data is worth once same-day correlation is counted.
    """
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "n_eff": 0.0}
    mean = float(v.mean())
    labels, inv = np.unique(b, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(labels))
    counts = np.bincount(inv, minlength=len(labels)).astype(float)
    d = len(labels)
    if d < 2:
        return {"n": n, "mean": mean, "se": float("nan"), "n_eff": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d, size=(resamples, d))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    se = float(means.std(ddof=1))
    iid_var = float(v.var(ddof=1)) / n if n > 1 else float("nan")
    n_eff = n * iid_var / se ** 2 if se > 0 else float("nan")
    return {"n": n, "mean": mean, "se": se, "n_eff": float(n_eff)}


def interval(stats: dict, z: float):
    return stats["mean"] - z * stats["se"], stats["mean"] + z * stats["se"]


def bucket_of(mid, edges=BUCKET_EDGES):
    """Index of the [edge_i, edge_{i+1}) bucket each mid falls in (the top edge is inclusive)."""
    m = np.asarray(mid, dtype=float)
    idx = np.searchsorted(np.asarray(edges), m, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def bucket_table(obs: pd.DataFrame, z: float, edges=BUCKET_EDGES, **boot) -> pd.DataFrame:
    """
    Per mid bucket (valid quotes only): mean mid, realized YES rate, and the
    mean net profit per contract of buying YES at the ask and of buying NO at
    1 - bid, each with a +/- z block-bootstrap interval.
    """
    d = obs[obs["valid"]].copy()
    d["bucket"] = bucket_of(d["mid"], edges)
    rows = []
    for b, g in d.groupby("bucket"):
        yes = block_bootstrap(g["outcome"], g["day"], **boot)
        ev_y = block_bootstrap(ev_buy_yes(g["outcome"], g["ask"]), g["day"], **boot)
        ev_n = block_bootstrap(ev_buy_no(g["outcome"], g["bid"]), g["day"], **boot)
        rows.append({
            "bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": len(g),
            "mid": float(g["mid"].mean()),
            "spread": float((g["ask"] - g["bid"]).mean()),
            "yes_rate": yes["mean"],
            "yes_lo": interval(yes, z)[0], "yes_hi": interval(yes, z)[1],
            "ev_yes": ev_y["mean"], "ev_yes_lo": interval(ev_y, z)[0], "ev_yes_hi": interval(ev_y, z)[1],
            "ev_no": ev_n["mean"], "ev_no_lo": interval(ev_n, z)[0], "ev_no_hi": interval(ev_n, z)[1],
            "n_eff": ev_y["n_eff"],
        })
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), LOGLOSS_EPS, 1 - LOGLOSS_EPS)
    return np.log(p / (1 - p))


def calibration_fit(prob, outcome, iterations: int = 50):
    """
    Logistic regression outcome ~ a + b * logit(prob), by Newton's method.
    A perfectly calibrated price gives a = 0, b = 1. b > 1 means prices are
    too close to 50% (favorites underpriced, longshots overpriced -- the
    classic favorite-longshot bias); b < 1 means they are too extreme.
    """
    x = _logit(prob)
    y = np.asarray(outcome, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def calibration_bootstrap(prob, outcome, blocks, resamples: int = 300, seed: int = BOOTSTRAP_SEED) -> dict:
    """calibration_fit with a day-block bootstrap standard error on each coefficient."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    labels, inv = np.unique(np.asarray(blocks), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(labels))]
    a, b = calibration_fit(prob, outcome)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])
        draws.append(calibration_fit(prob[pick], outcome[pick]))
    draws = np.array(draws)
    return {"a": a, "b": b, "a_se": float(draws[:, 0].std(ddof=1)), "b_se": float(draws[:, 1].std(ddof=1))}


def brier(prob, outcome):
    return (np.asarray(prob, dtype=float) - np.asarray(outcome, dtype=float)) ** 2


def log_loss(prob, outcome):
    p = np.clip(np.asarray(prob, dtype=float), LOGLOSS_EPS, 1 - LOGLOSS_EPS)
    y = np.asarray(outcome, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- fetching and caching ----------------------------------------------------

def _get(session, path: str, params: dict, retries: int = 6, pause: float = 0.12):
    """GET with backoff on 429 / 5xx. Kalshi's public tier rate-limits bursts."""
    url = f"{kalshi_api.BASE_URL}{path}"
    delay = 1.0
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == retries - 1:
                resp.raise_for_status()
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        resp.raise_for_status()
        time.sleep(pause)
        return resp.json()
    raise RuntimeError("unreachable")


def iter_markets(session, path: str, params: dict):
    """Every market from a cursor-paginated listing."""
    cursor = None
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        page = _get(session, path, p)
        markets = page.get("markets", [])
        yield from markets
        cursor = page.get("cursor")
        if not cursor or not markets:
            return


def fetch_settled_markets(session) -> list:
    """Raw records of every settled KXBTC15M market, live and historical, one per ticker."""
    seen = {}
    for m in iter_markets(session, "/markets", {"series_ticker": SERIES, "status": "settled"}):
        seen[m["ticker"]] = m
    for m in iter_markets(session, "/historical/markets", {"series_ticker": SERIES}):
        seen.setdefault(m["ticker"], m)
    return sorted(seen.values(), key=lambda m: m["close_time"])


def candle_batches(markets: list, cap: int = MAX_BATCH_CANDLES) -> list:
    """
    Groups parsed markets (sorted by close) into batch requests that stay under
    Kalshi's cap. The cap counts markets x minutes in the requested range, and
    the range has to span every market in the batch, so batches are runs of
    consecutive contracts: k back-to-back 15-minute contracts cost ~15k^2.
    Returns [(tickers, start_ts, end_ts)].
    """
    batches, cur = [], []

    def cost(group):
        lo = min(m["open_ts"] for m in group)
        hi = max(m["close_ts"] for m in group)
        return len(group) * ((hi - lo) // 60 + 1)

    for m in sorted(markets, key=lambda m: m["close_ts"]):
        if cur and cost(cur + [m]) > cap:
            batches.append(cur)
            cur = []
        cur.append(m)
    if cur:
        batches.append(cur)
    return [([m["ticker"] for m in g], min(m["open_ts"] for m in g), max(m["close_ts"] for m in g))
            for g in batches]


def _paths(cache_dir):
    return (os.path.join(cache_dir, "kalshi_markets.json"),
            os.path.join(cache_dir, "kalshi_candles.jsonl"))


def load_or_fetch_markets(session, cache_dir: str = CACHE_DIR, refresh: bool = False) -> list:
    """The cached raw market list, fetched once (or again with refresh=True)."""
    path, _ = _paths(cache_dir)
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raw = fetch_settled_markets(session)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    os.replace(tmp, path)
    return raw


def load_candles(cache_dir: str = CACHE_DIR) -> dict:
    """{ticker: raw candlesticks} from the cache; a torn last line (interrupted run) is ignored."""
    _, path = _paths(cache_dir)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["market_ticker"]] = rec["candlesticks"]
    return out


def fetch_missing_candles(session, markets: list, cache_dir: str = CACHE_DIR, progress=None) -> int:
    """
    Fetches 1-minute candlesticks for every parsed market not yet cached and
    appends each market's raw entry to the cache as one JSON line, so an
    interrupted run resumes where it stopped. Returns how many were fetched.
    """
    _, path = _paths(cache_dir)
    have = set(load_candles(cache_dir))
    todo = [m for m in markets if m["ticker"] not in have]
    batches = candle_batches(todo)
    os.makedirs(cache_dir, exist_ok=True)
    fetched = 0
    with open(path, "a", encoding="utf-8") as f:
        for i, (tickers, lo, hi) in enumerate(batches):
            page = _get(session, "/markets/candlesticks", {
                "market_tickers": ",".join(tickers), "start_ts": lo, "end_ts": hi, "period_interval": 1})
            got = {e["market_ticker"]: e for e in page.get("markets", [])}
            for t in tickers:
                # A market missing from the response is cached empty, so it is
                # counted as "no quote" rather than refetched forever.
                e = got.get(t, {"market_ticker": t, "candlesticks": []})
                f.write(json.dumps({"market_ticker": t, "candlesticks": e.get("candlesticks", [])}) + "\n")
                fetched += 1
            f.flush()
            if progress:
                progress(i + 1, len(batches))
    return fetched
