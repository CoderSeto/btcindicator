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
  - Step 3 (added with its code, before any Step 3 output was seen): the
    model is kalshi.settlement_probability with spot = the last Coinbase
    trade before quote time and vol = realized vol of the prior
    VOL_LOOKBACK_MIN minutes. Two spot variants (SPOT_SOURCES) x three edge
    margins (EDGE_MARGINS) x three horizons = 18 trading variants, selected
    and tested by the same rule as above, on mean profit per trade. Brier
    and log-loss are compared with the market mid on the same rows, as a
    paired difference with a day-block interval.
  - Amendment, made at review before any outcome had been looked at (only
    raw quotes had been seen): the fee is rounded up to the cent PER ORDER,
    as in the table printed in Kalshi's schedule, so a 1-contract order at
    50c pays 2c and a 100-contract order 1.75c a contract. Both order sizes
    (ORDER_SIZES) are reported in full, as separate variants; neither is
    chosen by its results. Every interval in both steps uses one Bonferroni
    z over the TOTAL number of tests in Steps 2 and 3 (TOTAL_TESTS), not a
    per-step count. Fills at the quoted top of book are assumed; candlesticks
    carry no depth, so the size really available there may be smaller.

Fees, verified 2026-11-07 against Kalshi's own fee schedule
(kalshi.com/docs/kalshi-fee-schedule.pdf, "Last updated and effective: July
7, 2026"): "fees = round up(M x 0.07 x C x P x (1-P))". Its text says the
round-up is "such that the fee + positionCost is rounded to a centicent",
but the table on the next page rounds each order to the cent ($0.02 for one
contract at $0.50); this uses the cent, the costlier reading. Maker fees apply only to
series listed with a maker multiplier; "There is no settlement fee."
KXBTC15M is not on the non-standard list, and GET /series/KXBTC15M reports
fee_type "quadratic" (no maker fee) with fee_multiplier 1.

Data: the live API lists settled markets from 2026-09-01; older ones are
under /historical/markets (cutoff from GET /historical/cutoff). The batch
endpoint GET /markets/candlesticks caps one request at 10,000 candlesticks
counted as markets x minutes in the requested range, and returned nothing
for markets older than about 2026-08-31; those come one at a time from
/historical/markets/{ticker}/candlesticks (see fetch_missing_candles).
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
# Fixed price bands of 10 cents -- constants, not quantiles of any data -- so
# train and held-out are bucketed identically by construction.
BUCKET_EDGES = tuple(round(0.1 * i, 1) for i in range(11))
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261107
TAKER_FEE_RATE = 0.07
FEE_MULTIPLIER = 1.0
ORDER_SIZES = (1, 100)              # contracts per order; the fee rounds up per order

# Step 3
VOL_LOOKBACK_MIN = 120              # same window as the live calculator
MIN_VOL_CANDLES = 110               # of the 121 one-minute candles the vol window needs
SPOT_SOURCES = ("coinbase", "basis_adjusted")
EDGE_MARGINS = (0.0, 0.02, 0.05)   # extra edge, beyond costs, required to trade
LOGLOSS_EPS = 1e-4

# Every test in the study, counted once (see the amendment above).
STEP2_TESTS = len(HORIZONS_MIN) * (len(BUCKET_EDGES) - 1) * 2 * len(ORDER_SIZES)       # bucket x side x size
STEP3_TRADING_TESTS = len(SPOT_SOURCES) * len(EDGE_MARGINS) * len(HORIZONS_MIN) * len(ORDER_SIZES)
STEP3_SCORING_TESTS = len(HORIZONS_MIN) * 2 * 2                                         # 2 models x Brier/log-loss
TOTAL_TESTS = STEP2_TESTS + STEP3_TRADING_TESTS + STEP3_SCORING_TESTS

MAX_BATCH_CANDLES = 10_000          # Kalshi's per-request cap (its own error message)
CACHE_DIR = "study_data"


# --- costs -------------------------------------------------------------------

def taker_fee(price, contracts: float = 1.0, multiplier: float = FEE_MULTIPLIER, rate: float = TAKER_FEE_RATE):
    """
    Kalshi's taker fee formula in dollars for `contracts` contracts at
    `price`, BEFORE the per-order round-up (see order_fee). Elementwise.
    """
    p = np.asarray(price, dtype=float)
    return multiplier * rate * contracts * p * (1.0 - p)


def order_fee(price, contracts: int):
    """
    The fee actually charged on one order of `contracts` at `price`: the
    formula rounded UP to the cent, per order, as in Kalshi's printed table.
    One contract at 50c pays 2c; 100 pay $1.75. The 1e-6 rounding first stops
    float residue (0.07*100*0.25 = 1.7500000000000002) from adding a cent.
    """
    raw = np.asarray(taker_fee(price, contracts), dtype=float)
    return np.ceil(np.round(raw * 100, 6)) / 100


def fee_per_contract(price, contracts: int):
    """order_fee spread over the order's contracts."""
    return order_fee(price, contracts) / contracts


def ev_buy_yes(outcome, ask, contracts: int):
    """Profit per contract of buying YES at the ask in orders of `contracts`: pays 1 on YES."""
    ask = np.asarray(ask, dtype=float)
    return np.asarray(outcome, dtype=float) - ask - fee_per_contract(ask, contracts)


def ev_buy_no(outcome, bid, contracts: int):
    """Profit per contract of buying NO at 1 - bid (selling YES at the bid) in orders of `contracts`."""
    cost = 1.0 - np.asarray(bid, dtype=float)
    return (1.0 - np.asarray(outcome, dtype=float)) - cost - fee_per_contract(cost, contracts)


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
    """
    The (bid, ask, last) standing at `ts`: the closing quote of the candle
    whose period ENDS at `ts`, covering (ts - 60, ts]. None when no candle ends
    exactly then -- never the next candle, which ends after `ts` and would be
    look-ahead, and never an earlier one, which would be a stale quote.
    """
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


def bucket_table(obs: pd.DataFrame, z: float, contracts: int, edges=BUCKET_EDGES, **boot) -> pd.DataFrame:
    """
    Per mid bucket (valid quotes only): mean mid, realized YES rate, and the
    mean net profit per contract of buying YES at the ask and of buying NO at
    1 - bid in orders of `contracts`, each with a +/- z block-bootstrap interval.
    `*_mde` is z times the SE: the smallest edge that bucket could have
    detected, so a null result says how large an edge it rules out.
    `edges` are fixed, never derived from `obs`.
    """
    d = obs[obs["valid"]].copy()
    d["bucket"] = bucket_of(d["mid"], edges)
    rows = []
    for b, g in d.groupby("bucket"):
        yes = block_bootstrap(g["outcome"], g["day"], **boot)
        ev_y = block_bootstrap(ev_buy_yes(g["outcome"], g["ask"], contracts), g["day"], **boot)
        ev_n = block_bootstrap(ev_buy_no(g["outcome"], g["bid"], contracts), g["day"], **boot)
        rows.append({
            "bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": len(g),
            "mid": float(g["mid"].mean()),
            "spread": float((g["ask"] - g["bid"]).mean()),
            "yes_rate": yes["mean"],
            "yes_lo": interval(yes, z)[0], "yes_hi": interval(yes, z)[1],
            "ev_yes": ev_y["mean"], "ev_yes_lo": interval(ev_y, z)[0], "ev_yes_hi": interval(ev_y, z)[1],
            "ev_no": ev_n["mean"], "ev_no_lo": interval(ev_n, z)[0], "ev_no_hi": interval(ev_n, z)[1],
            "ev_yes_mde": z * ev_y["se"], "ev_no_mde": z * ev_n["se"], "yes_rate_mde": z * yes["se"],
            "n_eff": ev_y["n_eff"],
        })
    return pd.DataFrame(rows)


def exclusions(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Per month and horizon: contracts, how many were excluded for having no
    candle at that exact minute or for failing the real-book filter, and the
    exclusion rate among contracts that settled YES vs NO. If exclusion is
    related to the outcome, the buckets describe a biased subset.
    """
    d = obs.copy()
    d["month"] = d["day"].str[:7]
    d["no_quote"] = d["bid"].isna() | d["ask"].isna()
    d["bad_book"] = ~d["no_quote"] & ~d["valid"]
    d["excluded"] = ~d["valid"]
    rows = []
    for (month, h), g in d.groupby(["month", "horizon"]):
        yes, no = g[g.outcome == 1], g[g.outcome == 0]
        rows.append({"month": month, "horizon": h, "contracts": len(g),
                     "no_quote": int(g.no_quote.sum()), "bad_book": int(g.bad_book.sum()),
                     "excluded_rate": float(g.excluded.mean()),
                     "excl_rate_yes": float(yes.excluded.mean()) if len(yes) else float("nan"),
                     "excl_rate_no": float(no.excluded.mean()) if len(no) else float("nan")})
    return pd.DataFrame(rows)


def exclusion_outcome_gap(obs: pd.DataFrame, **boot) -> dict:
    """
    Exclusion rate among YES-settled minus among NO-settled contracts, with a
    day-block bootstrap SE: each day's YES and NO groups are resampled together.
    """
    d = obs.copy()
    exc = (~d["valid"]).astype(float).to_numpy()
    y = d["outcome"].to_numpy() == 1
    labels, inv = np.unique(d["day"].to_numpy(), return_inverse=True)
    k = len(labels)
    ey = np.bincount(inv, weights=exc * y, minlength=k)
    ny = np.bincount(inv, weights=y.astype(float), minlength=k)
    en = np.bincount(inv, weights=exc * ~y, minlength=k)
    nn = np.bincount(inv, weights=(~y).astype(float), minlength=k)
    gap = ey.sum() / ny.sum() - en.sum() / nn.sum()
    rng = np.random.default_rng(boot.get("seed", BOOTSTRAP_SEED))
    idx = rng.integers(0, k, size=(boot.get("resamples", BOOTSTRAP_RESAMPLES), k))
    draws = ey[idx].sum(1) / ny[idx].sum(1) - en[idx].sum(1) / nn[idx].sum(1)
    return {"rate_yes": float(ey.sum() / ny.sum()), "rate_no": float(en.sum() / nn.sum()),
            "gap": float(gap), "se": float(draws.std(ddof=1))}


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

def _get(session, path: str, params: dict, retries: int = 8, pause: float = 0.12):
    """GET with backoff on 429 / 5xx; None on 404. Kalshi's public tier rate-limits bursts."""
    url = f"{kalshi_api.BASE_URL}{path}"
    delay = 1.0
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return None
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


def historical_cutoff(session):
    """Epoch seconds before which settled markets live under /historical/ (None if unknown)."""
    page = _get(session, "/historical/cutoff", {})
    ts = (page or {}).get("market_settled_ts")
    return _ts(ts) if ts else None


def fetch_market_candles(session, m: dict) -> list:
    """
    One market's 1-minute candlesticks from the per-market endpoints: the
    historical one first, then the live series one. [] only if both have none.
    """
    params = {"start_ts": m["open_ts"], "end_ts": m["close_ts"], "period_interval": 1}
    for path in (f"/historical/markets/{m['ticker']}/candlesticks",
                 f"/series/{SERIES}/markets/{m['ticker']}/candlesticks"):
        page = _get(session, path, params)
        if page and page.get("candlesticks"):
            return page["candlesticks"]
    return []


def fetch_missing_candles(session, markets: list, cache_dir: str = CACHE_DIR, progress=None,
                          historical_before: int = None, workers: int = 1, session_factory=None) -> int:
    """
    Fetches 1-minute candlesticks for every parsed market not yet cached and
    appends each market's raw entry to the cache as one JSON line, so an
    interrupted run resumes where it stopped. Returns how many were fetched.

    Routing (both measured 2026-11-07): the batch endpoint returned nothing
    for markets older than about 2026-08-31, though the per-market
    /historical/ endpoint serves them. So markets closing before
    `historical_before` (GET /historical/cutoff's market_settled_ts) go
    straight to the per-market endpoints, the rest are batched, and any
    market a batch omits is retried one at a time. A market is cached empty
    only when every endpoint had nothing, never because a batch skipped it.
    `workers` > 1 runs the per-market requests on that many threads, each
    with its own session from `session_factory`.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, path = _paths(cache_dir)
    have = set(load_candles(cache_dir))
    todo = [m for m in markets if m["ticker"] not in have]
    old = [m for m in todo if historical_before is not None and m["close_ts"] < historical_before]
    new = [m for m in todo if not (historical_before is not None and m["close_ts"] < historical_before)]
    os.makedirs(cache_dir, exist_ok=True)
    lock = threading.Lock()
    done = [0]

    with open(path, "a", encoding="utf-8") as f:
        def write(t, cs):
            with lock:
                f.write(json.dumps({"market_ticker": t, "candlesticks": cs}) + "\n")
                f.flush()
                done[0] += 1
                if progress:
                    progress(done[0], len(todo))

        singles = list(old)
        for tickers, lo, hi in candle_batches(new):
            page = _get(session, "/markets/candlesticks", {
                "market_tickers": ",".join(tickers), "start_ts": lo, "end_ts": hi, "period_interval": 1})
            got = {e["market_ticker"]: e.get("candlesticks") for e in (page or {}).get("markets", [])}
            for m in (m for m in new if m["ticker"] in set(tickers)):
                if got.get(m["ticker"]):
                    write(m["ticker"], got[m["ticker"]])
                else:
                    singles.append(m)

        if workers <= 1 or session_factory is None:
            for m in singles:
                write(m["ticker"], fetch_market_candles(session, m))
        else:
            local = threading.local()

            def one(m):
                if not hasattr(local, "s"):
                    local.s = session_factory()
                write(m["ticker"], fetch_market_candles(local.s, m))
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(one, singles))
    return done[0]


# --- Step 3: Coinbase history, the model, the basis -----------------------------

COINBASE_PRODUCT = "BTC-USD"
COINBASE_CHUNK_MIN = 299            # data_fetch.MAX_CANDLE_SPAN: Coinbase caps a request at 300 buckets


def _coinbase_day_path(cache_dir, day):
    return os.path.join(cache_dir, "coinbase_1m", f"{day}.csv")


def fetch_coinbase_day(day: str, fetch=None, pause: float = 0.15, now=None) -> pd.DataFrame:
    """
    Every 1-minute BTC-USD candle starting on UTC `day` (YYYY-MM-DD), in
    COINBASE_CHUNK_MIN-minute requests, deduplicated, with integer `ts` (the
    candle's start, epoch seconds). Coinbase omits minutes with no trades.
    On the current day it stops at `now`: Coinbase answers HTTP 400 to a
    range that ends in the future.
    """
    import data_fetch
    fetch = fetch or data_fetch.get_candles_range
    start = pd.Timestamp(day, tz="UTC")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    end = min(start + pd.Timedelta(days=1), now)
    parts = []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(minutes=COINBASE_CHUNK_MIN), end)
        parts.append(fetch(COINBASE_PRODUCT, 60, t, t2))
        t = t2
        time.sleep(pause)
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.concat(parts, ignore_index=True)
    # Explicit ns resolution before the epoch conversion (gotcha #2).
    ts = pd.to_datetime(df["time"]).values.astype("datetime64[ns]").astype("int64") // 1_000_000_000
    df = df.assign(ts=ts)
    lo, hi = int(start.timestamp()), int(end.timestamp())
    df = df[(df.ts >= lo) & (df.ts < hi)].drop_duplicates("ts").sort_values("ts")
    return df[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def load_or_fetch_coinbase(first_ts: int, last_ts: int, cache_dir: str = CACHE_DIR, fetch=None,
                           progress=None, today: str = None) -> pd.DataFrame:
    """
    1-minute BTC-USD candles covering [first_ts, last_ts], cached one CSV per
    UTC day. Only complete past days are cached; today is always refetched.
    """
    days = pd.date_range(pd.to_datetime(first_ts, unit="s").normalize(),
                         pd.to_datetime(last_ts, unit="s").normalize(), freq="D")
    today = today or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    os.makedirs(os.path.join(cache_dir, "coinbase_1m"), exist_ok=True)
    frames = []
    for i, d in enumerate(days):
        day = d.strftime("%Y-%m-%d")
        path = _coinbase_day_path(cache_dir, day)
        if os.path.exists(path) and day < today:
            frames.append(pd.read_csv(path))
        else:
            df = fetch_coinbase_day(day, fetch)
            if day < today:
                df.to_csv(path + ".tmp", index=False)
                os.replace(path + ".tmp", path)
            frames.append(df)
        if progress:
            progress(i + 1, len(days))
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


class MinuteGrid:
    """
    1-minute candles on a gap-aware grid: row i is the candle starting at
    t0 + 60 i, NaN where Coinbase had none, so "the candle before time t" is
    an index lookup and a missing minute stays visibly missing.
    """

    def __init__(self, df: pd.DataFrame):
        ts = df["ts"].to_numpy(dtype=np.int64)
        if (ts % 60).any():
            raise ValueError("candle starts must be whole minutes")
        self.t0 = int(ts.min())
        n = int((ts.max() - self.t0) // 60) + 1
        idx = (ts - self.t0) // 60
        self.cols = {}
        for c in ("open", "high", "low", "close"):
            a = np.full(n, np.nan)
            a[idx] = df[c].to_numpy(dtype=float)
            self.cols[c] = a

    def _i(self, start_ts: int):
        i = (int(start_ts) - self.t0) // 60
        return i if 0 <= i < len(self.cols["close"]) else None

    def last_close_before(self, t: int):
        """Close of the candle covering [t - 60, t): the last trade before t."""
        i = self._i(t - 60)
        if i is None:
            return None
        v = self.cols["close"][i]
        return None if np.isnan(v) else float(v)

    def minute_average_before(self, t: int):
        """
        Proxy for a 60-second average ending at t: the mean of the open, high,
        low and close of the candle covering [t - 60, t). A real 60-print
        average needs per-second data this doesn't have.
        """
        i = self._i(t - 60)
        if i is None:
            return None
        v = float(np.mean([self.cols[c][i] for c in ("open", "high", "low", "close")]))
        return None if np.isnan(v) else v

    def closes_before(self, t: int, count: int):
        """The `count` closes of the candles ending at or before t (NaN where missing), oldest first."""
        i = self._i(t - 60)
        if i is None or i - count + 1 < 0:
            return None
        return self.cols["close"][i - count + 1: i + 1]


def model_inputs(grid: MinuteGrid, t: int, lookback_min: int = VOL_LOOKBACK_MIN,
                 min_candles: int = MIN_VOL_CANDLES):
    """
    (spot, annual_vol) as the live calculator would have had them at t: spot
    is the last Coinbase trade before t, vol the realized vol of the
    `lookback_min` one-minute returns ending at t (kalshi.realized_volatility,
    the function the dashboard uses). Missing minutes inside the vol window
    are carried forward if at least `min_candles` of its candles exist;
    otherwise None, as when the spot minute itself is missing. Nothing from
    after t is used.
    """
    spot = grid.last_close_before(t)
    closes = grid.closes_before(t, lookback_min + 1)
    if spot is None or closes is None or np.count_nonzero(~np.isnan(closes)) < min_candles:
        return None
    filled = pd.Series(closes).ffill().bfill()
    vol = kalshi.realized_volatility(pd.DataFrame({"close": filled}), lookback_min, 60)["annual_vol"]
    if vol is None:
        return None
    return spot, vol


def model_observations(obs: pd.DataFrame, grid: MinuteGrid) -> pd.DataFrame:
    """
    Adds, per observation, the Coinbase spot and vol at quote time and two
    model probabilities from kalshi.settlement_probability:
      p_model     -- spot straight from Coinbase (what the dashboard does)
      p_model_adj -- spot rescaled by strike / (Coinbase proxy for the strike's
                     own 60s average at open), removing the Coinbase-vs-index
                     basis as it stood at open. The strike is published at
                     open, so this uses nothing from after quote time.
    Also the basis at open and at close in basis points (Kalshi's index value
    over the Coinbase proxy, minus 1). Rows the model can't price get NaN.
    """
    out = obs.copy()
    spot, vol, p, p_adj, gap_open, gap_close = ([] for _ in range(6))
    for r in obs.itertuples():
        proxy_open = grid.minute_average_before(r.open_ts)
        proxy_close = grid.minute_average_before(r.close_ts)
        gap_open.append((r.strike / proxy_open - 1) * 1e4 if proxy_open else np.nan)
        ev = r.expiration_value
        gap_close.append((ev / proxy_close - 1) * 1e4 if proxy_close and ev == ev and ev else np.nan)
        mi = model_inputs(grid, r.quote_ts)
        if mi is None:
            for col in (spot, vol, p, p_adj):
                col.append(np.nan)
            continue
        s, v = mi
        spot.append(s)
        vol.append(v)
        p.append(kalshi.settlement_probability(s, r.strike, r.horizon, v))
        p_adj.append(kalshi.settlement_probability(s * r.strike / proxy_open, r.strike, r.horizon, v)
                     if proxy_open else np.nan)
    out["spot"], out["vol"], out["p_model"], out["p_model_adj"] = spot, vol, p, p_adj
    out["basis_open_bp"], out["basis_close_bp"] = gap_open, gap_close
    return out


def model_trades(prob, bid, ask, outcome, margin: float, contracts: int):
    """
    Trade toward the model when its edge over the price paid, net of the fee,
    exceeds `margin`: buy YES at the ask if p - ask - fee(ask) > margin, buy NO
    at 1 - bid if (1 - p) - (1 - bid) - fee(1 - bid) > margin, else nothing.
    The fee is the per-contract share of an order of `contracts`.
    Returns (side, pnl): side +1 YES / -1 NO / 0 none, pnl NaN when no trade.
    The two can't both fire, since ask > bid.
    """
    p = np.asarray(prob, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    with np.errstate(invalid="ignore"):
        edge_yes = p - ask - fee_per_contract(ask, contracts)
        edge_no = (1 - p) - (1 - bid) - fee_per_contract(1 - bid, contracts)
    with np.errstate(invalid="ignore"):
        side = np.where(edge_yes > margin, 1, np.where(edge_no > margin, -1, 0))
    pnl = np.where(side == 1, ev_buy_yes(outcome, ask, contracts),
                   np.where(side == -1, ev_buy_no(outcome, bid, contracts), np.nan))
    return side, pnl
