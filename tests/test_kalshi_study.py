"""
Tests for kalshi_study.py.

Payload fixtures are VERBATIM responses from Kalshi's public API, captured
2026-11-07 from the user's machine:
    GET https://external-api.kalshi.com/trade-api/v2/markets?series_ticker=KXBTC15M&status=settled&limit=2
        -> tests/fixtures/kalshi_settled_markets_page.json
    GET https://external-api.kalshi.com/trade-api/v2/historical/markets?series_ticker=KXBTC15M&limit=2
        -> tests/fixtures/kalshi_historical_markets_page.json
    GET https://external-api.kalshi.com/trade-api/v2/markets/candlesticks?market_tickers=KXBTC15M-26NOV071845-45%2CKXBTC15M-26NOV071830-30&start_ts=1794091500&end_ts=1794095100&period_interval=1
        -> tests/fixtures/kalshi_batch_candles_live.json
    GET https://external-api.kalshi.com/trade-api/v2/markets/candlesticks?market_tickers=KXBTC15M-26SEP071945-45&start_ts=1788823800&end_ts=1788824700&period_interval=1
        -> tests/fixtures/kalshi_batch_candles_historical.json   (a market under /historical/)
Things the payloads show that the code depends on: the batch response is
{"markets": [{"market_ticker", "candlesticks"}]}; a candle's closing quote
is under yes_bid/yes_ask "close_dollars"; a decided market's last quotes are
bid 0.999 / ask 1.0000 (no offer), and the minute after close is bid 0 /
ask 1 with a null last price.

The fee test checks against the table printed in Kalshi's own fee schedule
(kalshi.com/docs/kalshi-fee-schedule.pdf, effective July 7, 2026), which
rounds each order's fee up to the cent.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

import kalshi_study as ks

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


LIVE_PAGE = load("kalshi_settled_markets_page.json")
HIST_PAGE = load("kalshi_historical_markets_page.json")
LIVE_CANDLES = load("kalshi_batch_candles_live.json")
HIST_CANDLES = load("kalshi_batch_candles_historical.json")


# --- fees and costs -----------------------------------------------------------

# (price, fee for 1 contract, fee for 100 contracts), from the schedule's table.
FEE_TABLE = [
    (0.01, 0.01, 0.07), (0.05, 0.01, 0.34), (0.10, 0.01, 0.63), (0.15, 0.01, 0.90),
    (0.20, 0.02, 1.12), (0.25, 0.02, 1.32), (0.30, 0.02, 1.47), (0.35, 0.02, 1.60),
    (0.40, 0.02, 1.68), (0.45, 0.02, 1.74), (0.50, 0.02, 1.75), (0.55, 0.02, 1.74),
    (0.60, 0.02, 1.68), (0.65, 0.02, 1.60), (0.70, 0.02, 1.47), (0.75, 0.02, 1.32),
    (0.80, 0.02, 1.12), (0.85, 0.01, 0.90), (0.90, 0.01, 0.63), (0.95, 0.01, 0.34),
    (0.99, 0.01, 0.07),
]


@pytest.mark.parametrize("price,one,hundred", FEE_TABLE)
def test_taker_fee_reproduces_kalshis_published_table(price, one, hundred):
    def to_cent(x):
        return math.ceil(round(x * 100, 9)) / 100
    assert to_cent(ks.taker_fee(price, 1)) == pytest.approx(one)
    assert to_cent(ks.taker_fee(price, 100)) == pytest.approx(hundred)


def test_buying_yes_pays_the_ask_and_the_fee():
    # YES at 0.60 that settles YES: +0.40 less 0.07*0.6*0.4 = 0.0168
    assert ks.ev_buy_yes(1, 0.60) == pytest.approx(0.40 - 0.0168)
    assert ks.ev_buy_yes(0, 0.60) == pytest.approx(-0.60 - 0.0168)


def test_buying_no_pays_one_minus_the_bid_and_the_fee_on_that_price():
    # bid 0.58 -> NO costs 0.42, fee 0.07*0.42*0.58 = 0.017052
    assert ks.ev_buy_no(0, 0.58) == pytest.approx(0.58 - 0.017052)
    assert ks.ev_buy_no(1, 0.58) == pytest.approx(-0.42 - 0.017052)


def test_holding_both_sides_always_loses_the_spread_and_two_fees():
    rng = np.random.default_rng(0)
    bid = rng.uniform(0.01, 0.95, 500)
    ask = bid + rng.uniform(0.001, 0.04, 500)
    for outcome in (0, 1):
        both = ks.ev_buy_yes(outcome, ask) + ks.ev_buy_no(outcome, bid)
        assert np.allclose(both, (bid - ask) - ks.taker_fee(ask) - ks.taker_fee(1 - bid))
        assert (both < 0).all()


@pytest.mark.parametrize("bid,ask,ok", [
    (0.46, 0.47, True), (0.0, 0.999, False), (0.0, 1.0, False), (0.999, 1.0, False),
    (0.5, 0.5, False), (0.001, 0.999, True), (float("nan"), 0.5, False),
])
def test_valid_book_needs_a_real_bid_and_offer(bid, ask, ok):
    assert bool(ks.valid_book(bid, ask)) is ok


# --- parsing verbatim payloads -------------------------------------------------

def test_settled_markets_parse_from_both_the_live_and_historical_listings():
    live = [ks.parse_settled_market(m) for m in LIVE_PAGE["markets"]]
    hist = [ks.parse_settled_market(m) for m in HIST_PAGE["markets"]]
    assert live[0] == {
        "ticker": "KXBTC15M-26NOV071845-45", "open_ts": 1794094200, "close_ts": 1794095100,
        "strike": 75678.19, "expiration_value": 75748.63, "outcome": 1,
    }
    assert live[1]["expiration_value"] == pytest.approx(75678.19)     # = next contract's strike
    assert hist[1]["ticker"] == "KXBTC15M-26SEP071930-30" and hist[1]["outcome"] == 0
    assert hist[1]["strike"] == pytest.approx(79111.6)


def test_an_unsettled_or_void_market_is_skipped():
    m = dict(LIVE_PAGE["markets"][0])
    for result in ("", "void", None):
        m["result"] = result
        assert ks.parse_settled_market(m) is None


def test_candles_give_the_quote_standing_at_each_minute():
    by = {e["market_ticker"]: ks.parse_candles(e["candlesticks"]) for e in LIVE_CANDLES["markets"]}
    c = by["KXBTC15M-26NOV071845-45"]
    close = 1794095100
    assert ks.quote_at(c, close - 600) == (0.77, 0.78, 0.77)
    assert ks.quote_at(c, close - 300) == (0.85, 0.86, 0.86)
    assert ks.quote_at(c, close - 120) == (0.988, 0.989, 0.989)
    assert ks.quote_at(c, close - 61) is None                          # not on a minute
    after = by["KXBTC15M-26NOV071830-30"][1794094260]                  # minute after its close
    assert after == (0.0, 1.0, None)


def test_historical_markets_parse_the_same_way():
    e = HIST_CANDLES["markets"][0]
    c = ks.parse_candles(e["candlesticks"])
    assert ks.quote_at(c, 1788824700 - 600) == (0.41, 0.42, 0.41)
    assert ks.quote_at(c, 1788824700 - 120) == (0.975, 0.976, 0.976)


def test_candles_also_accept_bare_close_keys():
    # Hand-built: the per-market /historical/markets/{t}/candlesticks endpoint
    # answered with "close" rather than "close_dollars" when probed.
    c = ks.parse_candles([{"end_period_ts": 60, "yes_bid": {"close": "0.4600"},
                           "yes_ask": {"close": "0.4700"}, "price": {"close": None}}])
    assert c == {60: (0.46, 0.47, None)}


def _fixture_markets():
    markets = [ks.parse_settled_market(m) for m in LIVE_PAGE["markets"] + HIST_PAGE["markets"]]
    candles = {e["market_ticker"]: ks.parse_candles(e["candlesticks"])
               for e in LIVE_CANDLES["markets"] + HIST_CANDLES["markets"]}
    return markets, candles


def test_observations_take_one_row_per_contract_per_horizon_and_keep_missing_quotes_visible():
    markets, candles = _fixture_markets()
    obs = ks.observations(markets, candles, horizons=(10, 2))
    assert len(obs) == 4 * 2
    row = obs[(obs.ticker == "KXBTC15M-26NOV071845-45") & (obs.horizon == 10)].iloc[0]
    assert (row.bid, row.ask, row.outcome, row.valid) == (0.77, 0.78, 1, True)
    assert row.mid == pytest.approx(0.775)
    assert row.day == "2026-11-07"
    # KXBTC15M-26SEP071930-30 has no candles in the fixtures: kept, NaN, invalid.
    gone = obs[obs.ticker == "KXBTC15M-26SEP071930-30"]
    assert len(gone) == 2 and gone["bid"].isna().all() and not gone["valid"].any()


# --- split and statistics ---------------------------------------------------------

def test_split_holds_out_the_last_thirty_percent_by_close_time():
    close = np.arange(100, 1100, 100)[::-1]           # 10 contracts, out of order
    cut = ks.split_cutoff(close, 0.3)
    assert cut == 800
    assert (close >= cut).sum() == 3


def test_split_cutoff_counts_contracts_not_rows():
    close = np.repeat(np.arange(10), 3)                # 3 horizons each
    assert ks.split_cutoff(close, 0.3) == 7


def test_bonferroni_z():
    assert ks.bonferroni_z(1) == pytest.approx(1.959964, abs=1e-6)
    assert ks.bonferroni_z(60) == pytest.approx(3.3415, abs=1e-3)


def test_block_bootstrap_matches_iid_theory_when_blocks_are_single_observations():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 4000)
    s = ks.block_bootstrap(v, np.arange(4000), resamples=2000)
    assert s["se"] == pytest.approx(v.std(ddof=1) / math.sqrt(4000), rel=0.06)
    assert s["n_eff"] == pytest.approx(4000, rel=0.12)


def test_block_bootstrap_counts_a_duplicated_day_once():
    # 400 days, each with 5 identical copies of one draw: worth 400 observations, not 2000.
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(400), 5)
    v = np.repeat(rng.normal(0, 1, 400), 5)
    s = ks.block_bootstrap(v, days, resamples=2000)
    assert s["n_eff"] == pytest.approx(400, rel=0.12)


def test_bucket_of_uses_half_open_deciles_with_the_top_edge_inclusive():
    assert list(ks.bucket_of([0.0, 0.0999, 0.1, 0.55, 0.9, 1.0])) == [0, 0, 1, 5, 9, 9]


def test_bucket_table_reports_the_yes_rate_and_net_ev_per_bucket():
    obs = pd.DataFrame({
        "valid": [True, True, True, True, False],
        "mid": [0.155, 0.155, 0.855, 0.855, 0.5],
        "bid": [0.15, 0.15, 0.85, 0.85, 0.0],
        "ask": [0.16, 0.16, 0.86, 0.86, 1.0],
        "outcome": [1, 0, 1, 1, 1],
        "day": ["a", "b", "a", "b", "a"],
    })
    t = ks.bucket_table(obs, z=2.0, resamples=200).set_index("bucket")
    assert list(t.index) == ["0.1-0.2", "0.8-0.9"]                   # the invalid row is dropped
    assert t.loc["0.1-0.2", "yes_rate"] == pytest.approx(0.5)
    assert t.loc["0.1-0.2", "ev_yes"] == pytest.approx(0.5 - 0.16 - 0.07 * 0.16 * 0.84)
    assert t.loc["0.8-0.9", "ev_no"] == pytest.approx(-0.15 - 0.07 * 0.15 * 0.85)


def test_calibration_fit_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.03, 0.97, 200_000)
    x = np.log(p / (1 - p))
    true = 1 / (1 + np.exp(-(0.2 + 1.5 * x)))
    y = rng.random(len(p)) < true
    a, b = ks.calibration_fit(p, y)
    assert a == pytest.approx(0.2, abs=0.03) and b == pytest.approx(1.5, abs=0.03)


def test_a_calibrated_price_fits_zero_and_one():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.03, 0.97, 200_000)
    y = rng.random(len(p)) < p
    a, b = ks.calibration_fit(p, y)
    assert a == pytest.approx(0, abs=0.03) and b == pytest.approx(1, abs=0.03)


def test_calibration_fit_is_at_the_likelihood_maximum():
    # Independent check: the log-likelihood is lower one small step away in any direction.
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 3000)
    y = (rng.random(3000) < p ** 0.8).astype(float)
    a, b = ks.calibration_fit(p, y)
    x = np.log(p / (1 - p))

    def ll(a_, b_):
        q = 1 / (1 + np.exp(-(a_ + b_ * x)))
        return float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))
    best = ll(a, b)
    for da, db in [(1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)]:
        assert ll(a + da, b + db) < best


def test_brier_and_log_loss():
    assert ks.brier([0.8, 0.3], [1, 0]).tolist() == pytest.approx([0.04, 0.09])
    assert ks.log_loss([0.8], [1])[0] == pytest.approx(-math.log(0.8))
    assert math.isfinite(ks.log_loss([0.0], [1])[0])                  # clipped, not inf


# --- batching and fetching --------------------------------------------------------

def _contracts(n, start=1_000_020_000):
    return [{"ticker": f"T{i}", "open_ts": start + 900 * i, "close_ts": start + 900 * (i + 1)}
            for i in range(n)]


def test_candle_batches_stay_under_kalshis_cap_and_cover_every_market_once():
    ms = _contracts(200)
    batches = ks.candle_batches(ms)
    tickers = [t for b in batches for t in b[0]]
    assert sorted(tickers) == sorted(m["ticker"] for m in ms)
    for tick, lo, hi in batches:
        assert len(tick) * ((hi - lo) // 60 + 1) <= ks.MAX_BATCH_CANDLES
    # 25 back-to-back contracts cost 25 * 376 = 9400; 26 would cost 10166.
    assert len(batches[0][0]) == 25


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.handler(url, params or {})


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(ks.time, "sleep", lambda s: None)


def test_get_backs_off_on_rate_limits(no_sleep):
    answers = iter([FakeResponse(429, {}), FakeResponse(503, {}), FakeResponse(200, {"ok": 1})])
    s = FakeSession(lambda u, p: next(answers))
    assert ks._get(s, "/x", {}) == {"ok": 1}
    assert len(s.calls) == 3


def test_settled_markets_paginate_and_merge_live_with_historical(no_sleep):
    live = LIVE_PAGE["markets"]
    hist = HIST_PAGE["markets"]

    def handler(url, p):
        if url.endswith("/historical/markets"):
            return FakeResponse(200, {"markets": hist, "cursor": ""})
        if p.get("cursor") == "c2":
            return FakeResponse(200, {"markets": [live[1], hist[0]], "cursor": ""})
        return FakeResponse(200, {"markets": [live[0]], "cursor": "c2"})
    raw = ks.fetch_settled_markets(FakeSession(handler))
    assert [m["ticker"] for m in raw] == [
        "KXBTC15M-26SEP071930-30", "KXBTC15M-26SEP071945-45",
        "KXBTC15M-26NOV071830-30", "KXBTC15M-26NOV071845-45"]


def test_candle_cache_resumes_and_records_markets_kalshi_did_not_return(tmp_path, no_sleep):
    markets = [ks.parse_settled_market(m) for m in LIVE_PAGE["markets"] + HIST_PAGE["markets"]]
    entries = {e["market_ticker"]: e for e in LIVE_CANDLES["markets"] + HIST_CANDLES["markets"]}

    def handler(url, p):
        assert url.endswith("/markets/candlesticks") and p["period_interval"] == 1
        want = p["market_tickers"].split(",")
        return FakeResponse(200, {"markets": [entries[t] for t in want if t in entries]})
    s = FakeSession(handler)
    assert ks.fetch_missing_candles(s, markets, str(tmp_path)) == 4
    cached = ks.load_candles(str(tmp_path))
    assert set(cached) == {m["ticker"] for m in markets}
    assert cached["KXBTC15M-26SEP071930-30"] == []                    # not returned: cached empty
    # A torn last line from an interrupted run is ignored, and nothing is refetched.
    with open(tmp_path / "kalshi_candles.jsonl", "a", encoding="utf-8") as f:
        f.write('{"market_ticker": "T')
    n = len(s.calls)
    assert ks.fetch_missing_candles(s, markets, str(tmp_path)) == 0
    assert len(s.calls) == n
