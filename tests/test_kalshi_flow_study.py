"""
Tests for kalshi_flow_study.py: the Binance.US trade cache (pagination, resume,
gaps), the flow features (against a naive brute force on hand-built trades
with exact answers, and for look-ahead), the taker-side and timestamp rules,
the models (a flow model is its counterpart plus flow columns and nothing
else), the verdict rule, and the whole pipeline on a synthetic world whose
answer is known. Nothing touches the network: the API is a fake in this file.

The one verbatim payload is the aggTrades example from Binance's own REST
documentation (binance-spot-api-docs, rest-api.md, "Compressed/Aggregate trades
list"; Binance.US serves the same schema). Re-verify it against the live docs
when you can.
"""
import math

import numpy as np
import pandas as pd
import pytest

import exchange_feeds
import kalshi_flow_study as fl
import kalshi_forecast_study as fc
import kalshi_study as ks
from test_kalshi_forecast_study import synthetic_world

DOCS_AGG_TRADE = {"a": 26129, "p": "0.01633102", "q": "4.70443515", "f": 27781, "l": 27781,
                  "T": 1498793709153, "m": True, "M": True}

DAY = "2026-03-01"
D0 = fl.day_start_ms(DAY)                                  # 1772323200000


# ---------- the family ----------

def test_the_family_is_eight_models_and_sixty_intervals():
    assert len(fl.MODELS) == 8 and len(set(fl.MODELS)) == 8
    assert fl.FLOW_MODELS == ("market_flow", "market_flow_regime", "model_flow", "both_flow")
    assert set(fl.BASE_MODELS) == set(fl.COUNTERPART.values()) and len(fl.BASE_MODELS) == 4
    assert fl.FAMILY_TESTS == 60                                  # (8 x 3 + 4 counterparts + 2 components) x 2 metrics
    assert len(set(fl.COMPARISONS)) == 30
    assert fl.BASELINES is fc.BASELINES and len(fl.FLOW_FEATURES) == 7
    assert ks.bonferroni_z(fl.FAMILY_TESTS) == pytest.approx(3.34, abs=0.01)


def test_a_flow_model_is_its_counterpart_plus_flow_columns_and_nothing_else():
    for flow, base in fl.COUNTERPART.items():
        f, b = fl.model_columns(flow), fl.model_columns(base)
        assert f[:len(b)] == b
        extra = f[len(b):]
        assert extra[:7] == list(fl.FLOW_FEATURES)
        assert all(c in fl.FLOW_FEATURES or "_x_" in c for c in extra)      # every added column is flow
    assert [c for c in fl.model_columns("market_flow_regime") if "_x_" in c] == \
        ["imb_s_x_high_vol", "imb_l_x_high_vol", "imb_s_x_us_hours", "imb_l_x_us_hours"]
    for base in fl.BASE_MODELS:
        assert not set(fl.model_columns(base)) & set(fl.FLOW_FEATURES)        # no flow in any flow-free model
    with pytest.raises(ValueError):
        fl.model_columns("market_vibes")


# ---------- time: gotcha 2 ----------

def test_epoch_ms_is_right_whatever_the_datetime_resolution():
    want = 1_772_323_200_000                                       # 2026-03-01 00:00:00 UTC
    for unit in ("s", "ms", "us", "ns"):
        t = np.array(["2026-03-01T00:00:00"], dtype=f"datetime64[{unit}]")
        assert fl.epoch_ms(t)[0] == want, unit
    assert fl.epoch_ms(pd.to_datetime([want // 1000], unit="s"))[0] == want          # the call that is not reliably ns
    assert fl.day_start_ms("2026-03-01") == want and fl.day_of_ms(want + 86_399_999) == "2026-03-01"
    assert fl.day_of_ms(want + 86_400_000) == "2026-03-02"


def test_implausible_timestamps_fail_loudly():
    fl.check_plausible_ms([D0, D0 + 5])                                              # fine
    with pytest.raises(ValueError, match="implausible"):
        fl.check_plausible_ms([D0 // 1000])                                          # seconds passed as milliseconds
    with pytest.raises(ValueError, match="implausible"):
        fl.check_plausible_ms([86_400_000])                                          # a 1970 date: the gotcha's signature
    with pytest.raises(ValueError, match="implausible"):
        fl.check_plausible_ms([D0 * 1000])                                           # microseconds
    with pytest.raises(ValueError, match="outside the requested range"):
        fl.check_plausible_ms([D0 + 10], D0 + 20, D0 + 30)
    fl.check_plausible_ms([])                                                        # nothing to check


def test_quote_times_in_milliseconds_are_refused_by_the_feature_builder():
    trades = hand_trades([(D0 + 1000, 100.0, 1.0, True)])
    with pytest.raises(ValueError, match="implausible quote times"):
        fl.flow_features(trades, [D0 + 60_000])                                      # ms passed where seconds are expected


# ---------- taker side: gotcha 17 ----------

def test_the_documented_payload_parses_and_is_buyer_maker_means_the_taker_sold():
    p = fl.parse_page([DOCS_AGG_TRADE])
    assert p["id"][0] == 26129 and p["ts"][0] == 1498793709153
    assert p["price"][0] == pytest.approx(0.01633102) and p["qty"][0] == pytest.approx(4.70443515)
    assert not p["buy"][0]                                         # m = true: the buyer was resting, so the taker SOLD
    assert fl.parse_page([{**DOCS_AGG_TRADE, "m": False}])["buy"][0]
    # the side comes from the one parser every venue's convention is pinned in
    assert exchange_feeds.parse_binance_trade(DOCS_AGG_TRADE)["side"] == "sell"


def test_an_unparseable_trade_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unparseable"):
        fl.parse_page([{"a": 1, "T": D0, "p": "1.0"}])


def test_a_taker_buy_pushes_every_signed_feature_up_and_a_taker_sell_down():
    t = D0 // 1000 + 3600
    buys = [{"a": i, "p": "100", "q": "200", "T": t * 1000 - 1000 * i, "m": False} for i in range(1, 13)]
    sells = [{**b, "m": True} for b in buys]
    for page, sign in ((buys, 1.0), (sells, -1.0)):
        tr = fl.parse_page(sorted(page, key=lambda r: r["a"]))
        tr["holes"] = []
        f = fl.flow_features(tr, [t]).iloc[0]
        assert f.imb_s == sign and f.imb_l == sign and f.cnt_imb_l == sign
        assert f.large_imb_l == sign                               # 100 x 200 = $20,000 each: all large
        assert math.copysign(1, f.net_surge) == sign


# ---------- fetching ----------

def make_raw(n, start_ms=D0, step_ms=7_000, first_id=5000, seed=0):
    rng = np.random.default_rng(seed)
    return [{"a": first_id + i, "p": f"{70_000 + float(rng.normal(0, 30)):.2f}", "q": f"{float(rng.uniform(0.001, 0.2)):.5f}",
             "f": i, "l": i, "T": int(start_ms + step_ms * i), "m": bool(rng.random() < 0.5), "M": True} for i in range(n)]


class FakeApi:
    """aggTrades as Binance documents it: by time (span under an hour, from startTime) or by fromId, 1000 a page."""

    def __init__(self, raw):
        self.raw = sorted(raw, key=lambda r: r["a"])
        self.calls = []

    def __call__(self, params):
        self.calls.append(dict(params))
        assert params["symbol"] == fl.SYMBOL and params["limit"] <= 1000
        if "fromId" in params:
            assert "startTime" not in params and "endTime" not in params
            rows = [r for r in self.raw if r["a"] >= params["fromId"]]
        else:
            assert params["endTime"] - params["startTime"] < fl.HOUR_MS
            rows = [r for r in self.raw if params["startTime"] <= r["T"] <= params["endTime"]]
        return rows[:params["limit"]]


def test_fetch_returns_exactly_the_trades_in_the_range_across_many_pages():
    raw = make_raw(4500)                                           # 7s apart: 31,500 s, most of a day, 5 pages
    api = FakeApi(raw)
    lo, hi = D0 + 3_600_000, D0 + 6 * 3_600_000
    got = fl.fetch_trades(api, lo, hi)
    want = [r for r in raw if lo <= r["T"] <= hi]
    assert list(got["id"]) == [r["a"] for r in want] and len(want) > 2000
    assert list(got["ts"]) == [r["T"] for r in want]
    assert list(got["buy"]) == [not r["m"] for r in want]
    assert got["price"] == pytest.approx([float(r["p"]) for r in want])
    assert sum("fromId" in c for c in api.calls) >= 2             # paged by id after the first page


def test_both_ends_of_the_range_are_inclusive():
    raw = make_raw(10, step_ms=1000)
    got = fl.fetch_trades(FakeApi(raw), raw[2]["T"], raw[6]["T"])
    assert list(got["id"]) == [r["a"] for r in raw[2:7]]


def test_trades_sharing_a_millisecond_at_a_page_boundary_are_all_kept():
    # 1,500 trades in ONE millisecond straddling the 1000-trade page: a time cursor (last T + 1) would drop 500
    raw = make_raw(1500, step_ms=0) + make_raw(10, start_ms=D0 + 5000, first_id=6500, seed=1)
    got = fl.fetch_trades(FakeApi(raw), D0, D0 + 120_000)
    assert len(got["id"]) == 1510 and (np.diff(got["id"]) == 1).all()


def test_the_first_trade_is_found_hours_into_a_quiet_day_and_an_empty_day_is_empty():
    raw = make_raw(30, start_ms=D0 + 5 * fl.HOUR_MS + 123)
    api = FakeApi(raw)
    got = fl.fetch_trades(api, D0, D0 + fl.DAY_MS - 1)
    assert len(got["id"]) == 30
    assert sum("startTime" in c for c in api.calls) == 6         # hours 0..5, one request each
    empty = fl.fetch_trades(FakeApi([]), D0, D0 + fl.DAY_MS - 1)
    assert len(empty["id"]) == 0 and empty["buy"].dtype == bool


def test_fetch_refuses_timestamps_that_are_not_milliseconds():
    bad = [{**r, "T": r["T"] // 1000} for r in make_raw(5)]

    def api(params):
        return bad if "startTime" in params else []
    with pytest.raises(ValueError, match="implausible"):
        fl.fetch_trades(api, D0, D0 + fl.DAY_MS - 1)


def test_fetch_is_resumable_one_file_per_day(tmp_path):
    raw = make_raw(3000, step_ms=60_000)                           # one a minute: just over two days
    api = FakeApi(raw)
    end = D0 + 2 * fl.DAY_MS + 3_600_000
    seen = []
    assert fl.fetch_missing_days(api, str(tmp_path), D0, end, progress=lambda i, n, day: seen.append(day)) == 3
    assert seen == ["2026-03-01", "2026-03-02", "2026-03-03"]
    calls = len(api.calls)
    assert fl.fetch_missing_days(api, str(tmp_path), D0, end) == 0 and len(api.calls) == calls      # nothing refetched
    t = fl.load_trades(str(tmp_path), D0, end)
    want = [r for r in raw if r["T"] <= end]
    assert list(t["id"]) == [r["a"] for r in want] and t["holes"] == [] and t["missing_ids"] == 0
    # a day cached only up to an earlier end is fetched again when more of it is needed; complete days are not
    later = end + 3_600_000
    assert fl.fetch_missing_days(api, str(tmp_path), D0, later) == 1
    assert fl.load_day(str(tmp_path), "2026-03-03")[1] == later
    # a torn file is treated as absent and refetched
    with open(fl._day_path(str(tmp_path), "2026-03-02"), "wb") as f:
        f.write(b"not an npz")
    assert fl.load_day(str(tmp_path), "2026-03-02") is None
    assert fl.fetch_missing_days(api, str(tmp_path), D0, later) == 1
    assert fl.load_trades(str(tmp_path), D0, later)["holes"] == []


def test_a_missing_day_and_a_break_in_the_ids_are_both_counted_as_holes(tmp_path):
    raw = make_raw(4000, step_ms=60_000)
    del raw[1000:1040]                                             # the API (or the cache) lost 40 consecutive trades
    api = FakeApi(raw)
    end = D0 + 2 * fl.DAY_MS + 3_600_000
    fl.fetch_missing_days(api, str(tmp_path), D0, end)
    import os
    os.remove(fl._day_path(str(tmp_path), "2026-03-03"))           # and a whole day never arrived
    t = fl.load_trades(str(tmp_path), D0, end)
    assert t["missing_ids"] == 40 and len(t["id_gaps"]) == 1
    before, after = D0 + 999 * 60_000, D0 + 1040 * 60_000          # the trades either side of the break
    assert t["id_gaps"][0] == (before, after)
    assert (D0 + 2 * fl.DAY_MS, end) in t["holes"]                 # the uncovered day
    assert len(t["holes"]) == 2
    # a cache with nothing at all is one hole over the whole span
    empty = fl.load_trades(str(tmp_path / "none"), D0, end)
    assert empty["holes"] == [(D0, end)] and len(empty["ts"]) == 0


def test_days_fetched_in_parallel_give_the_same_cache(tmp_path):
    raw = make_raw(3000, step_ms=60_000)
    end = D0 + 2 * fl.DAY_MS - 1
    fl.fetch_missing_days(FakeApi(raw), str(tmp_path / "a"), D0, end, workers=1)
    fl.fetch_missing_days(FakeApi(raw), str(tmp_path / "b"), D0, end, workers=3)
    a, b = fl.load_trades(str(tmp_path / "a"), D0, end), fl.load_trades(str(tmp_path / "b"), D0, end)
    assert np.array_equal(a["id"], b["id"]) and np.array_equal(a["ts"], b["ts"]) and a["holes"] == b["holes"] == []


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_binance_get_waits_out_a_rate_limit_and_slows_near_the_weight_limit(monkeypatch):
    naps = []
    monkeypatch.setattr(fl.time, "sleep", naps.append)

    class Session:
        def __init__(self, responses):
            self.responses, self.urls = list(responses), []

        def get(self, url, params=None, timeout=None):
            self.urls.append(url)
            return self.responses.pop(0)

    s = Session([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(503),
                 FakeResponse(200, [DOCS_AGG_TRADE], {"x-mbx-used-weight-1m": "900"})])
    state = {}
    assert fl.binance_get(s, {"symbol": fl.SYMBOL}, state=state) == [DOCS_AGG_TRADE]
    assert naps == [7.0, 4.0, 5.0]                                 # Retry-After honoured, a doubled backoff, the weight brake
    assert state["used_weight"] == 900
    assert s.urls[0] == "https://api.binance.us/api/v3/aggTrades"  # the public, no-auth endpoint
    with pytest.raises(RuntimeError):
        fl.binance_get(Session([FakeResponse(429)] * 2), {}, retries=2)


# ---------- flow features: brute force on hand-built trades ----------

def hand_trades(rows, holes=()):
    """rows: (ts_ms, price, qty, taker_buy). Ids follow the order given."""
    rows = sorted(rows)
    return {"id": np.arange(1, len(rows) + 1, dtype=np.int64), "ts": np.array([r[0] for r in rows], np.int64),
            "price": np.array([r[1] for r in rows], float), "qty": np.array([r[2] for r in rows], float),
            "buy": np.array([r[3] for r in rows], bool), "holes": list(holes)}


def brute_features(trades, quote_s):
    """The definitions, written as loops over every trade: no sorting assumption, no prefix sums, no early exit."""
    t = quote_s * 1000
    acc = {k: 0.0 for k in ("bs", "ss", "bl", "sl", "nbl", "nsl", "lb", "ls", "ns")}
    for ts, price, qty, buy in zip(trades["ts"], trades["price"], trades["qty"], trades["buy"]):
        notional = price * qty
        if t - fl.LONG_MIN * 60_000 <= ts <= t:
            acc["bl" if buy else "sl"] += notional
            acc["nbl" if buy else "nsl"] += 1
            if notional >= fl.LARGE_USD:
                acc["lb" if buy else "ls"] += notional
        if t - fl.SHORT_MIN * 60_000 <= ts <= t:
            acc["bs" if buy else "ss"] += notional
            acc["ns"] += 1
    n_l, tot_l, tot_s = acc["nbl"] + acc["nsl"], acc["bl"] + acc["sl"], acc["bs"] + acc["ss"]
    return {"imb_s": (acc["bs"] - acc["ss"]) / tot_s if acc["ns"] else 0.0,
            "imb_l": (acc["bl"] - acc["sl"]) / tot_l if n_l else float("nan"),
            "cnt_imb_l": (acc["nbl"] - acc["nsl"]) / n_l if n_l else float("nan"),
            "large_imb_l": (acc["lb"] - acc["ls"]) / tot_l if n_l else float("nan"),
            "net_surge": max(-fl.SURGE_CLIP, min(fl.SURGE_CLIP, (acc["bs"] - acc["ss"]) / (tot_l * fl.SHORT_MIN / fl.LONG_MIN)))
            if n_l else float("nan"),
            "short_empty": 0.0 if acc["ns"] else 1.0, "n_long": n_l}


Q = D0 // 1000 + 7200                                            # a quote time two hours into the day (seconds)
QM = Q * 1000


def test_flow_features_by_hand():
    # long window [Q - 30 min, Q]; short window [Q - 5 min, Q]. Prices of 100 make notional = 100 x qty.
    trades = hand_trades([
        (QM - 31 * 60_000, 100.0, 500.0, True),      # before the long window: ignored ($50,000 buy)
        (QM - 20 * 60_000, 100.0, 150.0, True),      # long only: buy $15,000 (large)
        (QM - 10 * 60_000, 100.0, 30.0, False),      # long only: sell $3,000
        (QM - 4 * 60_000, 100.0, 20.0, True),        # short: buy $2,000
        (QM - 1 * 60_000, 100.0, 60.0, False),       # short: sell $6,000
        (QM + 1, 100.0, 900.0, True),                # one millisecond after the quote: must not be read
    ])
    f = fl.flow_features(trades, [Q]).iloc[0]
    # short: buy 2,000, sell 6,000 -> (2000 - 6000) / 8000
    assert f.imb_s == pytest.approx(-0.5)
    # long: buy 17,000, sell 9,000 -> 8000 / 26000
    assert f.imb_l == pytest.approx(8000 / 26000)
    # counts: 2 buys, 2 sells
    assert f.cnt_imb_l == 0.0 and f.n_long == 4
    # large: only the $15,000 buy -> 15000 / 26000
    assert f.large_imb_l == pytest.approx(15000 / 26000)
    # surge: short net -4,000 over what the long window implies for 5 minutes, 26000 x 5/30
    assert f.net_surge == pytest.approx(-4000 / (26000 * 5 / 30))
    assert f.short_empty == 0.0 and not f.hole
    assert f.thin                                                  # four trades is under the minimum of ten


def test_window_edges_are_inclusive_at_both_ends_and_nothing_after_the_quote_is_read():
    base = [(QM - 15 * 60_000 - 5000 * i, 100.0, 1.0, bool(i % 2)) for i in range(12)]      # filler, inside the long window
    at_quote = hand_trades(base + [(QM, 100.0, 50.0, True)])
    after = hand_trades(base + [(QM + 1, 100.0, 50.0, True)])
    assert fl.flow_features(at_quote, [Q]).iloc[0].imb_s == 1.0                           # T == t counts
    assert fl.flow_features(after, [Q]).iloc[0].short_empty == 1.0                        # T == t + 1 ms does not
    edge = hand_trades(base + [(QM - 5 * 60_000, 100.0, 50.0, False)])
    outside = hand_trades(base + [(QM - 5 * 60_000 - 1, 100.0, 50.0, False)])
    assert fl.flow_features(edge, [Q]).iloc[0].imb_s == -1.0                              # exactly 5 minutes back counts
    assert fl.flow_features(outside, [Q]).iloc[0].short_empty == 1.0
    long_edge = hand_trades([(QM - 30 * 60_000, 100.0, 1.0, True)])
    long_out = hand_trades([(QM - 30 * 60_000 - 1, 100.0, 1.0, True)])
    assert fl.flow_features(long_edge, [Q]).iloc[0].n_long == 1 and fl.flow_features(long_out, [Q]).iloc[0].n_long == 0


def test_flow_features_match_brute_force_on_random_trades():
    rng = np.random.default_rng(21)
    n = 6000
    ts = np.sort(D0 + rng.integers(0, 6 * 3_600_000, n))
    ts[100:140] = ts[100]                                          # a burst inside one millisecond
    trades = hand_trades(list(zip(ts.tolist(), rng.uniform(60_000, 80_000, n).tolist(),
                                  np.where(rng.random(n) < 0.03, rng.uniform(0.2, 2, n), rng.uniform(0.0001, 0.05, n)).tolist(),
                                  (rng.random(n) < 0.5).tolist())))
    quotes = [D0 // 1000 + 60 * k for k in range(1, 361, 7)] + [int(ts[100] // 1000) + 1]
    got = fl.flow_features(trades, quotes)
    saw_large = False
    for k, q in enumerate(quotes):
        want = brute_features(trades, q)
        for name, value in want.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(got[name].iloc[k]), name
            else:
                assert got[name].iloc[k] == pytest.approx(value, rel=1e-9, abs=1e-9), (name, q)
        saw_large = saw_large or abs(want["large_imb_l"] or 0) > 0
    assert saw_large                                               # the large-trade path was really exercised


def test_an_empty_short_window_reads_zero_and_says_so_rather_than_posing_as_balanced():
    filler = [(QM - 20 * 60_000 - 1000 * i, 100.0, 1.0, True) for i in range(12)]          # all buys, long window only
    f = fl.flow_features(hand_trades(filler), [Q]).iloc[0]
    assert f.imb_s == 0.0 and f.net_surge == 0.0 and f.short_empty == 1.0                 # exactly zero, flagged
    assert f.imb_l == 1.0 and not f.thin
    # a truly balanced short window is also 0, and is told apart by the flag
    bal = hand_trades(filler + [(QM - 1000, 100.0, 5.0, True), (QM - 2000, 100.0, 5.0, False)])
    g = fl.flow_features(bal, [Q]).iloc[0]
    assert g.imb_s == 0.0 and g.short_empty == 0.0


def test_an_empty_window_is_exactly_zero_despite_huge_running_sums():
    # prefix sums of ~1e12 either side of an empty window: subtraction residue must not become a reading
    early = [(QM - 3 * 3_600_000 + i, 97_123.37, 9_999.123, bool(i % 2)) for i in range(2000)]
    f = fl.flow_features(hand_trades(early), [Q]).iloc[0]
    assert f.n_long == 0 and f.imb_s == 0.0 and math.isnan(f.imb_l) and f.thin


def test_no_feature_changes_when_every_trade_after_the_quote_changes():
    rng = np.random.default_rng(22)
    ts = np.sort(D0 + rng.integers(0, 4 * 3_600_000, 3000))
    rows = list(zip(ts.tolist(), rng.uniform(60_000, 80_000, 3000).tolist(), rng.uniform(0.001, 0.4, 3000).tolist(),
                    (rng.random(3000) < 0.5).tolist()))
    changed = [(t, p, q * 50, not b) if t > QM else (t, p, q, b) for t, p, q, b in rows]
    a, b = fl.flow_features(hand_trades(rows), [Q]).iloc[0], fl.flow_features(hand_trades(changed), [Q]).iloc[0]
    for name in ("imb_s", "imb_l", "cnt_imb_l", "large_imb_l", "net_surge", "short_empty", "n_long"):
        assert a[name] == b[name], name
    # ...and it has teeth: a trade just before the quote does move them
    moved = fl.flow_features(hand_trades(rows + [(QM - 1, 70_000.0, 5.0, True)]), [Q]).iloc[0]
    assert moved.imb_s != a.imb_s and moved.imb_l != a.imb_l


def test_windows_with_hole_matches_brute_force():
    rng = np.random.default_rng(23)
    holes = [(int(a), int(a + w)) for a, w in zip(rng.integers(0, 10_000, 15), rng.integers(0, 300, 15))]
    lo = rng.integers(0, 10_000, 500)
    hi = lo + rng.integers(0, 400, 500)
    want = [any(a <= h and b >= l for a, b in holes) for l, h in zip(lo, hi)]
    assert list(fl.windows_with_hole(holes, lo, hi)) == want and 0 < sum(want) < 500
    assert not fl.windows_with_hole([], lo, hi).any()
    # touching counts: a hole ending exactly where the window starts, or starting exactly where it ends
    assert list(fl.windows_with_hole([(5, 10)], [10, 11, 0, 0], [20, 20, 5, 4])) == [True, False, True, False]


def test_a_row_whose_window_touches_a_hole_is_flagged_and_a_quiet_stretch_is_not():
    filler = [(QM - 25 * 60_000 + 1000 * i, 100.0, 1.0, bool(i % 2)) for i in range(20)]
    clean = fl.flow_features(hand_trades(filler), [Q, Q + 3600]).iloc
    assert not clean[0].hole and not clean[0].thin
    assert not clean[1].hole and clean[1].thin                    # an hour later: no trades, consecutive ids -> quiet, not a hole
    holed = fl.flow_features(hand_trades(filler, holes=[(QM - 40 * 60_000, QM - 29 * 60_000)]), [Q, Q + 3600]).iloc
    assert holed[0].hole and not holed[0].thin                    # a hole is reported as a hole, not as thin
    assert not holed[1].hole


# ---------- the dataset, the models and the verdicts, on a synthetic world ----------

def leaky_trades(cb, leak, seed=0, per_minute=4):
    """
    Trades in every minute of the Coinbase frame. With leak > 0 a trade is a taker buy with
    probability 0.5 + leak * sign(the NEXT six minutes' return): flow that knows where the
    price is going, which a forecast should be able to use. leak = 0 is a coin.
    """
    rng = np.random.default_rng(seed)
    close = cb["close"].to_numpy()
    n = len(close)
    future = np.sign(np.concatenate([close[6:], np.full(6, close[-1])]) - close)
    minute = np.repeat(np.arange(n), per_minute)
    ts = cb["ts"].to_numpy()[minute] * 1000 + rng.integers(0, 60_000, len(minute))
    order = np.argsort(ts, kind="stable")
    ts, minute = ts[order], minute[order]
    buy = rng.random(len(ts)) < 0.5 + leak * future[minute]
    return {"id": np.arange(1, len(ts) + 1, dtype=np.int64), "ts": ts.astype(np.int64),
            "price": close[minute] * 0.9995, "qty": rng.uniform(0.01, 0.3, len(ts)), "buy": buy, "holes": []}


def build_world(leak, seed=0, n_days=36, per_day=40):
    markets, candles, cb = synthetic_world(n_days=n_days, per_day=per_day, seed=seed, market="true")
    grid = ks.MinuteGrid(cb)
    trades = leaky_trades(cb, leak, seed=seed + 100)
    d = fl.add_flow(fc.build_dataset(markets, candles, grid, cb), trades, grid)
    return {"markets": markets, "candles": candles, "cb": cb, "grid": grid, "trades": trades, "d": d}


@pytest.fixture(scope="module")
def leaky():
    w = build_world(leak=0.35)
    w["fitted"] = fl.tune_and_fit(w["d"])
    w["pred"] = fl.predictions(w["fitted"], w["d"][w["d"].use & (w["d"].split == "test")])
    w["skills"] = fl.skill_table(w["pred"], ks.bonferroni_z(fl.FAMILY_TESTS))
    return w


def test_add_flow_keeps_every_row_and_scores_only_rows_with_a_flow_reading(leaky):
    d = leaky["d"]
    assert len(d) == 3 * len(leaky["markets"]) and not d.duplicated(["ticker", "horizon"]).any()
    assert (d.use == (d.usable & d.flow_ok)).all() and d.use.mean() > 0.95
    assert np.isfinite(d.loc[d.use, list(fl.FLOW_FEATURES)].to_numpy()).all()
    # the features in the frame are the brute-force ones at that row's quote time
    r = d[d.use].iloc[777]
    want = brute_features(leaky["trades"], int(r.quote_ts))
    for name in ("imb_s", "imb_l", "cnt_imb_l", "large_imb_l", "net_surge", "short_empty"):
        assert r[name] == pytest.approx(want[name], rel=1e-9, abs=1e-12), name
    # divergence: long imbalance minus the clipped, vol-scaled 30-minute Coinbase return
    close = leaky["cb"].set_index("ts")["close"]
    ret = math.log(close[r.quote_ts - 60] / close[r.quote_ts - 60 - 1800])
    scaled = ret / (r.vol / math.sqrt(fc.MINUTES_PER_YEAR) * math.sqrt(30))
    assert r.div_l == pytest.approx(r.imb_l - max(-1.0, min(1.0, scaled / 2)), rel=1e-9)
    hour = (int(r.quote_ts) // 3600) % 24
    assert r.us_hours == float(13 <= hour < 21) and set(d.us_hours.unique()) == {0.0, 1.0}


def test_rows_with_a_hole_or_too_few_trades_are_excluded_and_counted_not_imputed(leaky):
    trades = dict(leaky["trades"])
    target = leaky["d"][leaky["d"].use].iloc[500]
    t_ms = int(target.quote_ts) * 1000
    trades["holes"] = [(t_ms - 10 * 60_000, t_ms - 9 * 60_000)]    # one minute missing inside its long window
    d = fl.add_flow(leaky["d"].drop(columns=[c for c in leaky["d"].columns if c in
                                             ("hole", "thin", "flow_ok", "use", "n_long", "div_l", "us_hours") + fl.FLOW_FEATURES]),
                    trades, leaky["grid"])
    row = d[(d.ticker == target.ticker) & (d.horizon == target.horizon)].iloc[0]
    assert row.hole and not row.flow_ok and not row.use and row.usable
    assert 1 <= int(d.hole.sum()) < 60                             # only the rows whose window touches that minute
    # no trades at all: every row is thin, none is scored, and no feature is invented for them
    empty = {k: v[:0] for k, v in leaky["trades"].items() if k != "holes"}
    empty["holes"] = []
    e = fl.add_flow(leaky["d"][["ticker", "horizon", "quote_ts", "vol", "usable", "close_ts"]].head(50), empty, leaky["grid"])
    assert e.thin.all() and not e.use.any() and e.imb_l.isna().all()


def test_high_vol_threshold_comes_from_fit_rows_and_interactions_are_products(leaky):
    d, fitted = leaky["d"], leaky["fitted"]
    fit = d[d.use & (d.split == "fit")]
    assert fitted["vol_threshold"] == pytest.approx(float(fit.vol.median()))
    x = fl.design_frame(d[d.use], fitted["vol_threshold"])
    assert set(x.high_vol.unique()) == {0.0, 1.0}
    assert np.allclose(x.imb_s_x_high_vol, x.imb_s * x.high_vol) and np.allclose(x.imb_l_x_us_hours, x.imb_l * x.us_hours)
    assert np.allclose(x.logit_mid, fc.logit(d[d.use].mid))
    for m in fl.MODELS:
        assert set(fl.model_columns(m)) <= set(x.columns)


def test_tuning_and_fitting_never_read_the_held_out_rows(leaky):
    d = leaky["d"].copy()
    test = d.split == "test"
    d.loc[test, "outcome"] = 1 - d.loc[test, "outcome"]
    d.loc[test, ["mid", "p_asis", "imb_s", "imb_l", "vol"]] = 0.321
    again, first = fl.tune_and_fit(d), leaky["fitted"]
    assert again["l2"] == first["l2"] and again["vol_threshold"] == first["vol_threshold"]
    assert again["base_rate"] == first["base_rate"] and again["tuning"].equals(first["tuning"])
    for m in fl.MODELS:
        for h in ks.HORIZONS_MIN:
            assert np.array_equal(first["models"][m]["per_horizon"][h]["beta"], again["models"][m]["per_horizon"][h]["beta"])
    t = first["tuning"]
    assert (t.model == "market_base").sum() == 1 and (t.model == "both_flow").sum() == len(fc.L2_GRID)
    for m in fl.MODELS:
        best = t[t.model == m].sort_values("val_log_loss").iloc[0]
        assert first["l2"][m] == best.l2


def test_flow_that_knows_the_future_is_found_and_passes_the_rule(leaky):
    s = leaky["skills"]
    assert len(s) == fl.FAMILY_TESTS and not s.duplicated(["model", "reference", "metric"]).any()
    v = fl.flow_verdicts(s).set_index("model")
    assert v.loc["market_flow", "helps"] and v.loc["market_flow", "verdict"] == "FLOW HELPS"
    assert v.loc["both_flow", "helps"] and v.loc["model_flow", "beats_counterpart"]
    vs_market = s[(s.model == "market_flow") & (s.reference == "market")]
    assert (vs_market.verdict == "BETTER").all()                  # the market here knows only spot; flow knows more
    # the planted effect has the planted sign: taker buying raises P(YES)
    coef = fl.flow_coefficients(leaky["fitted"])
    net = coef[(coef.model == "market_flow") & coef.input.isin(["imb_s", "imb_l", "cnt_imb_l", "net_surge"])]
    assert net.groupby("horizon").beta_std.sum().gt(0).all()
    card = fl.scorecard(leaky["pred"], calibration_resamples=20).set_index("forecast")
    assert card.loc["market_flow", "brier"] < card.loc["market_base", "brier"]
    assert list(card.index) == list(fl.BASELINES + fl.MODELS)
    assert math.isnan(card.loc["base_rate", "cal_b"]) and card.loc["base_rate", "sharp_sd"] < 1e-9
    assert set(card.kind) == {"baseline", "flow", "flow-free"}


def test_scorecard_numbers_are_plain_means_of_the_edge_studys_losses(leaky):
    pred = leaky["pred"]
    card = fl.scorecard(pred, calibration_resamples=20).set_index("forecast")
    y = pred.outcome.to_numpy()
    for name in ("market", "both_flow"):
        p = pred[name].to_numpy()
        brier = sum((a - b) ** 2 for a, b in zip(p, y)) / len(y)
        ll = -sum(math.log(min(max(a, 1e-4), 1 - 1e-4)) if b else math.log(1 - min(max(a, 1e-4), 1 - 1e-4)) for a, b in zip(p, y)) / len(y)
        assert card.loc[name, "brier"] == pytest.approx(brier) and card.loc[name, "log_loss"] == pytest.approx(ll)
        assert card.loc[name, "sharp_sd"] == pytest.approx(float(np.std(p)))
        assert card.loc[name, "cal_b"] == pytest.approx(ks.calibration_fit(p, y)[1])
    # each skill row is fc.skill on the same losses
    z = ks.bonferroni_z(fl.FAMILY_TESTS)
    row = leaky["skills"][(leaky["skills"].model == "both_flow") & (leaky["skills"].reference == "both_base") & (leaky["skills"].metric == "brier")].iloc[0]
    want = fc.skill(ks.brier(pred.both_flow, y), ks.brier(pred.both_base, y), pred.day.to_numpy(), z)
    assert (row.skill, row.lo, row.hi) == pytest.approx((want["skill"], want["lo"], want["hi"]))


def test_flow_that_is_a_coin_is_not_called_helpful():
    w = build_world(leak=0.0, seed=5, n_days=30, per_day=32)
    pred = fl.predictions(fl.tune_and_fit(w["d"]), w["d"][w["d"].use & (w["d"].split == "test")])
    v = fl.flow_verdicts(fl.skill_table(pred, ks.bonferroni_z(fl.FAMILY_TESTS)))
    assert not v.helps.any() and not v.beats_counterpart.any()
    assert set(v.verdict) <= {"no detectable effect", "flow makes it worse"}


def skills_frame(cp_brier, cp_ll, mk_brier, mk_ll):
    rows = []
    for m in fl.FLOW_MODELS:
        for metric, verdict in (("brier", cp_brier), ("log_loss", cp_ll)):
            rows.append({"model": m, "reference": fl.COUNTERPART[m], "kind": "counterpart", "metric": metric, "verdict": verdict})
        for metric, verdict in (("brier", mk_brier), ("log_loss", mk_ll)):
            rows.append({"model": m, "reference": "market", "kind": "baseline", "metric": metric, "verdict": verdict})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("cp_brier, cp_ll, mk_brier, mk_ll, helps, text", [
    ("BETTER", "BETTER", "not distinguishable", "not distinguishable", True, "FLOW HELPS"),
    ("BETTER", "BETTER", "BETTER", "not distinguishable", True, "FLOW HELPS"),
    ("BETTER", "not distinguishable", "BETTER", "BETTER", False, "no detectable effect"),      # one metric is not enough
    ("BETTER", "BETTER", "WORSE", "not distinguishable", False, "better than its counterpart but worse than the market: does not count"),
    ("not distinguishable", "not distinguishable", "not distinguishable", "not distinguishable", False, "no detectable effect"),
    ("WORSE", "not distinguishable", "WORSE", "WORSE", False, "flow makes it worse"),
])
def test_the_verdict_rule(cp_brier, cp_ll, mk_brier, mk_ll, helps, text):
    v = fl.flow_verdicts(skills_frame(cp_brier, cp_ll, mk_brier, mk_ll))
    assert len(v) == 4 and (v.helps == helps).all() and (v.verdict == text).all()


# ---------- the split and the pristine-set check ----------

def test_the_existing_split_is_left_alone_and_a_pristine_set_makes_everything_earlier_train(leaky):
    d = leaky["d"]
    assert fl.assign_splits(d) is d
    closes = np.sort(d.close_ts.unique())
    after = int(closes[-200])                                      # pretend the last 199 contracts are new
    p = fl.assign_splits(d, pristine_after=after)
    assert (p[p.close_ts > after].split == "test").all() and p[p.split == "test"].ticker.nunique() == 199
    train = closes[closes <= after]
    val_cut = ks.split_cutoff(train, fc.VALIDATION_FRACTION)
    assert p[p.split == "validation"].close_ts.min() == val_cut and p[p.split == "fit"].close_ts.max() < val_cut
    assert p[p.split == "validation"].close_ts.max() <= after
    assert (d.split != p.split).any() and len(p) == len(d)


def test_counting_new_contracts_uses_the_public_listing_and_only_cleanly_settled_ones(monkeypatch):
    seen = {}

    def fake_iter(session, path, params):
        seen.update(path=path, params=params)
        yield {"ticker": "A", "result": "yes", "floor_strike": 70_000, "open_time": "2026-11-10T17:30:00Z", "close_time": "2026-11-10T17:45:00Z"}
        yield {"ticker": "B", "result": "no", "floor_strike": 70_000, "open_time": "2026-11-10T17:45:00Z", "close_time": "2026-11-10T18:00:00Z"}
        yield {"ticker": "C", "result": "", "floor_strike": 70_000, "open_time": "2026-11-10T18:00:00Z", "close_time": "2026-11-10T18:15:00Z"}
        yield {"ticker": "D", "result": "yes", "floor_strike": None, "open_time": "2026-11-10T18:15:00Z", "close_time": "2026-11-10T18:30:00Z"}
        yield {"ticker": "E", "result": "yes", "floor_strike": 70_000, "open_time": "2026-11-10T17:15:00Z", "close_time": "2026-11-10T17:30:00Z"}
    monkeypatch.setattr(ks, "iter_markets", fake_iter)
    last = 1_794_331_800                                           # 2026-11-10 17:30:00 UTC
    assert fl.count_new_settled(object(), last) == 2               # A and B; C unsettled, D has no strike, E is not newer
    assert seen["path"] == "/markets" and seen["params"]["min_close_ts"] == last + 1
    assert seen["params"]["series_ticker"] == "KXBTC15M" and seen["params"]["status"] == "settled"


def test_meta_round_trips_and_is_empty_when_absent(tmp_path):
    assert fl.read_meta(str(tmp_path)) == {}
    fl.write_meta(str(tmp_path), {"new_settled_contracts": 8, "pristine_after": None})
    assert fl.read_meta(str(tmp_path)) == {"new_settled_contracts": 8, "pristine_after": None}
