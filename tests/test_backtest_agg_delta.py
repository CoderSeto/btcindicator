"""
Tests for backtest_agg_delta.py.

reconstruct_agg_delta_series is cross-validated against a deliberately naive
O(n*m) brute force that re-scans every trade for every candle (no sorting
assumption, no incremental add/subtract), which is what would expose a bug
in the two-pointer sliding window.

run_agg_delta_backtest is exercised end to end with the network layers
(candle history, Binance.US trades) stubbed, on a hand-constructed scenario
whose hit/total counts are worked out in the comments. That scenario is run
at several datetime resolutions to pin the pandas-resolution bug from
CLAUDE.md gotcha #2 (datetime64[s] silently becoming near-1970 timestamps).
"""
import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest
import backtest_agg_delta as bad
import exchange_feeds


# ---------- reconstruct_agg_delta_series ----------

def _brute_force(trades, candle_times, window):
    out = []
    for t in candle_times:
        buy = sell = 0.0
        for tr in trades:
            if t - window <= tr["time"] <= t:
                n = tr["price"] * tr["size"]
                if tr["side"] == "buy":
                    buy += n
                else:
                    sell += n
        total = buy + sell
        out.append({"delta_ratio": (buy - sell) / total if total > 0 else None,
                    "total_notional": total})
    return out


def _tr(time, side, price=100.0, size=1.0):
    return {"time": time, "side": side, "price": price, "size": size}


def test_reconstruct_matches_brute_force_on_random_trades():
    rng = random.Random(2024)
    for _ in range(25):
        trades = [_tr(rng.uniform(0, 600), rng.choice(["buy", "sell"]),
                      rng.uniform(50, 150), rng.uniform(0.01, 3)) for _ in range(rng.randint(0, 200))]
        rng.shuffle(trades)  # input order must not matter
        candle_times = sorted(rng.uniform(-30, 650) for _ in range(rng.randint(1, 60)))
        window = rng.choice([10, 60, 120])
        got = bad.reconstruct_agg_delta_series(trades, candle_times, window)
        want = _brute_force(trades, candle_times, window)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g["total_notional"] == pytest.approx(w["total_notional"], abs=1e-9)
            if w["delta_ratio"] is None:
                assert g["delta_ratio"] is None
            else:
                assert g["delta_ratio"] == pytest.approx(w["delta_ratio"], abs=1e-9)


def test_reconstruct_window_is_inclusive_at_both_ends():
    trades = [_tr(940.0, "buy"), _tr(1000.0, "sell")]
    # t=1000, window 60 -> [940, 1000]: both counted.
    r = bad.reconstruct_agg_delta_series(trades, [1000.0], 60)[0]
    assert r["total_notional"] == 200.0 and r["delta_ratio"] == 0.0
    # A hair later, the 940 trade has aged out.
    r = bad.reconstruct_agg_delta_series(trades, [1000.001], 60)[0]
    assert r["total_notional"] == 100.0 and r["delta_ratio"] == -1.0
    # A trade after t is not counted.
    r = bad.reconstruct_agg_delta_series(trades, [999.999], 60)[0]
    assert r["total_notional"] == 100.0 and r["delta_ratio"] == 1.0


def test_reconstruct_hand_computed_ratio():
    trades = [_tr(10, "buy", 100, 3), _tr(11, "sell", 200, 1)]  # buy 300, sell 200
    r = bad.reconstruct_agg_delta_series(trades, [12.0], 60)[0]
    assert r["total_notional"] == 500.0
    assert r["delta_ratio"] == pytest.approx(0.2)


def test_reconstruct_no_trades_gives_none_and_zero():
    assert bad.reconstruct_agg_delta_series([], [1.0, 2.0], 60) == [
        {"delta_ratio": None, "total_notional": 0.0}] * 2
    # Trades exist but none within any candle's window.
    out = bad.reconstruct_agg_delta_series([_tr(0, "buy")], [500.0], 60)
    assert out[0]["delta_ratio"] is None and out[0]["total_notional"] == 0.0


def test_reconstruct_empty_candles():
    assert bad.reconstruct_agg_delta_series([_tr(1, "buy")], [], 60) == []


def test_reconstruct_does_not_mutate_input_order():
    trades = [_tr(5, "buy"), _tr(1, "sell")]
    bad.reconstruct_agg_delta_series(trades, [6.0], 60)
    assert [t["time"] for t in trades] == [5, 1]


# ---------- run_agg_delta_backtest (end to end, network stubbed) ----------

T0 = 1_780_000_000          # 2026-ish, comfortably above _MIN_PLAUSIBLE_MS
N = 30
LOOKAHEAD = 2
WARMUP = 2


def _cfg(tmp_path):
    return SimpleNamespace(
        PRODUCT_ID="BTC-USD", GRANULARITY=60, BACKTEST_HISTORY_CANDLES=N,
        BACKTEST_REQUEST_PAUSE=0, BACKTEST_LOOKAHEAD_CANDLES=LOOKAHEAD,
        BACKTEST_RESULTS_PATH=str(tmp_path / "backtest_results.json"),
        BACKTEST_MIN_SAMPLE=15, AGG_WINDOW_SECONDS=60, AGG_DELTA_THRESHOLD=0.15,
        AGG_DELTA_BACKTEST_HOURS=24, AGG_DELTA_BACKTEST_REQUEST_PAUSE=0,
        AGG_DELTA_BACKTEST_MAX_REQUESTS=10,
    )


def _candles(unit="s", start=T0):
    times = np.array([start + 60 * k for k in range(N)], dtype="int64").astype(f"datetime64[{unit}]") \
        if unit == "s" else (np.array([start + 60 * k for k in range(N)], dtype="datetime64[s]").astype(f"datetime64[{unit}]"))
    closes = [100 + k if k <= 15 else 130 - k for k in range(N)]
    return pd.DataFrame({"time": pd.Series(times), "close": [float(c) for c in closes]})


def _scenario_trades(skip=(20,)):
    # One trade 30s before each candle close, so each window holds exactly that
    # trade: ratio is +1 (buy, when k % 3 == 0) or -1 (sell otherwise).
    return [_tr(T0 + 60 * k - 30, "buy" if k % 3 == 0 else "sell") for k in range(N) if k not in skip]


def _ht(bucket):
    return {"hits": bucket["hits"], "total": bucket["total"]}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    base = {"indicator_buckets": {"rsi_bullish": {"hits": 7, "total": 10}}, "marker": 1}
    state = {"df": _candles(), "trades": _scenario_trades(), "base": base, "calls": {}}

    monkeypatch.setattr(backtest, "load_results", lambda c: state["base"])
    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: state["df"])
    monkeypatch.setattr(backtest, "_warmup_candles", lambda c: WARMUP)
    monkeypatch.setattr(exchange_feeds, "to_binance_symbol", lambda p: "BTCUSDT")

    def fake_fetch(symbol, start_ms, end_ms, pause=0, max_requests=0):
        state["calls"] = {"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms}
        return state["trades"]
    monkeypatch.setattr(exchange_feeds, "fetch_binance_us_trade_history", fake_fetch)
    return cfg, state


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_end_to_end_hit_rates_at_every_datetime_resolution(stubbed, unit):
    cfg, state = stubbed
    state["df"] = _candles(unit)
    res = bad.run_agg_delta_backtest(cfg)
    assert "error" not in res

    # Hand-worked (warmup=2, lookahead=2 -> i = 2..27):
    #   closes rise through i=15 then fall; ratio is +1 when i%3==0 else -1.
    #   Bullish calls: i=3,6,9,12 (future higher -> 4 hits), i=15 (future lower -> miss),
    #                  i=18,21,24,27 (future lower -> 4 misses)      => 4 / 9
    #   Bearish calls: i=2,4,5,7,8,10,11,13 (future higher -> 8 misses),
    #                  i=14 exact tie -> skipped,
    #                  i=20 no trade coverage -> skipped,
    #                  i=16,17,19,22,23,25,26 (future lower -> 7 hits)  => 7 / 15
    assert _ht(res["indicator_buckets"]["agg_delta_bullish"]) == {"hits": 4, "total": 9}
    assert _ht(res["indicator_buckets"]["agg_delta_bearish"]) == {"hits": 7, "total": 15}

    meta = res["agg_delta_backtest"]
    assert meta["source"] == "binance_us_only"
    assert meta["trades_fetched"] == N - 1
    assert meta["candles_with_coverage"] == N - 1
    assert meta["candles_total"] == N


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_end_to_end_requests_correct_real_world_time_range(stubbed, unit):
    cfg, state = stubbed
    state["df"] = _candles(unit)
    bad.run_agg_delta_backtest(cfg)
    last_ms = (T0 + 60 * (N - 1)) * 1000
    assert state["calls"]["end_ms"] == last_ms           # not a near-1970 value
    assert state["calls"]["start_ms"] == T0 * 1000       # 24h cap > candle span
    assert state["calls"]["symbol"] == "BTCUSDT"


def test_end_to_end_merges_into_base_and_writes_file(stubbed, tmp_path):
    cfg, state = stubbed
    bad.run_agg_delta_backtest(cfg)
    saved = json.loads((tmp_path / "backtest_results.json").read_text())
    assert saved["marker"] == 1
    assert saved["indicator_buckets"]["rsi_bullish"] == {"hits": 7, "total": 10}   # untouched
    assert _ht(saved["indicator_buckets"]["agg_delta_bullish"]) == {"hits": 4, "total": 9}


def test_end_to_end_omits_bucket_with_no_directional_calls(stubbed):
    cfg, state = stubbed
    state["trades"] = [_tr(T0 + 60 * k - 30, "buy") for k in range(N)]   # only bullish calls
    res = bad.run_agg_delta_backtest(cfg)
    assert "agg_delta_bullish" in res["indicator_buckets"]
    assert "agg_delta_bearish" not in res["indicator_buckets"]


def test_end_to_end_balanced_flow_is_neutral(stubbed):
    cfg, state = stubbed
    # Equal buy and sell notional in every window -> ratio 0, below threshold.
    state["trades"] = [t for k in range(N) for t in (_tr(T0 + 60 * k - 30, "buy"), _tr(T0 + 60 * k - 20, "sell"))]
    res = bad.run_agg_delta_backtest(cfg)
    assert "agg_delta_bullish" not in res["indicator_buckets"]
    assert "agg_delta_bearish" not in res["indicator_buckets"]


# ---------- error paths: nothing gets written ----------

def test_error_without_base_backtest(stubbed, monkeypatch, tmp_path):
    cfg, _ = stubbed
    monkeypatch.setattr(backtest, "load_results", lambda c: None)
    assert "run `python backtest.py` first" in bad.run_agg_delta_backtest(cfg)["error"]
    monkeypatch.setattr(backtest, "load_results", lambda c: {"error": "x"})
    assert "error" in bad.run_agg_delta_backtest(cfg)
    assert not (tmp_path / "backtest_results.json").exists()


def test_error_without_symbol_mapping(stubbed, monkeypatch):
    cfg, _ = stubbed
    monkeypatch.setattr(exchange_feeds, "to_binance_symbol", lambda p: None)
    assert "No Binance.US symbol mapping" in bad.run_agg_delta_backtest(cfg)["error"]


def test_error_with_too_few_candles(stubbed):
    cfg, state = stubbed
    state["df"] = _candles().head(10)      # needs warmup + lookahead + 10 = 14
    assert "Only got 10 candles" in bad.run_agg_delta_backtest(cfg)["error"]
    state["df"] = _candles().head(0)
    assert "error" in bad.run_agg_delta_backtest(cfg)


def test_error_when_no_trades_returned(stubbed, tmp_path):
    cfg, state = stubbed
    state["trades"] = []
    assert "No Binance.US trades" in bad.run_agg_delta_backtest(cfg)["error"]
    assert not (tmp_path / "backtest_results.json").exists()


def test_sanity_check_catches_implausible_timestamps(stubbed, tmp_path):
    cfg, state = stubbed
    state["df"] = _candles(start=1_000)     # 1970 -- what the old resolution bug produced
    res = bad.run_agg_delta_backtest(cfg)
    assert "implausible end time" in res["error"]
    assert not (tmp_path / "backtest_results.json").exists()



def test_agg_delta_buckets_carry_their_own_base_rate_and_n_eff(stubbed):
    # Scored candles (covered, not tied), i = 2..27 minus 14 (tie) and 20 (no coverage) = 24:
    # up at i = 2..13 (12), down at 15..27 minus 20 (12) -> base rate 0.5 each way, measured over
    # this backtest's own window rather than borrowed from the candle backtest's.
    cfg, state = stubbed
    res = bad.run_agg_delta_backtest(cfg)
    bull = res["indicator_buckets"]["agg_delta_bullish"]
    assert bull["base_rate"] == pytest.approx(0.5)
    assert res["agg_delta_backtest"]["base_rates"] == {"bullish": 0.5, "bearish": 0.5, "total": 24}
    # Bullish calls sit 3 candles apart, lookahead is 2: no overlap, n_eff == n.
    assert bull["n_eff"] == 9
    # Bearish calls include runs of adjacent candles (4,5 / 7,8 / ...): overlapping windows.
    bear = res["indicator_buckets"]["agg_delta_bearish"]
    assert bear["n_eff"] < bear["total"]
