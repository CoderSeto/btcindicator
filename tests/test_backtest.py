"""
Tests for backtest.py.

run_backtest is tested with the signal/divergence/indicator layers replaced
by scripted fakes, so every expected hit and total can be worked out by hand
from a price series and a vote schedule (the arithmetic is in the comments).
That isolates what backtest.py itself owns: the walk-forward loop, the
lookahead outcome, tie handling, bucket accounting and persistence. A final
pair of tests runs the real indicator stack to check that the pipeline holds
together and that indicators are causal (no look-ahead leaking into signals).

fetch_history is tested against a fake exchange that answers range queries
from a fixed candle series, capped at 300 per request like Coinbase.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest
import config as real_cfg
import data_fetch
import divergence
import indicators
import signals


# ---------- fetch_history ----------

def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


class FakeExchange:
    """Holds `n` one-minute candles ending now; answers (start, end) range
    queries with the newest <= 300 candles inside the range."""

    def __init__(self, n, granularity=60):
        end = _naive(pd.Timestamp.now(tz="UTC")).floor("min")
        times = [end - pd.Timedelta(seconds=granularity * k) for k in range(n)][::-1]
        self.df = pd.DataFrame({
            "time": pd.to_datetime(times), "low": 1.0, "high": 3.0, "open": 2.0,
            "close": [float(i) for i in range(n)], "volume": 10.0,
        })
        self.calls = []

    def __call__(self, product_id, granularity, start, end):
        self.calls.append((start, end))
        s, e = _naive(start), _naive(end)
        sub = self.df[(self.df["time"] >= s) & (self.df["time"] <= e)].tail(300)
        return sub.reset_index(drop=True)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(backtest.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_fetch_history_paginates_dedupes_sorts_and_trims(monkeypatch, no_sleep):
    ex = FakeExchange(1000)
    monkeypatch.setattr(data_fetch, "get_candles_range", ex)
    out = backtest.fetch_history("BTC-USD", 60, 700, pause=0.25)

    assert len(out) == 700                                    # trimmed to the target
    assert out["time"].is_monotonic_increasing
    assert out["time"].is_unique                              # page boundaries overlap by one candle
    assert list(out["close"]) == [float(i) for i in range(300, 1000)]   # the newest 700, in order
    assert len(ex.calls) >= 3 and no_sleep and set(no_sleep) == {0.25}
    assert out.index.tolist() == list(range(700))


def test_fetch_history_returns_everything_when_history_runs_out(monkeypatch, no_sleep):
    ex = FakeExchange(450)
    monkeypatch.setattr(data_fetch, "get_candles_range", ex)
    out = backtest.fetch_history("BTC-USD", 60, 5000, pause=0)
    assert len(out) == 450 and out["time"].is_unique


def test_fetch_history_keeps_partial_data_after_a_request_error(monkeypatch, no_sleep, capsys):
    ex = FakeExchange(1000)
    state = {"n": 0}

    def flaky(*a):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        return ex(*a)
    monkeypatch.setattr(data_fetch, "get_candles_range", flaky)
    out = backtest.fetch_history("BTC-USD", 60, 900, pause=0)
    assert len(out) == 300                                    # just the first page
    assert "stopping pagination early" in capsys.readouterr().out


def test_fetch_history_no_data_gives_empty_frame_with_columns(monkeypatch, no_sleep):
    monkeypatch.setattr(data_fetch, "get_candles_range",
                        lambda *a: pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"]))
    out = backtest.fetch_history("BTC-USD", 60, 500, pause=0)
    assert out.empty and list(out.columns) == ["time", "low", "high", "open", "close", "volume"]

    def raises(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(data_fetch, "get_candles_range", raises)
    assert backtest.fetch_history("BTC-USD", 60, 500, pause=0).empty


def test_fetch_history_stops_on_a_single_candle_page(monkeypatch, no_sleep):
    calls = []

    def one(product, gran, start, end):
        calls.append(1)
        return pd.DataFrame({"time": [pd.Timestamp("2026-01-01")], "low": 1.0, "high": 1.0,
                             "open": 1.0, "close": 1.0, "volume": 1.0})
    monkeypatch.setattr(data_fetch, "get_candles_range", one)
    out = backtest.fetch_history("BTC-USD", 60, 1000, pause=0)
    assert len(out) == 1 and len(calls) == 1                  # no infinite loop


# ---------- warmup ----------

def test_warmup_is_largest_lookback_plus_five():
    c = SimpleNamespace(SMA_SLOW=21, BOLLINGER_PERIOD=20, MACD_SLOW=26, MACD_SIGNAL=9,
                        RSI_PERIOD=14, VOLUME_SPIKE_WINDOW=20, DIVERGENCE_LOOKBACK=30)
    assert backtest._warmup_candles(c) == 35 + 5              # MACD: 26 + 9
    c.DIVERGENCE_LOOKBACK = 50
    assert backtest._warmup_candles(c) == 55


# ---------- run_backtest, scripted signals ----------

N = 30
LOOKAHEAD = 2
SCRIPT_CFG = dict(
    PRODUCT_ID="BTC-USD", GRANULARITY=60, BACKTEST_HISTORY_CANDLES=N, BACKTEST_REQUEST_PAUSE=0,
    BACKTEST_LOOKAHEAD_CANDLES=LOOKAHEAD,
    SMA_SLOW=3, BOLLINGER_PERIOD=3, MACD_SLOW=2, MACD_SIGNAL=1, RSI_PERIOD=3,
    VOLUME_SPIKE_WINDOW=3, DIVERGENCE_LOOKBACK=4, DIVERGENCE_SWING_ORDER=2,
)
WARMUP = 9                                                    # max(3,3,3,3,3,4) + 5


def script_closes():
    # Rises through k=15, then falls: close[15]=115, close[16]=114, close[17]=113 ...
    return [float(100 + k if k <= 15 else 130 - k) for k in range(N)]


def actual_dir(i):
    """Independent restatement of the outcome rule, for the hand-worked cases."""
    c = script_closes()
    f, p = c[i + LOOKAHEAD], c[i]
    return "bullish" if f > p else "bearish" if f < p else None


def script_votes(i):
    rsi = "bullish" if i % 2 == 0 else "bearish"
    macd = {0: "bullish", 1: "neutral", 2: "bearish"}[i % 3]
    return {"rsi": rsi, "macd": macd}


def script_div(i):
    return {0: "bullish", 1: "bearish"}.get(i % 5)


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    cfg = SimpleNamespace(**SCRIPT_CFG, BACKTEST_RESULTS_PATH=str(tmp_path / "bt.json"))
    df = pd.DataFrame({
        "idx": range(N), "close": script_closes(),
        "time": pd.date_range("2026-01-01", periods=N, freq="min"),
    })
    seen = {"eval": [], "div": []}

    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: df.copy())
    monkeypatch.setattr(indicators, "compute_all", lambda d, c: d)

    def fake_eval(pair, c):
        assert len(pair) == 2                                # crossover-aware: last two rows only
        i = int(pair.iloc[-1]["idx"])
        assert int(pair.iloc[0]["idx"]) == i - 1
        seen["eval"].append(i)
        return signals.SignalSnapshot(price=float(pair.iloc[-1]["close"]), individual=script_votes(i))
    monkeypatch.setattr(signals, "evaluate_price_only", fake_eval)

    def fake_div(window, lookback, order):
        i = int(window.iloc[-1]["idx"])
        assert (lookback, order) == (SCRIPT_CFG["DIVERGENCE_LOOKBACK"], SCRIPT_CFG["DIVERGENCE_SWING_ORDER"])
        assert len(window) == min(i + 1, lookback)           # trailing window, ending at i
        seen["div"].append(i)
        return {"type": script_div(i)}
    monkeypatch.setattr(divergence, "detect_divergence", fake_div)
    return cfg, seen, df


def test_walk_forward_covers_exactly_warmup_to_end_minus_lookahead(scripted):
    cfg, seen, _ = scripted
    backtest.run_backtest(cfg)
    assert seen["eval"] == list(range(WARMUP, N - LOOKAHEAD))   # 9..27: never peeks past the data
    assert seen["div"] == seen["eval"]


def test_indicator_buckets_hand_computed(scripted):
    cfg, _, _ = scripted
    b = backtest.run_backtest(cfg)["indicator_buckets"]
    # actual: i<=13 bullish, i=14 tie (skipped), i>=15 bearish.
    # rsi bullish at even i: 10,12 hit; 16,18,20,22,24,26 miss              -> 2/8
    assert b["rsi_bullish"] == {"hits": 2, "total": 8}
    # rsi bearish at odd i: 9,11,13 miss; 15,17,19,21,23,25,27 hit          -> 7/10
    assert b["rsi_bearish"] == {"hits": 7, "total": 10}
    # macd bullish at i%3==0: 9,12 hit; 15,18,21,24,27 miss                 -> 2/7
    assert b["macd_bullish"] == {"hits": 2, "total": 7}
    # macd bearish at i%3==2: 11 miss; 14 tie skipped; 17,20,23,26 hit      -> 4/5
    assert b["macd_bearish"] == {"hits": 4, "total": 5}
    assert not any(k.startswith("macd_neutral") for k in b)       # neutral votes are not calls


def test_divergence_buckets_hand_computed(scripted):
    cfg, _, _ = scripted
    d = backtest.run_backtest(cfg)["divergence_buckets"]
    # bullish at i%5==0: 10 hit; 15,20,25 miss; bearish at i%5==1: 11 miss; 16,21,26 hit
    assert d["bullish"] == {"hits": 1, "total": 4}
    assert d["bearish"] == {"hits": 3, "total": 4}


def test_strength_buckets_match_independent_tally(scripted):
    cfg, _, _ = scripted
    got = backtest.run_backtest(cfg)["strength_buckets"]
    want = {}
    for i in range(WARMUP, N - LOOKAHEAD):
        snap = signals.SignalSnapshot(price=0.0, individual=script_votes(i))
        actual = actual_dir(i)
        if actual is None or snap.composite == "neutral":
            continue
        h, t = want.get(f"{snap.composite}_{snap.strength}", (0, 0))
        want[f"{snap.composite}_{snap.strength}"] = (h + (actual == snap.composite), t + 1)
    assert got == {k: {"hits": h, "total": t} for k, (h, t) in want.items()}
    # Spot-check: i=9 votes rsi bearish / macd bullish -> neutral composite, never bucketed.
    # 18 non-tie positions, minus the 6 where rsi and macd oppose (i=9,15,20,21,26,27) -> neutral
    assert sum(v["total"] for v in got.values()) == 12
    assert all(not k.startswith("neutral") for k in got)


def test_exact_ties_are_skipped_everywhere(scripted):
    cfg, _, df = scripted
    res = backtest.run_backtest(cfg)
    # i=14 has close[14]==close[16]; its calls (rsi bullish, macd bearish, i%5==4 no div) are uncounted.
    assert actual_dir(14) is None
    n_rsi = sum(v["total"] for k, v in res["indicator_buckets"].items() if k.startswith("rsi_"))
    assert n_rsi == (N - LOOKAHEAD - WARMUP) - 1                  # 19 positions minus the tie


def test_results_metadata_and_file(scripted):
    cfg, _, _ = scripted
    res = backtest.run_backtest(cfg)
    assert (res["product_id"], res["granularity"], res["lookahead_candles"], res["candles_used"]) == \
           ("BTC-USD", 60, LOOKAHEAD, N)
    assert pd.Timestamp(res["generated_at"]).tzinfo is not None
    with open(cfg.BACKTEST_RESULTS_PATH) as f:
        assert json.load(f) == res
    assert backtest.load_results(cfg) == res                      # and it round-trips through the loader


def test_run_backtest_errors_without_writing(scripted, monkeypatch, tmp_path):
    cfg, _, df = scripted
    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: df.head(WARMUP + LOOKAHEAD + 9))
    res = backtest.run_backtest(cfg)
    assert res["error"].startswith(f"Only got {WARMUP + LOOKAHEAD + 9} historical candles")
    assert f"at least {WARMUP + LOOKAHEAD + 10}" in res["error"]
    assert (res["product_id"], res["granularity"]) == ("BTC-USD", 60)
    assert not (tmp_path / "bt.json").exists()

    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: df.head(0))
    assert "error" in backtest.run_backtest(cfg)

    # Exactly the minimum is enough.
    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: df.head(WARMUP + LOOKAHEAD + 10))
    assert "error" not in backtest.run_backtest(cfg)


# ---------- run_backtest, real indicator stack ----------

def _synthetic_ohlcv(n=500, seed=11):
    rng = np.random.default_rng(seed)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="min"),
        "open": np.r_[close[0], close[:-1]], "close": close,
        "high": close * 1.0005, "low": close * 0.9995,
        "volume": rng.uniform(5, 15, n),
    })


def _real_cfg(tmp_path):
    ns = SimpleNamespace(**{k: getattr(real_cfg, k) for k in dir(real_cfg) if k.isupper()})
    ns.BACKTEST_RESULTS_PATH = str(tmp_path / "bt.json")
    ns.BACKTEST_HISTORY_CANDLES = 500
    ns.BACKTEST_REQUEST_PAUSE = 0
    return ns


def test_real_pipeline_produces_consistent_buckets(monkeypatch, tmp_path):
    cfg = _real_cfg(tmp_path)
    monkeypatch.setattr(backtest, "fetch_history", lambda *a, **k: _synthetic_ohlcv())
    res = backtest.run_backtest(cfg)
    assert "error" not in res

    assert res["indicator_buckets"], "a 500-candle random walk should trigger at least some indicator calls"
    for section in ("strength_buckets", "divergence_buckets", "indicator_buckets"):
        for key, b in res[section].items():
            assert 0 <= b["hits"] <= b["total"], (section, key, b)
    for key in res["indicator_buckets"]:
        assert key.rsplit("_", 1)[1] in ("bullish", "bearish")
    # Per-indicator buckets can only hold the 6 candle signals (no order-book, no agg_delta here).
    names = {k.rsplit("_", 1)[0] for k in res["indicator_buckets"]}
    assert names <= {"sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume"}
    # The composite can never be called more often than there are scored positions.
    positions = len(_synthetic_ohlcv()) - cfg.BACKTEST_LOOKAHEAD_CANDLES - backtest._warmup_candles(cfg)
    assert sum(b["total"] for b in res["strength_buckets"].values()) <= positions
    # Deterministic: same data, same answer.
    again = backtest.run_backtest(cfg)
    for section in ("strength_buckets", "divergence_buckets", "indicator_buckets"):
        assert again[section] == res[section]


def test_indicators_are_causal_so_the_backtest_cannot_see_the_future():
    """backtest.py computes indicators once over the whole frame and then reads
    row i. That is only honest if row i never depends on rows after it."""
    df = _synthetic_ohlcv()
    full = indicators.compute_all(df.copy(), real_cfg)
    cut = 300
    part = indicators.compute_all(df.iloc[:cut].copy(), real_cfg)
    cols = [c for c in full.columns if c not in df.columns]
    assert cols, "expected indicator columns to be added"
    for c in cols:
        a, b = full[c].iloc[:cut].reset_index(drop=True), part[c].reset_index(drop=True)
        pd.testing.assert_series_equal(a, b, check_names=False, check_exact=False, rtol=1e-9, atol=1e-9,
                                       obj=f"indicator column {c!r}")


# ---------- load_results ----------

def _write(tmp_path, obj, raw=None):
    p = tmp_path / "r.json"
    p.write_text(raw if raw is not None else json.dumps(obj))
    return SimpleNamespace(PRODUCT_ID="BTC-USD", GRANULARITY=60, BACKTEST_RESULTS_PATH=str(p))


def test_load_results_missing_and_corrupt(tmp_path):
    cfg = SimpleNamespace(PRODUCT_ID="BTC-USD", GRANULARITY=60, BACKTEST_RESULTS_PATH=str(tmp_path / "none.json"))
    assert backtest.load_results(cfg) is None
    assert backtest.load_results(_write(tmp_path, None, raw="{oops")) is None
    assert backtest.load_results(_write(tmp_path, None, raw="")) is None


def test_load_results_rejects_other_product_or_granularity(tmp_path):
    data = {"product_id": "BTC-USD", "granularity": 60, "x": 1}
    assert backtest.load_results(_write(tmp_path, data)) == data
    assert backtest.load_results(_write(tmp_path, {**data, "product_id": "ETH-USD"})) is None
    assert backtest.load_results(_write(tmp_path, {**data, "granularity": 300})) is None
    assert backtest.load_results(_write(tmp_path, {"x": 1})) is None        # legacy file with no identity


def test_load_results_hands_back_a_stored_error_result(tmp_path):
    data = {"error": "too few", "product_id": "BTC-USD", "granularity": 60}
    assert backtest.load_results(_write(tmp_path, data)) == data              # callers must check "error"


# ---------- get_confidence ----------

RESULTS = {
    "lookahead_candles": 5,
    "strength_buckets": {"bullish_3": {"hits": 30, "total": 40}, "bearish_2": {"hits": 3, "total": 10}},
    "divergence_buckets": {"bullish": {"hits": 6, "total": 10}, "bearish": {"hits": 0, "total": 0}},
}


def _snap(**votes):
    return signals.SignalSnapshot(price=1.0, individual=votes)


def test_confidence_unavailable_without_results():
    for r in (None, {}, {"error": "x"}):
        assert backtest.get_confidence(r, _snap(a="bullish"), {}, 15) == {"available": False}


def test_confidence_signal_rate_n_and_low_sample_flag():
    snap = _snap(a="bullish", b="bullish", c="bullish", d="bearish")
    out = backtest.get_confidence(RESULTS, snap, {"type": None}, 15)
    assert out["available"] and out["lookahead_candles"] == 5
    assert out["signal"] == {"direction": "bullish", "rate": 0.75, "n": 40, "low_sample": False}
    assert "divergence" not in out

    low = backtest.get_confidence(RESULTS, _snap(a="bearish", b="bearish"), {}, 15)
    assert low["signal"] == {"direction": "bearish", "rate": 0.3, "n": 10, "low_sample": True}
    # The flag flips exactly at min_sample.
    assert backtest.get_confidence(RESULTS, _snap(a="bearish", b="bearish"), {}, 10)["signal"]["low_sample"] is False


def test_confidence_neutral_or_unseen_state_has_no_signal_entry():
    assert "signal" not in backtest.get_confidence(RESULTS, _snap(a="bullish", b="bearish"), {}, 15)  # tie
    assert "signal" not in backtest.get_confidence(RESULTS, _snap(a="bullish"), {}, 15)               # bullish_1: never bucketed


def test_confidence_divergence_lookup():
    out = backtest.get_confidence(RESULTS, _snap(), {"type": "bullish"}, 15)
    assert out["divergence"] == {"direction": "bullish", "rate": 0.6, "n": 10, "low_sample": True}
    # An empty (total=0) bucket is not reported, rather than dividing by zero.
    assert "divergence" not in backtest.get_confidence(RESULTS, _snap(), {"type": "bearish"}, 15)
    assert "divergence" not in backtest.get_confidence(RESULTS, _snap(), None, 15)
