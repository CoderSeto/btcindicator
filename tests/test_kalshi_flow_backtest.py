"""
Smoke test for review/kalshi_flow_backtest.py: write a synthetic study cache to disk in the formats
the studies read (raw market records, candlesticks as JSON lines, one Coinbase CSV and one Binance.US
trade file per UTC day), run the real script over it, and check the report is complete -- limits,
exclusion counts and the held-out statement included -- and reaches the right verdict on worlds whose
answer is known. Nothing touches the network: the Kalshi count and the Binance.US API are faked.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

import kalshi_flow_study as fl
import kalshi_study as ks
from test_kalshi_flow_study import FakeApi, leaky_trades
from test_kalshi_forecast_backtest import JAN_1_2026, no_network, write_cache
from test_kalshi_forecast_study import synthetic_world

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("kalshi_flow_backtest", ROOT / "review" / "kalshi_flow_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_trades(cache, trades, skip_days=()):
    day = trades["ts"] // fl.DAY_MS
    for d in np.unique(day):
        name = fl.day_of_ms(int(d) * fl.DAY_MS)
        if name in skip_days:
            continue
        keep = day == d
        fl.save_day(str(cache), name, {k: trades[k][keep] for k in ("id", "ts", "price", "qty", "buy")},
                    int(d) * fl.DAY_MS + fl.DAY_MS - 1)


def build_cache(cache, leak, seed, skip_days=(), meta=None):
    markets, candles, cb = synthetic_world(n_days=30, per_day=32, seed=seed, market="true", t0=JAN_1_2026)
    write_cache(cache, markets, candles, cb)
    write_trades(cache, leaky_trades(cb, leak, seed=seed + 100), skip_days)
    if meta is not None:
        fl.write_meta(str(cache), meta)
    return markets, cb


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "fetch_coinbase_day", no_network)
    return tmp_path / "study_data"


def test_the_script_prints_a_complete_report_and_finds_flow_that_knows_the_future(cache, capsys):
    build_cache(cache, leak=0.35, seed=2, meta={"new_settled_contracts": 8, "pristine_after": None})
    out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert "Family: 8 fitted models (4 with flow, each with its flow-free counterpart), 60 skill intervals -> Bonferroni z = 3.34" in text
    # the held-out statement: not pristine, and how many new contracts there were
    assert "NOT pristine" in text and "Only 8 contract(s) had settled after the cache's last close" in text
    # every limitation the task names is in the output (principle 2)
    for phrase in ("LIMITS, which apply to every number below", "ONE venue: Binance.US BTCUSDT only", "Bybit was\n    geo-blocked",
                   "quotes USDT, not USD", "No order-book data", "one-minute closes", "EXCLUDED and counted", "not a\n    trading edge",
                   "calibration alone is trivial"):
        assert phrase in text, phrase
    for section in ("aggTrades cached", "gaps:", "Rows (one per contract per horizon)", "Tuning (train only", "HELD-OUT scorecard",
                    "THE EFFECT OF FLOW", "vs each baseline", "DOES FLOW HELP?", "HELD-OUT per horizon", "HELD-OUT reliability",
                    "Fitted flow coefficients", "Raw probabilities side by side", "SUMMARY"):
        assert section in text, section
    for name in fl.BASELINES + fl.MODELS:
        assert name in text
    assert "n_eff" in text and "small" in text
    assert len(out["skills"]) == fl.FAMILY_TESTS
    assert out["verdicts"].set_index("model").loc["market_flow", "helps"]
    assert "FLOW HELPS: " in text and "DOES NOT IMPROVE THE FORECAST" not in text
    assert "A better forecast is not a trading edge" in text
    saved = pd.read_csv(cache / "flow_predictions.csv")
    assert len(saved) == len(out["predictions"]) and set(fl.BASELINES + fl.MODELS) <= set(saved.columns)
    assert (out["dataset"].loc[out["predictions"].index, "split"] == "test").all()


def test_flow_that_is_a_coin_gets_the_plain_null_and_a_missing_day_is_counted(cache, capsys):
    build_cache(cache, leak=0.0, seed=5, skip_days=("2026-01-10",))
    out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert not out["verdicts"].helps.any()
    assert "BINANCE.US TRADE FLOW DOES NOT IMPROVE THE FORECAST: no flow model passes the rule." in text
    assert "FLOW HELPS: " not in text
    assert "NO MODEL HERE BEATS THE MARKET MID" in text
    # no --fetch was run: the output says the pristine check was not made, rather than implying it was
    assert "Whether newer contracts exist was not checked" in text
    # the missing day is one uncovered span, and every row whose window touches it is excluded and counted
    d = out["dataset"]
    assert "1 uncovered span(s)" in text
    lost = d[(d.day == "2026-01-10") & d.usable]
    assert len(lost) > 50 and lost.hole.all() and not lost.use.any()
    assert int(d[d.usable].hole.sum()) >= len(lost)
    assert d[d.use].day.nunique() == d.day.nunique() - 1


def test_fetch_counts_new_contracts_and_caches_trades_resumably(cache, capsys, monkeypatch):
    markets, cb = build_cache(cache, leak=0.0, seed=5)
    import shutil
    shutil.rmtree(cache / fl.TRADE_DIR)                            # start with no trades cached
    script = load_script()
    start_ms, end_ms = script.trade_span_ms([{"open_ts": markets[0]["open_ts"], "close_ts": markets[-1]["close_ts"]}])
    t = leaky_trades(cb, 0.0, seed=1, per_minute=1)
    keep = (t["ts"] >= start_ms) & (t["ts"] <= end_ms + 3_600_000)
    raw = [{"a": int(a), "p": f"{p:.2f}", "q": f"{q:.5f}", "T": int(ts), "m": bool(not b)}
           for a, p, q, ts, b in zip(t["id"][keep], t["price"][keep], t["qty"][keep], t["ts"][keep], t["buy"][keep])]
    api = FakeApi(raw)
    monkeypatch.setattr(fl, "binance_get", lambda session, params: api(params))
    counted = []
    monkeypatch.setattr(fl, "count_new_settled", lambda session, after: counted.append(after) or 8)
    monkeypatch.setattr(ks.kalshi_api, "BASE_URL", ks.kalshi_api.BASE_URL)

    script.main(["--cache-dir", str(cache), "--fetch", "--kalshi-base-url", "https://example.invalid/v2/", "--workers", "1"])
    text = capsys.readouterr().out
    assert "Contracts settled after the cache's last close: 8 (a pristine held-out set needs 1000)" in text
    assert counted == [markets[-1]["close_ts"]] and ks.kalshi_api.BASE_URL == "https://example.invalid/v2"
    meta = fl.read_meta(str(cache))
    assert meta["new_settled_contracts"] == 8 and meta["pristine_after"] is None and meta["cache_last_close"] == markets[-1]["close_ts"]
    got = fl.load_trades(str(cache), start_ms, end_ms)
    want = [r for r in raw if start_ms - start_ms % fl.DAY_MS <= r["T"] <= end_ms]
    assert len(got["ts"]) == len(want) and got["holes"] == [] and got["missing_ids"] == 0
    assert list(got["buy"][:50]) == [not r["m"] for r in want[:50]]                   # isBuyerMaker -> the taker sold

    calls = len(api.calls)
    script.main(["--cache-dir", str(cache), "--fetch", "--workers", "1"])            # again: nothing to do
    assert len(api.calls) == calls and counted == [markets[-1]["close_ts"]]          # the count is not repeated either
    assert "fetched 0 day(s)" in capsys.readouterr().out


def test_no_trades_cached_tells_you_to_fetch(cache):
    markets, candles, cb = synthetic_world(n_days=30, per_day=32, seed=5, market="true", t0=JAN_1_2026)
    write_cache(cache, markets, candles, cb)
    with pytest.raises(SystemExit) as e:
        load_script().main(["--cache-dir", str(cache)])
    assert "--fetch" in str(e.value)


def test_the_new_files_read_as_utf8_and_hold_no_key_fee_or_order_code():
    for name in ("kalshi_flow_study.py", "review/kalshi_flow_backtest.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        low = text.lower()
        assert "api_key" not in low and "apikey" not in low and "authorization" not in low and "/portfolio/orders" not in low
        assert "order_fee" not in text and "ev_buy" not in text  # a forecasting study: no cost or profit arithmetic
        assert "bybit.com" not in low and "x-mbx-apikey" not in low
