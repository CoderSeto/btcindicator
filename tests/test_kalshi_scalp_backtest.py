"""
Smoke test for review/kalshi_scalp_backtest.py: build a synthetic study cache on disk in the exact
formats kalshi_study writes and reads (raw Kalshi market records, raw candlesticks as JSON lines,
one Coinbase CSV per UTC day), run the real script over it, and check it prints a complete report.
The script is meant to be run once, on real data, on the user's machine; this is what makes it
unlikely to crash there. Nothing touches the network (a Coinbase fetch would raise).
"""
import importlib.util
import json
import os
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import kalshi_study as ks
from test_kalshi_scalp_study import synthetic_markets_and_candles

ROOT = pathlib.Path(__file__).resolve().parent.parent
JAN_1_2026 = 1_767_225_600                      # a UTC midnight, safely before "today"


def load_script():
    spec = importlib.util.spec_from_file_location("kalshi_scalp_backtest", ROOT / "review" / "kalshi_scalp_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_cache(cache, markets, candles):
    os.makedirs(cache, exist_ok=True)
    raw = [{"ticker": m["ticker"], "result": "yes" if m["outcome"] else "no", "floor_strike": m["strike"],
            "open_time": iso(m["open_ts"]), "close_time": iso(m["close_ts"]), "expiration_value": "75000.00"}
           for m in markets]
    (cache / "kalshi_markets.json").write_text(json.dumps(raw))
    with open(cache / "kalshi_candles.jsonl", "w") as f:
        for tk, q in candles.items():
            sticks = [{"end_period_ts": ts, "yes_bid": {"close_dollars": f"{b:.4f}"},
                       "yes_ask": {"close_dollars": f"{a:.4f}"}, "price": {"close_dollars": f"{last:.4f}"}}
                      for ts, (b, a, last) in sorted(q.items())]
            f.write(json.dumps({"market_ticker": tk, "candlesticks": sticks}) + "\n")
    # Coinbase: one CSV per UTC day, from 4 hours before the first open through the last close's day
    os.makedirs(cache / "coinbase_1m", exist_ok=True)
    first, last = markets[0]["open_ts"] - 4 * 3600, markets[-1]["close_ts"]
    rng = np.random.default_rng(1)
    px = 75_000.0
    for day in pd.date_range(pd.to_datetime(first, unit="s").normalize(), pd.to_datetime(last, unit="s").normalize(), freq="D"):
        ts = int(day.timestamp()) + 60 * np.arange(1440)
        close = px * np.exp(np.cumsum(rng.normal(0, 3e-4, 1440)))
        px = float(close[-1])
        pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}) \
            .to_csv(cache / "coinbase_1m" / f"{day:%Y-%m-%d}.csv", index=False)


def no_network(*a, **k):
    raise AssertionError("the backtest tried to fetch from Coinbase")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "fetch_coinbase_day", no_network)
    return tmp_path / "study_data"


def test_the_script_prints_a_complete_report_on_a_cached_study(cache, capsys):
    markets, candles = synthetic_markets_and_candles(n_days=60, momentum=0.9, seed=11, t0=JAN_1_2026)
    write_cache(cache, markets, candles)
    results, out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert "Family: 48 tests" in text and "Bonferroni z = 3.2" in text
    assert "Coverage" in text and "Cost floor and oracle ceiling" in text and "ORACLE" in text
    assert "TRAIN:" in text and "Selected on train" in text and "HELD-OUT" in text
    assert "Fills are assumed at the quoted top of book, in AND out" in text      # the limits are in the output (principle 2)
    assert len(results) == 96
    assert not out["selected"].empty and "CONFIRMED" in text                      # the planted pattern is found
    assert "contract_momentum" in text


def test_a_random_walk_reports_that_nothing_was_selected(cache, capsys):
    markets, candles = synthetic_markets_and_candles(n_days=30, momentum=0.0, seed=4, t0=JAN_1_2026)
    write_cache(cache, markets, candles)
    results, out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert out["selected"].empty
    assert "Selected on train (interval entirely above zero): none" in text and "CONFIRMED" not in text


def test_missing_cache_tells_you_what_to_run_first(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_script().main(["--cache-dir", str(tmp_path / "nothing")])
    assert "kalshi_market_study.py fetch" in str(e.value)
