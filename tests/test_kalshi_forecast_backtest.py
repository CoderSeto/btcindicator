"""
Smoke test for review/kalshi_forecast_backtest.py: write a synthetic study cache to disk in the
formats kalshi_study reads (raw market records, candlesticks as JSON lines, one Coinbase CSV per
UTC day), run the real script over it, and check the report is complete -- limits included -- and
reaches the right verdict on a world whose answer is known. Nothing touches the network (a
Coinbase fetch would raise).
"""
import importlib.util
import json
import os
import pathlib
from datetime import datetime, timezone

import pandas as pd
import pytest

import kalshi_forecast_study as fc
import kalshi_study as ks
from test_kalshi_forecast_study import synthetic_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
JAN_1_2026 = 1_767_225_600                      # a UTC midnight, safely before "today"


def load_script():
    spec = importlib.util.spec_from_file_location("kalshi_forecast_backtest", ROOT / "review" / "kalshi_forecast_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_cache(cache, markets, candles, cb):
    os.makedirs(cache / "coinbase_1m", exist_ok=True)
    raw = [{"ticker": m["ticker"], "result": "yes" if m["outcome"] else "no", "floor_strike": m["strike"],
            "open_time": iso(m["open_ts"]), "close_time": iso(m["close_ts"]), "expiration_value": m["expiration_value"]}
           for m in markets]
    (cache / "kalshi_markets.json").write_text(json.dumps(raw), encoding="utf-8")
    with open(cache / "kalshi_candles.jsonl", "w", encoding="utf-8") as f:
        for tk, q in candles.items():
            sticks = [{"end_period_ts": ts, "yes_bid": {"close_dollars": f"{b:.4f}"},
                       "yes_ask": {"close_dollars": f"{a:.4f}"}, "price": {"close_dollars": f"{last:.4f}"}}
                      for ts, (b, a, last) in sorted(q.items())]
            f.write(json.dumps({"market_ticker": tk, "candlesticks": sticks}) + "\n")
    for day, g in cb.groupby(cb.ts // 86400):
        name = datetime.fromtimestamp(int(day) * 86400, tz=timezone.utc).strftime("%Y-%m-%d")
        g.to_csv(cache / "coinbase_1m" / f"{name}.csv", index=False)


def no_network(*a, **k):
    raise AssertionError("the backtest tried to fetch from Coinbase")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "fetch_coinbase_day", no_network)
    return tmp_path / "study_data"


def test_the_script_prints_a_complete_report_and_recalibrates_a_miscalibrated_market(cache, capsys):
    write_cache(cache, *synthetic_world(n_days=30, per_day=32, seed=2, market="shrunk", t0=JAN_1_2026))
    out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert "Family: 7 models, 54 skill intervals -> Bonferroni z = 3.3" in text
    # the limits are in the output itself (principle 2), including the two the task names
    assert "LIMITS, which apply to every number below" in text
    assert "The market mid is a STRONG baseline" in text and "Calibration alone is trivial" in text
    assert "not a\n    trading edge" in text or "not a trading edge" in text
    for section in ("Tuning (train only", "HELD-OUT scorecard", "HELD-OUT skill", "vs each baseline",
                    "a fitted combination vs each of its components", "HELD-OUT per horizon", "HELD-OUT reliability",
                    "Raw probabilities side by side", "SUMMARY", "Base rate"):
        assert section in text, section
    for name in fc.BASELINES + fc.MODELS:
        assert name in text
    assert "small" in text and "n_eff" in text                   # thin buckets are flagged, by effective sample size
    assert len(out["skills"]) == fc.FAMILY_TESTS
    s = out["skills"]
    assert s[(s.model == "platt_market") & (s.reference == "market") & (s.metric == "brier")].verdict.iloc[0] == "BETTER"
    assert "Better than the market mid, interval above zero" in text and "NO MODEL HERE BEATS THE MARKET MID" not in text
    assert "A better forecast is not a trading edge" in text
    # only held-out rows are scored, and the per-row probabilities are saved side by side
    saved = pd.read_csv(cache / "forecast_predictions.csv")
    assert len(saved) == len(out["predictions"]) and set(fc.BASELINES + fc.MODELS) <= set(saved.columns)
    _, test_cut = fc.split_cutoffs([{"close_ts": c} for c in out["dataset"].close_ts.unique()])
    assert saved.close_ts.min() >= test_cut


def test_a_market_that_is_already_right_gets_the_plain_null(cache, capsys):
    write_cache(cache, *synthetic_world(n_days=30, per_day=32, seed=4, market="true", t0=JAN_1_2026))
    out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    s = out["skills"]
    assert not (s[s.reference == "market"].verdict == "BETTER").any()
    assert "NO MODEL HERE BEATS THE MARKET MID" in text and "Better than the market mid" not in text


def test_a_forming_candle_is_dropped_before_anything_is_computed(cache):
    markets, candles, cb = synthetic_world(n_days=30, per_day=32, seed=2, t0=JAN_1_2026)
    write_cache(cache, markets, candles, cb)
    last = int(cb.ts.max())
    now = datetime.fromtimestamp(last + 20, tz=timezone.utc)     # 20 seconds into the last candle: still forming
    _, _, grid, frame = load_script().load_inputs(str(cache), now=now)
    assert int(frame.ts.max()) == last - 60 and grid.last_close_before(last + 60) is None
    _, _, _, whole = load_script().load_inputs(str(cache))       # well after it closed: kept
    assert int(whole.ts.max()) == last


def test_missing_cache_tells_you_what_to_run_first(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_script().main(["--cache-dir", str(tmp_path / "nothing")])
    assert "kalshi_market_study.py fetch" in str(e.value)


def test_the_new_files_read_as_utf8_and_hold_no_key_fee_or_order_code():
    for name in ("kalshi_forecast_study.py", "review/kalshi_forecast_backtest.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        low = text.lower()
        assert "api_key" not in low and "authorization" not in low and "/portfolio/orders" not in low
        assert "order_fee" not in text and "ev_buy" not in text  # a forecasting study: no cost or profit arithmetic
