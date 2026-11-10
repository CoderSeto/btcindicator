"""
Smoke test for review/kalshi_strategy_backtest.py: build a synthetic study cache on disk in the
exact formats kalshi_study writes and reads (raw Kalshi market records, raw candlesticks as JSON
lines, one Coinbase CSV per UTC day), run the real script over it, and check it prints a complete
report -- limits included -- and reaches the right verdict on a planted pattern and on a coin.
Nothing touches the network (a Coinbase fetch would raise; the Kalshi fetch is replaced).
"""
import importlib.util
import json
import os
import pathlib
from datetime import datetime, timezone

import pytest

import kalshi_strategy_study as st
import kalshi_study as ks
from test_kalshi_strategy_study import synthetic

ROOT = pathlib.Path(__file__).resolve().parent.parent
JAN_1_2026 = 1_767_225_600                      # a UTC midnight, safely before "today"


def load_script():
    spec = importlib.util.spec_from_file_location("kalshi_strategy_backtest", ROOT / "review" / "kalshi_strategy_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_cache(cache, markets, candles, cb):
    os.makedirs(cache / "coinbase_1m", exist_ok=True)
    raw = [{"ticker": m["ticker"], "result": "yes" if m["outcome"] else "no", "floor_strike": m["strike"],
            "open_time": iso(m["open_ts"]), "close_time": iso(m["close_ts"]), "expiration_value": "75000.00"}
           for m in markets]
    (cache / "kalshi_markets.json").write_text(json.dumps(raw), encoding="utf-8")
    with open(cache / "kalshi_candles.jsonl", "w", encoding="utf-8") as f:
        for tk, q in candles.items():
            sticks = [{"end_period_ts": ts, "yes_bid": {"close_dollars": f"{b:.4f}"},
                       "yes_ask": {"close_dollars": f"{a:.4f}"}, "price": {"close_dollars": f"{last:.4f}"}}
                      for ts, (b, a, last) in sorted(q.items())]
            f.write(json.dumps({"market_ticker": tk, "candlesticks": sticks}) + "\n")
    cb = cb.assign(day=(cb.ts // 86400))
    for day, g in cb.groupby("day"):
        name = datetime.fromtimestamp(int(day) * 86400, tz=timezone.utc).strftime("%Y-%m-%d")
        g.drop(columns="day").to_csv(cache / "coinbase_1m" / f"{name}.csv", index=False)


def no_network(*a, **k):
    raise AssertionError("the backtest tried to fetch from Coinbase")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "fetch_coinbase_day", no_network)
    return tmp_path / "study_data"


def test_the_script_prints_a_complete_report_and_finds_a_planted_streak(cache, capsys):
    write_cache(cache, *synthetic(persistence=0.9, seed=3, t0=JAN_1_2026))
    results, out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert "Family: 40 tests" in text and "Bonferroni z = 3.23" in text
    assert "LIMITS, which apply to every number below" in text                   # limits in the output (principle 2)
    assert "upper bound on a real order" in text and "descriptive" in text
    assert "Coverage" in text and "TRAIN:" in text and "HELD-OUT" in text and "SUMMARY" in text
    assert "win vs b/e" in text and "Direction information" in text
    assert len(results) == 2 * st.STRATEGY_TESTS
    assert set(out["selected"].rule) == {"streak_follow"} and "CONFIRMED" in text
    assert "NO RULE HERE HAS SHOWN AN EDGE" not in text
    assert "small" in text                                                       # thin cells are flagged, not hidden
    for rule in st.RULES:
        assert rule in text


def test_a_coin_reports_plainly_that_nothing_has_an_edge(cache, capsys):
    write_cache(cache, *synthetic(persistence=0.5, seed=8, t0=JAN_1_2026, bid=0.69, ask=0.71))
    results, out = load_script().main(["--cache-dir", str(cache)])
    text = capsys.readouterr().out
    assert out["selected"].empty
    assert "Selected on train (interval entirely above zero): none" in text
    assert "NO RULE HERE HAS SHOWN AN EDGE" in text and "CONFIRMED" not in text
    assert "not selected (no claim)" in text
    # a 0.70 mid has a favorite, so the vol rules trade here
    assert (results[results.rule == "vol_regime"].n > 0).all()


def test_missing_cache_tells_you_what_to_run_first(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_script().main(["--cache-dir", str(tmp_path / "nothing")])
    assert "kalshi_market_study.py fetch" in str(e.value)


def test_fetch_runs_the_edge_studys_fetch_against_the_host_given_for_this_run_only(monkeypatch):
    script = load_script()
    monkeypatch.setattr(ks.kalshi_api, "BASE_URL", ks.kalshi_api.BASE_URL)      # restored after the test
    seen = []

    class FakeStudy:
        @staticmethod
        def cmd_fetch():
            seen.append(ks.kalshi_api.BASE_URL)

    monkeypatch.setattr(script, "_review_script", lambda name: FakeStudy if name == "kalshi_market_study" else None)
    script.fetch("https://example.invalid/trade-api/v2/")
    assert seen == ["https://example.invalid/trade-api/v2"]


def test_the_new_files_read_as_utf8_and_hold_no_key_or_order_endpoint():
    for name in ("kalshi_strategy_study.py", "review/kalshi_strategy_backtest.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        low = text.lower()
        assert "api_key" not in low and "authorization" not in low and "/portfolio/orders" not in low
