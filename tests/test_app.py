"""
Tests for the Streamlit dashboard (app.py), driven by Streamlit's own AppTest
harness: the real script runs top to bottom, with only the outside world
stubbed -- Coinbase REST, the WebSocket feed handles, Kalshi's API, and the
log/result paths (redirected to a temp dir). Indicators, signals, outlook,
Kalshi math, and the order-book aggregator all run for real; the live-feed
scenario drives a genuine BookAggregator (on a fake clock) and TradeAggregator,
so the snapshot dicts the page reads have the shapes the feeds really produce.

app.py is a script, not a module, so its two pure helpers (_esc and
_strike_path_liquidity) are lifted out of its source with `ast` and unit
tested directly.
"""
import ast
import json
import pathlib
import re
import time as real_time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import backtest
import config as cfg
import data_fetch
import divergence
import exchange_feeds
import kalshi
import kalshi_api
import orderbook
import orderbook_feeds as obf
from test_dashboard_data import REST_BOOK, candles

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
OPEN_MARKET = json.loads((ROOT / "tests" / "fixtures" / "kalshi_open_markets.json").read_text())["markets"][0]
CLOSE = datetime(2026, 11, 7, 18, 15, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 7, 18, 7, 30, tzinfo=timezone.utc)              # 7.5 minutes before CLOSE


# ---------- helpers ----------

def kalshi_live(**over):
    return {**kalshi_api.parse_market({**OPEN_MARKET, **over}, now=NOW)}


class Env:
    """Everything a scenario can vary, plus counters for what the page asked for."""

    def __init__(self):
        self.df = candles()
        self.kalshi = kalshi_live()
        self.candle_requests = []
        self.kalshi_calls = 0
        self.candle_error = None
        self.feed_snapshot = None
        self.book_snapshot = None


def make_book_snapshot(monkeypatch):
    """A real BookAggregator on a fake clock: 60 one-second samples of a bid-heavy two-venue book
    with one large ask wall at 75,730 (between spot 75,700 and the 75,743.66 Kalshi strike)."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(obf, "time", SimpleNamespace(time=lambda: clock["t"], sleep=real_time.sleep))
    agg = obf.BookAggregator(cfg)
    for venue, shift in (("coinbase", 0.0), ("binance_us", 2.0)):
        bids = [[75_699.99 + shift - i, 2.0] for i in range(60)]
        asks = [[75_700.01 + shift + i, 0.5] for i in range(60)]
        asks[30][1] = 10.0                                               # 10 BTC ~ $757k at ~75,730
        agg.apply_snapshot(venue, bids, asks)
        agg.set_status(venue, "connected")
    for _ in range(60):
        clock["t"] += 1.0
        assert agg.sample() is not None
    snap = agg.snapshot()
    assert snap["available"] and any(w["held"] for w in snap["walls"])
    return snap


def make_feed_snapshot():
    agg = exchange_feeds.TradeAggregator(60, 100_000, 50, 50)
    now = real_time.time()
    agg.record_trade("coinbase", "buy", 75_700.0, 2.0, ts=now)          # $151k -> a whale
    agg.record_trade("binance_us", "buy", 75_700.0, 0.5, ts=now)
    agg.record_trade("bybit", "sell", 75_700.0, 0.2, ts=now)
    agg.record_liquidation("bybit", "long", 75_600.0, 3.0, ts=now)
    for ex in ("coinbase", "binance_us", "bybit"):
        agg.set_status(ex, "connected")
    return agg.snapshot()


@pytest.fixture
def env(monkeypatch, tmp_path):
    e = Env()
    # app.py writes these module-level values; monkeypatch puts them back afterwards.
    for name in ("PRODUCT_ID", "POLL_INTERVAL", "COMPOSITE_ALERT_THRESHOLD"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))
    monkeypatch.setattr(cfg, "ENABLE_EXCHANGE_FEEDS", False)
    monkeypatch.setattr(cfg, "ENABLE_BOOK_FEEDS", False)
    monkeypatch.setattr(cfg, "LOG_ALL_SIGNALS", False)
    monkeypatch.setattr(cfg, "LOG_KALSHI_PREDICTIONS", False)
    monkeypatch.setattr(cfg, "ALERT_LOG_PATH", str(tmp_path / "alerts.csv"))
    monkeypatch.setattr(cfg, "SIGNAL_LOG_PATH", str(tmp_path / "signals.csv"))
    monkeypatch.setattr(cfg, "KALSHI_PREDICTIONS_LOG_PATH", str(tmp_path / "kpred.csv"))
    monkeypatch.setattr(cfg, "BACKTEST_RESULTS_PATH", str(tmp_path / "bt.json"))
    e.tmp = tmp_path

    def get_candles(product_id, *a, **k):
        e.candle_requests.append(product_id)
        if e.candle_error:
            raise RuntimeError(e.candle_error)
        return e.df.copy()
    monkeypatch.setattr(data_fetch, "get_candles", get_candles)
    monkeypatch.setattr(data_fetch, "get_order_book", lambda *a, **k: REST_BOOK)

    def current():
        e.kalshi_calls += 1
        return e.kalshi
    monkeypatch.setattr(kalshi_api, "current_btc15m_market", current)

    class Handle:
        def __init__(self, which):
            self.which = which

        def snapshot(self):
            return e.feed_snapshot if self.which == "feed" else e.book_snapshot
    monkeypatch.setattr(exchange_feeds, "start_feeds", lambda c: Handle("feed"))
    monkeypatch.setattr(obf, "start_book_feeds", lambda c: Handle("book"))
    e.make_book_snapshot = lambda: make_book_snapshot(monkeypatch)
    st.cache_resource.clear()
    return e


def run(env=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def go_live(env):
    env.feed_snapshot = make_feed_snapshot()
    env.book_snapshot = env.make_book_snapshot()
    cfg.ENABLE_EXCHANGE_FEEDS = True
    cfg.ENABLE_BOOK_FEEDS = True


def metric(at, label):
    found = [m for m in at.metric if m.label == label]
    assert found, f"no metric labelled {label!r}; have {[m.label for m in at.metric]}"
    return found[0]


def all_text(at):
    out = []
    for kind in ("markdown", "caption", "warning", "error", "info", "success"):
        out += [e.value for e in getattr(at, kind)]
    out += [m.label + " " + str(m.value) + " " + str(m.delta) for m in at.metric]
    return "\n".join(out)


def no_exception(at):
    assert not at.exception, [e.value for e in at.exception]


# ---------- the pure helpers, lifted from app.py ----------

def _load_helpers():
    tree = ast.parse((ROOT / "app.py").read_text())
    wanted = {"_esc", "_strike_path_liquidity"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {f.name for f in funcs} == wanted
    ns = {"orderbook": orderbook}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "app.py", "exec"), ns)
    return ns["_esc"], ns["_strike_path_liquidity"]


esc, strike_path = _load_helpers()


def test_esc_escapes_every_dollar_sign():
    assert esc("$100,008.85 vs strike $100,000.00") == r"\$100,008.85 vs strike \$100,000.00"
    assert esc("no money here") == "no money here" and esc("") == ""


def book_for_path(extra_walls=()):
    asks = [(75_701.0, 1.0), (75_720.0, 2.0), (75_730.0, 10.0), (75_743.0, 1.0), (75_800.0, 5.0)]
    bids = [(75_699.0, 1.0), (75_680.0, 2.0), (75_650.0, 4.0), (75_500.0, 9.0)]
    return {"bids": bids, "asks": asks}


def path_args(spot=75_700.0, strike=75_743.66, venues=None, walls=(), available=True):
    snap = {"available": True, "books": venues if venues is not None else {"coinbase": book_for_path()},
            "walls": list(walls)}
    kal = {"available": available, "current_price": spot, "strike": strike}
    return snap, kal, cfg


def test_strike_path_above_spot_counts_resting_asks_between():
    out = strike_path(*path_args())
    want = sum(p * s for p, s in book_for_path()["asks"] if 75_700.0 <= p <= 75_743.66)   # brute force
    assert out["notional"] == pytest.approx(want) and want > 0
    assert (out["side"], out["venue"], out["settles"], out["direction_word"]) == ("ask", "coinbase", "YES", "up through")
    assert (out["low"], out["high"]) == (75_700.0, 75_743.66)


def test_strike_path_below_spot_counts_resting_bids():
    out = strike_path(*path_args(strike=75_660.0))
    want = sum(p * s for p, s in book_for_path()["bids"] if 75_660.0 <= p <= 75_700.0)
    assert out["notional"] == pytest.approx(want)
    assert (out["side"], out["settles"], out["direction_word"]) == ("bid", "NO", "down through")


def test_strike_path_prefers_coinbase_then_falls_back_to_the_first_venue():
    other = {"bids": [(75_699.0, 1.0)], "asks": [(75_720.0, 1.0)]}
    assert strike_path(*path_args(venues={"bybit": other, "coinbase": book_for_path()}))["venue"] == "coinbase"
    assert strike_path(*path_args(venues={"bybit": other, "binance_us": other}))["venue"] == "bybit"


def test_strike_path_beyond_the_retained_band_is_unanswerable():
    far = 75_700.0 * (1 + cfg.BOOK_RETAIN_BPS / 10_000.0 * 1.01)
    assert strike_path(*path_args(strike=far)) is None
    near_edge = 75_700.0 * (1 + cfg.BOOK_RETAIN_BPS / 10_000.0 * 0.99)
    assert strike_path(*path_args(strike=near_edge)) is not None


@pytest.mark.parametrize("snap", [None, {}, {"available": False, "books": {"coinbase": {}}}, {"available": True, "books": {}},
                                  {"available": True, "books": {"coinbase": {}}}])
def test_strike_path_none_without_a_usable_book(snap):
    assert strike_path(snap, {"available": True, "current_price": 75_700.0, "strike": 75_743.0}, cfg) is None


def test_strike_path_none_when_kalshi_unavailable():
    assert strike_path(*path_args(available=False)) is None


def test_strike_path_lists_only_held_walls_on_the_same_venue_side_and_range():
    w = lambda **kw: {"held": True, "venue": "coinbase", "side": "ask", "price": 75_730.0, "notional": 757_000, **kw}
    walls = [w(), w(held=False), w(venue="bybit"), w(side="bid"), w(price=75_900.0), w(price=75_699.0)]
    assert strike_path(*path_args(walls=walls))["walls"] == [walls[0]]


# ---------- the page, no live feeds ----------

def test_renders_without_feeds_or_kalshi(env):
    env.kalshi = {"available": False, "reason": "Couldn't reach Kalshi's API: offline"}
    at = run()
    no_exception(at)
    assert not at.error
    price = metric(at, "BTC-USD")
    assert price.value == f"${env.df['close'].iloc[-1]:,.2f}"
    comp = metric(at, "Composite")
    assert re.fullmatch(r"[^\w]* (BULLISH|BEARISH|NEUTRAL)", comp.value)
    assert re.fullmatch(r"\d+/\d+ aligned", comp.delta)
    assert metric(at, "Order book (1 venue)").delta in ("buy pressure", "sell pressure", "balanced")   # REST fallback
    assert "Auto-track couldn't reach Kalshi" in all_text(at)
    assert len(at.tabs) == 5
    assert "Level-2 depth feeds are disabled" in all_text(at)
    assert "Multi-exchange feeds are disabled" in all_text(at)


def test_candle_fetch_failure_shows_an_error_and_stops(env):
    env.candle_error = "coinbase is down"
    at = run()
    no_exception(at)
    assert [e.value for e in at.error] == ["Couldn't fetch live data: coinbase is down"]
    assert not at.metric                                         # st.stop(): nothing below the error rendered
    assert not at.tabs


def test_too_little_history_shows_the_not_enough_data_message(env):
    env.df = env.df.head(10)
    at = run()
    assert "Not enough candle history" in at.error[0].value and not at.metric


def test_badges_show_only_signals_that_exist_and_warm_up_note_names_the_rest(env):
    at = run()
    badges = " ".join(m.value for m in at.markdown if "text-align:center" in m.value)
    for label in ("SMA", "EMA", "RSI", "MACD", "BOLL", "VOL", "BOOK"):
        assert label in badges
    assert "DELTA" not in badges and "FLOW" not in badges      # absent, not neutral
    warm = [c.value for c in at.caption if c.value.startswith("Warming up:")]
    assert warm and "Agg. Exchange Delta" in warm[0] and "Book Flow" in warm[0] and "Order Book" not in warm[0]


def test_outlook_box_has_headline_and_disclaimer(env):
    at = run()
    text = all_text(at)
    assert re.search(r"(Most indicators agree|A majority|A slim majority|Indicators are split)", text)
    assert "not a prediction and not financial advice" in text


# ---------- kalshi block ----------

def test_kalshi_auto_track_fills_strike_and_countdown_and_shows_model_vs_market(env):
    at = run()
    no_exception(at)
    assert at.number_input(key="kalshi_strike").value == 75_743.66
    assert at.number_input(key="kalshi_minutes").value == 7.5
    assert at.session_state["kalshi_last_ticker"] == OPEN_MARKET["ticker"]

    want = kalshi.fair_value(env.df, 75_743.66, 7.5, cfg.KALSHI_VOL_LOOKBACK_MINUTES)
    assert metric(at, "P(YES)").value == f"{want['p_yes'] * 100:.1f}%"
    assert metric(at, "P(NO)").value == f"{want['p_no'] * 100:.1f}%"
    assert metric(at, "Kalshi P(YES)").value == f"{want['p_yes'] * 100:.1f}%"      # header strip agrees
    assert metric(at, "Kalshi P(YES)").delta.endswith("pts vs market")
    assert re.fullmatch(r"\d+¢", metric(at, "Market YES").value)
    assert OPEN_MARKET["ticker"] in all_text(at)


def test_kalshi_p_yes_is_not_blended_with_the_bot_signals(env):
    # The alert threshold changes how the page reacts to the signals; the model's P(YES) must not move.
    at = run()
    slider(at, "Alert threshold").set_value(9).run()
    quiet = metric(at, "P(YES)").value
    slider(at, "Alert threshold").set_value(1).run()
    assert metric(at, "P(YES)").value == quiet
    assert metric(at, "Kalshi P(YES)").value == quiet


def test_manual_strike_survives_refreshes_within_the_same_window(env):
    at = run()
    at.number_input(key="kalshi_strike").set_value(75_000.0).run()
    assert at.number_input(key="kalshi_strike").value == 75_000.0
    at.run()
    assert at.number_input(key="kalshi_strike").value == 75_000.0           # not fought every refresh
    assert metric(at, "P(YES)").value != ""
    assert float(metric(at, "P(YES)").value.rstrip("%")) > 90               # spot ~700 above that strike


def test_new_window_resets_the_strike(env):
    at = run()
    at.number_input(key="kalshi_strike").set_value(75_000.0).run()
    env.kalshi = kalshi_live(ticker="KXBTC15M-26NOV071330-30", floor_strike=75_800.0)
    at.run()
    assert at.number_input(key="kalshi_strike").value == 75_800.0
    assert at.session_state["kalshi_last_ticker"] == "KXBTC15M-26NOV071330-30"


def test_unchecking_auto_track_stops_polling_kalshi_and_keeps_manual_values(env):
    at = run()
    calls_after_first = env.kalshi_calls
    at.checkbox(key="kalshi_auto_track").uncheck().run()
    at.number_input(key="kalshi_strike").set_value(75_600.0).run()
    at.run()
    assert env.kalshi_calls == calls_after_first                           # no further requests
    assert at.number_input(key="kalshi_strike").value == 75_600.0
    assert metric(at, "Ann. vol").value.endswith("%")                      # no market quote -> vol shown instead
    assert metric(at, "Kalshi P(YES)").delta.endswith("min left")


def test_kalshi_unavailable_still_gives_a_model_value_from_manual_inputs(env):
    env.kalshi = {"available": False, "reason": "No open KXBTC15M market found right now."}
    at = run()
    no_exception(at)
    assert at.session_state["kalshi_strike"] == float(round(env.df["close"].iloc[-1]))   # defaults to spot
    assert 0 < float(metric(at, "P(YES)").value.rstrip("%")) < 100
    assert "No open KXBTC15M market" in all_text(at)


def test_zero_minutes_left_resolves_on_spot_vs_strike(env):
    env.kalshi = {"available": False, "reason": "x"}
    at = run()
    at.number_input(key="kalshi_strike").set_value(75_000.0)
    at.number_input(key="kalshi_minutes").set_value(0.0).run()
    assert metric(at, "P(YES)").value == "100.0%"


def test_predictions_are_logged_only_when_enabled_and_tracking_a_real_ticker(env):
    path = pathlib.Path(cfg.KALSHI_PREDICTIONS_LOG_PATH)
    run()
    assert not path.exists()                                               # off by default

    cfg.LOG_KALSHI_PREDICTIONS = True
    at = run()
    df = pd.read_csv(path)
    assert list(df["ticker"]) == [OPEN_MARKET["ticker"]]
    assert df.loc[0, "strike"] == 75_743.66
    assert df.loc[0, "p_yes"] == pytest.approx(float(metric(at, "P(YES)").value.rstrip("%")) / 100, abs=5e-4)
    assert df.loc[0, "kalshi_yes_bid"] == pytest.approx(float(OPEN_MARKET["yes_bid_dollars"]))

    path.unlink()
    at.checkbox(key="kalshi_auto_track").uncheck().run()
    assert not path.exists()                                               # no ticker, nothing to log


# ---------- sidebar / alerts / logging ----------

def slider(at, label_start):
    return next(s for s in at.sidebar.slider if s.label.startswith(label_start))


def test_alert_threshold_controls_alerting_and_the_alert_log(env):
    cfg.COMPOSITE_ALERT_THRESHOLD = 9                                      # start with alerting effectively off
    at = run()
    log = pathlib.Path(cfg.ALERT_LOG_PATH)
    assert not log.exists()
    assert "No alerts logged yet" in all_text(at)
    assert "aligned " not in " ".join(m.value for m in at.markdown if "\U0001F6A8" in m.value)

    slider(at, "Alert threshold").set_value(1).run()
    no_exception(at)
    assert log.exists()
    rows = pd.read_csv(log)
    assert len(rows) >= 1 and set(rows["composite"]) <= {"bullish", "bearish"}
    assert any("\U0001F6A8" in m.value and "logged to" in m.value for m in at.markdown)
    shown = [d.value for d in at.dataframe if "composite" in d.value.columns]
    assert shown and len(shown[0]) == len(rows)


def test_corrupt_alert_log_is_reported_not_crashed_on(env):
    pathlib.Path(cfg.ALERT_LOG_PATH).write_text("a,b\n1,2\n3,4,5,6\n")   # a later row wider than the earlier ones
    cfg.COMPOSITE_ALERT_THRESHOLD = 9                                      # no alert -> the bad file is left to be read
    at = run()
    no_exception(at)
    assert any("looks corrupted" in e.value for e in at.error)


def test_signal_log_gets_a_fixed_schema_row_each_run(env):
    import main
    cfg.LOG_ALL_SIGNALS = True
    run()
    rows = pd.read_csv(cfg.SIGNAL_LOG_PATH)
    assert list(rows.columns) == main.SIGNAL_LOG_HEADER
    assert len(rows) == 1 and pd.isna(rows.loc[0, "agg_delta"])           # blank, not 0 / "neutral"


def test_changing_the_pair_refetches_for_that_pair(env):
    at = run()
    assert set(env.candle_requests) == {"BTC-USD"}
    at.sidebar.selectbox[0].select("ETH-USD").run()
    no_exception(at)
    assert env.candle_requests[-1] == "ETH-USD"
    assert any("ETH-USD" in m.value for m in at.markdown)
    assert cfg.PRODUCT_ID == "BTC-USD"                                     # the shared module is untouched


def test_refresh_interval_applies_to_this_session_only(env):
    at = run()
    slider(at, "Refresh interval").set_value(45).run()
    assert any("refresh 45s" in m.value for m in at.markdown)
    assert cfg.POLL_INTERVAL == 30                                         # config.py default, unchanged


def test_one_session_does_not_change_another_sessions_defaults(env):
    # app.py used to write the sidebar values into the shared `config` module, so a new tab opened
    # on the previous tab's pair and threshold -- and two live tabs could render or log one pair's
    # data under the other's label mid-run.
    a = run()
    a.sidebar.selectbox[0].select("ETH-USD").run()
    slider(a, "Alert threshold").set_value(7).run()
    b = run()
    no_exception(b)
    assert b.sidebar.selectbox[0].value == "BTC-USD"
    assert slider(b, "Alert threshold").value == cfg.COMPOSITE_ALERT_THRESHOLD
    assert a.sidebar.selectbox[0].value == "ETH-USD"                      # A keeps its own choice


def test_reruns_from_clicks_do_not_duplicate_log_rows(env):
    # Every rerun -- a click, a refresh tick, another tab -- used to append to both logs.
    cfg.LOG_ALL_SIGNALS = True
    cfg.COMPOSITE_ALERT_THRESHOLD = 1
    at = run()
    at.sidebar.button[0].click().run()                                     # "Refresh now"
    at.checkbox(key="kalshi_auto_track").uncheck().run()
    assert len(pd.read_csv(cfg.SIGNAL_LOG_PATH)) == 1
    assert len(pd.read_csv(cfg.ALERT_LOG_PATH)) == 1


# ---------- backtest sidebar ----------

def sidebar_text(at):
    return " ".join(c.value for c in at.sidebar.caption)


def test_backtest_sidebar_states(env):
    assert "No backtest yet for BTC-USD" in sidebar_text(run())

    path = pathlib.Path(cfg.BACKTEST_RESULTS_PATH)
    path.write_text(json.dumps({"product_id": "BTC-USD", "granularity": 60, "candles_used": 3999,
                                "generated_at": "2026-11-07T12:34:56+00:00", "indicator_buckets": {}}))
    assert "Last backtest: 3999 candles of BTC-USD, generated 2026-11-07 12:34 UTC" in sidebar_text(run())

    path.write_text(json.dumps({"product_id": "BTC-USD", "granularity": 60, "error": "Only got 10 candles"}))
    assert "Last backtest attempt failed: Only got 10 candles" in sidebar_text(run())

    path.write_text(json.dumps({"product_id": "ETH-USD", "granularity": 60, "candles_used": 5}))
    assert "No backtest yet for BTC-USD" in sidebar_text(run())            # other pair's results ignored


def test_run_backtest_button_runs_it_and_reports_failures(env, monkeypatch):
    calls = []
    monkeypatch.setattr(backtest, "run_backtest", lambda c: calls.append(1) or {})
    at = run()
    next(b for b in at.sidebar.button if b.label == "Run backtest now").click().run()
    no_exception(at)
    assert calls == [1] and not at.sidebar.error

    def boom(c):
        raise RuntimeError("history unavailable")
    monkeypatch.setattr(backtest, "run_backtest", boom)
    at = run()
    next(b for b in at.sidebar.button if b.label == "Run backtest now").click().run()
    assert [e.value for e in at.sidebar.error] == ["Backtest failed: history unavailable"]


def test_backtest_results_show_up_as_weights_and_confidence(env):
    import dashboard_data
    base = dashboard_data.load_state(cfg)
    votes = {k: v for k, v in base.snapshot.individual.items()
             if k in ("sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume") and v != "neutral"}
    assert votes
    pathlib.Path(cfg.BACKTEST_RESULTS_PATH).write_text(json.dumps({
        "product_id": "BTC-USD", "granularity": 60, "candles_used": 4000, "generated_at": "2026-11-07T00:00:00+00:00",
        "lookahead_candles": 5, "strength_buckets": {}, "divergence_buckets": {},
        "indicator_buckets": {f"{n}_{v}": {"hits": 80, "total": 100} for n, v in votes.items()},
    }))
    at = run()
    no_exception(at)
    weights = [d.value for d in at.dataframe if "Weight" in d.value.columns]
    assert weights
    table = weights[0].set_index("Indicator")["Weight"]
    assert any(float(w) > 1.0 for w in table)                              # an 80% hit rate is up-weighted
    assert "Weighted by each indicator's historical accuracy" in all_text(at)


def test_weights_table_without_backtest_data_shows_unit_weights(env):
    at = run()
    weights = [d.value for d in at.dataframe if "Weight" in d.value.columns]
    assert weights and set(weights[0]["Weight"]) == {"1.00"}


# ---------- dollar amounts must not turn into LaTeX ----------

def test_divergence_text_with_dollar_amounts_is_escaped(env, monkeypatch):
    detail = "Price made a higher high ($75,500 to $75,700) while RSI made a lower high (70 to 60)."
    for typ in ("bearish", "bullish"):
        monkeypatch.setattr(divergence, "detect_divergence", lambda *a, t=typ, **k: {"type": t, "detail": detail})
        at = run()
        no_exception(at)
        texts = [w.value for w in at.warning] + [c.value for c in at.caption]
        with_dollars = [t for t in texts if "75,500" in t]
        assert with_dollars, "divergence detail never reached the page"
        for t in with_dollars:
            assert not re.search(r"(?<!\\)\$", t), f"unescaped $ would render as LaTeX: {t!r}"


def test_every_money_string_on_the_page_is_escaped(env):
    go_live(env)
    at = run()
    no_exception(at)
    offenders = []
    for kind in ("caption", "warning"):
        for e in getattr(at, kind):
            if re.search(r"(?<!\\)\$", e.value):
                offenders.append((kind, e.value))
    assert not offenders, offenders


# ---------- the page with live feeds ----------

def test_live_book_feeds_drive_the_order_book_header_and_tab(env):
    go_live(env)
    at = run()
    no_exception(at)
    head = metric(at, "Order book")
    assert re.fullmatch(r"\+\d\.\d\d", head.value)                         # bid-heavy book -> positive imbalance
    assert re.fullmatch(r"held \d+s · 2 venues", head.delta)
    assert metric(at, f"Imbalance ({cfg.BOOK_BAND_BPS:.0f}bps)").value.startswith("+")
    assert metric(at, "Venue agreement").value.endswith("%")
    assert metric(at, "Held for").delta == "bullish"
    assert "Order book (1 venue)" not in [m.label for m in at.metric]      # no REST fallback when depth is live
    status = " ".join(m.value for m in at.markdown if "connected" in m.value)
    assert "coinbase" in status and "binance_us" in status and "\U0001F7E2" in status


def test_live_book_venue_and_wall_tables(env):
    go_live(env)
    at = run()
    venues = next(d.value for d in at.dataframe if "Microprice" in d.value.columns)
    assert set(venues["Venue"]) == {"coinbase", "binance_us"}
    assert venues["Imbalance"].str.startswith("+").all()
    walls = next(d.value for d in at.dataframe if "vs typical level" in d.value.columns)
    assert (walls["Side"] == "ask (above)").any()
    assert walls["Notional"].str.startswith("$").all()
    assert "Resting walls that have held" in all_text(at).replace("**", "")


def test_live_book_wall_appears_in_the_outlook_note(env):
    go_live(env)
    text = all_text(run())
    assert "Order book: \\$" in text and "above mid" in text and "not a floor or a ceiling" in text


def test_strike_path_liquidity_is_context_next_to_an_unchanged_p_yes(env):
    go_live(env)
    with_book = run()
    caps = " ".join(c.value for c in with_book.caption)
    assert "resting asks between spot and strike on coinbase" in caps
    assert "not** folded into P(YES)" in caps
    assert "Resting liquidity between spot and the strike" in all_text(with_book)
    assert "does not adjust P(YES)" in caps

    # The same page without depth feeds produces the same model number: the readout never feeds P(YES).
    env.book_snapshot, cfg.ENABLE_BOOK_FEEDS = None, False
    st.cache_resource.clear()
    without_book = run()
    assert metric(without_book, "P(YES)").value == metric(with_book, "P(YES)").value
    assert "between spot and strike" not in " ".join(c.value for c in without_book.caption)


def test_trade_feed_tab_shows_volume_whales_and_liquidations(env):
    go_live(env)
    at = run()
    assert metric(at, "Buy volume (60s)").value == "$189,250"            # 2.0 x 75,700 + 0.5 x 75,700
    assert metric(at, "Sell volume (60s)").value == "$15,140"            # 0.2 x 75,700
    assert metric(at, "Aggregated delta").value.startswith("+")
    whales = next(d.value for d in at.dataframe if {"time", "exchange", "side", "notional", "price"} <= set(d.value.columns))
    assert list(whales["exchange"]) == ["coinbase"] and whales["notional"].iloc[0] == "$151,400"
    liqs = next(d.value for d in at.dataframe if "liquidated_side" in d.value.columns)
    assert list(liqs["liquidated_side"]) == ["long"]
    assert any(m.value.count("DELTA") for m in at.markdown)               # the delta badge now exists


def test_depth_feeds_still_connecting_shows_status_not_numbers(env):
    env.book_snapshot = {"available": False, "reason": "no venue order book has connected yet",
                         "status": {"coinbase": "connecting", "bybit": "reconnecting"}, "samples": 0}
    cfg.ENABLE_BOOK_FEEDS = True
    at = run()
    no_exception(at)
    assert "Waiting on depth feeds -- no venue order book has connected yet" in all_text(at)
    assert metric(at, "Order book (1 venue)")                              # REST fallback still drives the header
    status = " ".join(m.value for m in at.markdown if "title='" in m.value)
    assert "\U0001F7E1" in status                                          # connecting/reconnecting are amber, not red


def test_feeds_enabled_but_no_handle_says_so(env):
    cfg.ENABLE_BOOK_FEEDS = True
    cfg.ENABLE_EXCHANGE_FEEDS = True
    env.book_snapshot = None
    env.feed_snapshot = None
    text = all_text(run())
    assert "Depth feeds not available for this pair." in text
    assert "Feeds not available for this pair." in text


def test_venue_status_dot_colours(env):
    env.book_snapshot = {"available": False, "reason": "x", "samples": 0,
                         "status": {"a": "connected", "b": "syncing book", "c": "error: boom"}}
    cfg.ENABLE_BOOK_FEEDS = True
    at = run()
    dots = {}
    for m in at.markdown:
        hit = re.search(r"title='(\w): ([^']*)'>(\S+) <b>", m.value)
        if hit:
            dots[hit.group(1)] = hit.group(3)
    assert dots == {"a": "\U0001F7E2", "b": "\U0001F7E1", "c": "\U0001F534"}      # green / amber / red


# ---------- chart tab ----------

def test_chart_tab_renders_without_error(env):
    at = run()
    no_exception(at)
    charts = at.get("plotly_chart")
    assert len(charts) == 1


# ---------- review regressions (see REVIEW_FINDINGS.md) ----------

def test_manually_edited_strike_is_never_logged_under_the_live_ticker(env):
    # The calibration report scores each logged row against the *ticker's* real outcome, which
    # settles on the market's own strike. A row computed at a hand-typed strike but tagged with
    # that ticker scores the model on a question nobody asked it.
    cfg.LOG_KALSHI_PREDICTIONS = True
    at = run()
    at.number_input(key="kalshi_strike").set_value(75_000.0).run()
    at.run()
    df = pd.read_csv(cfg.KALSHI_PREDICTIONS_LOG_PATH)
    assert list(df["strike"]) == [75_743.66]                               # only the auto-tracked row


def test_kalshi_auto_track_only_follows_the_btc_contract_on_btc_usd(env):
    # KXBTC15M is a BTC/USD contract. On ETH-USD the page would otherwise price the BTC strike
    # (~$75.7k) against ETH candles and log a ~0% P(YES) under the BTC ticker.
    cfg.LOG_KALSHI_PREDICTIONS = True
    at = run()
    path = pathlib.Path(cfg.KALSHI_PREDICTIONS_LOG_PATH)
    path.unlink()
    calls = env.kalshi_calls
    eth = env.df.copy()
    eth[["open", "high", "low", "close"]] /= 25.0                          # ~$3k, an ETH-like price
    env.df = eth
    at.sidebar.selectbox[0].select("ETH-USD").run()
    no_exception(at)
    assert env.kalshi_calls == calls                                       # the BTC market isn't fetched
    assert not path.exists()                                               # nor logged against ETH prices
    assert "only tracks KXBTC15M on BTC-USD" in all_text(at)


def test_settlement_averaging_caveat_states_the_right_direction(env):
    # Averaging lowers the terminal variance, which makes the true probability MORE extreme than
    # the point-in-time model (see test_kalshi.py::test_settlement_averaging_makes_the_true_
    # probability_more_extreme). The Notes tab previously said the opposite.
    text = all_text(run())
    assert "overstates how extreme" not in text
    assert "less extreme than the true" in text


def test_fair_value_uses_the_configured_candle_size(env, monkeypatch):
    # With 5-minute candles the page must annualize per 5 minutes, not per minute.
    monkeypatch.setattr(cfg, "GRANULARITY", 300)
    at = run()
    want = kalshi.fair_value(env.df, 75_743.66, 7.5, cfg.KALSHI_VOL_LOOKBACK_MINUTES, candle_seconds=300)
    assert metric(at, "P(YES)").value == f"{want['p_yes'] * 100:.1f}%"


def test_weights_note_does_not_claim_agg_delta_is_always_unit_weight(env):
    # weighting.py weights agg_delta from backtest_agg_delta.py's buckets, so once that has run
    # the table shows a non-1.00 weight right next to a note saying it "always" shows 1.00.
    go_live(env)                                                           # buy-heavy trades -> agg_delta bullish
    pathlib.Path(cfg.BACKTEST_RESULTS_PATH).write_text(json.dumps({
        "product_id": "BTC-USD", "granularity": 60, "candles_used": 4000, "generated_at": "2026-11-07T00:00:00+00:00",
        "lookahead_candles": 5, "strength_buckets": {}, "divergence_buckets": {},
        "indicator_buckets": {"agg_delta_bullish": {"hits": 70, "total": 100}},
    }))
    at = run()
    no_exception(at)
    table = next(d.value for d in at.dataframe if "Weight" in d.value.columns).set_index("Indicator")["Weight"]
    assert table["Agg. Exchange Delta"] == "1.40"                          # 1 + 2.0 * (0.70 - 0.5)
    assert "aggregated exchange delta always show 1.00" not in all_text(at)



def test_weights_table_explains_each_weight_and_noise_stays_at_one(env):
    import dashboard_data
    votes = {k: v for k, v in dashboard_data.load_state(cfg).snapshot.individual.items()
             if k in ("sma_cross", "ema_cross", "macd") and v != "neutral"}
    assert len(votes) >= 2
    (strong, sd), (noisy, nd) = list(votes.items())[:2]
    pathlib.Path(cfg.BACKTEST_RESULTS_PATH).write_text(json.dumps({
        "product_id": "BTC-USD", "granularity": 60, "candles_used": 4000, "generated_at": "2026-11-07T00:00:00+00:00",
        "lookahead_candles": 5, "strength_buckets": {}, "divergence_buckets": {},
        "base_rates": {"bullish": 0.5, "bearish": 0.5, "total": 3900},
        "indicator_buckets": {
            f"{strong}_{sd}": {"hits": 650, "total": 1000, "n_eff": 300.0, "base_rate": 0.5},   # z ~ 5.2
            f"{noisy}_{nd}": {"hits": 1030, "total": 2000, "n_eff": 570.0, "base_rate": 0.5},   # 51.5%, z ~ 0.7
        },
    }))
    at = run()
    no_exception(at)
    table = next(d.value for d in at.dataframe if "Weight" in d.value.columns).set_index("Indicator")
    assert table.loc[long(strong), "Weight"] == "1.30"
    assert "edge +15.0 pts" in table.loc[long(strong), "Why"]
    assert table.loc[long(noisy), "Weight"] == "1.00"
    assert "within noise" in table.loc[long(noisy), "Why"]


def test_sidebar_asks_for_a_rerun_when_results_predate_effective_sample_sizes(env):
    pathlib.Path(cfg.BACKTEST_RESULTS_PATH).write_text(json.dumps({
        "product_id": "BTC-USD", "granularity": 60, "candles_used": 3999, "generated_at": "2026-11-07T00:00:00+00:00",
        "lookahead_candles": 5, "strength_buckets": {}, "divergence_buckets": {}, "indicator_buckets": {}}))
    assert "re-run the backtest" in sidebar_text(run())


def long(name):
    return {"sma_cross": "SMA Cross", "ema_cross": "EMA Cross", "macd": "MACD"}[name]
