"""
edge_recorder.Recorder on fakes, and kalshi_api's orderbook/trades parsers.

The parser inputs here are HAND-BUILT from Kalshi's documented schema
(GET /markets/{ticker}/orderbook lists resting bids per side as
[price, quantity], in integer cents under `yes`/`no` and as dollar strings
under `yes_dollars`/`no_dollars`; GET /markets/trades returns trades with
yes_price / yes_price_dollars, count / count_fp, taker_side, created_time).
Kalshi was unreachable from the environment this was written in, so they are
NOT verbatim payloads: run `python edge_recorder.py --capture` and the
`test_captured_*` tests below pick the real files up and apply the same
assertions. Until then the parsers are untrusted (edge_recorder docstring).
"""
import csv
import json
import pathlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
import edge_recorder
import kalshi
import kalshi_api
from edge_recorder import RECORD_FIELDS, TRADE_FIELDS, Recorder

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class Cfg:
    def __init__(self, tmp_path):
        for k in dir(config):
            if k.isupper():
                setattr(self, k, getattr(config, k))
        self.EDGE_RECORD_PATH = str(tmp_path / "edge_record.csv")
        self.EDGE_TRADES_PATH = str(tmp_path / "edge_trades.csv")


# --- parsers ---------------------------------------------------------------

def test_parse_orderbook_cents_and_dollars_agree_and_mirror_no_bid():
    cents = {"orderbook": {"yes": [[55, 5], [56, 100]], "no": [[43, 20], [40, 7]]}}
    dollars = {"orderbook": {"yes_dollars": [["0.5500", "5"], ["0.5600", "100"]],
                             "no_dollars": [["0.4300", "20"], ["0.4000", "7"]]}}
    for payload in (cents, dollars):
        q = kalshi_api.parse_orderbook(payload)
        assert q["yes_bid"] == pytest.approx(0.56) and q["yes_bid_size"] == 100
        # YES ask is the mirror of the best NO bid: 1 - 0.43, with its size.
        assert q["yes_ask"] == pytest.approx(0.57) and q["yes_ask_size"] == 20


def test_parse_orderbook_empty_sides_are_none_not_zero():
    q = kalshi_api.parse_orderbook({"orderbook": {"yes": [], "no": [[30, 1]]}})
    assert q["yes_bid"] is None and q["yes_bid_size"] is None
    assert q["yes_ask"] == pytest.approx(0.70)


@pytest.mark.parametrize("payload", [
    {"orderbook": {"bids": [[1, 2]]}},            # unknown side names
    {"orderbook": {"yes": [[150, 1]], "no": []}},  # price out of range
    {"orderbook": {"yes": [[55]], "no": []}},      # not [price, qty]
    {"orderbook": "nope"},
    [],
])
def test_parse_orderbook_fails_loudly_on_unrecognised_schema(payload):
    with pytest.raises(kalshi_api.KalshiSchemaError):
        kalshi_api.parse_orderbook(payload)


def test_parse_trades_both_forms_sorted_oldest_first():
    payload = {"trades": [
        {"trade_id": "b", "ticker": "T", "created_time": "2026-11-08T10:00:05Z",
         "yes_price_dollars": "0.5600", "count_fp": "3.00", "taker_side": "no"},
        {"trade_id": "a", "ticker": "T", "created_time": "2026-11-08T10:00:01Z",
         "yes_price": 55, "count": 2, "taker_side": "yes"},
    ], "cursor": ""}
    out = kalshi_api.parse_trades(payload)
    assert [t["trade_id"] for t in out] == ["a", "b"]
    assert out[0]["yes_price"] == pytest.approx(0.55) and out[0]["count"] == 2
    assert out[1]["yes_price"] == pytest.approx(0.56) and out[1]["count"] == 3
    assert out[0]["created_time"].tzinfo is not None


@pytest.mark.parametrize("payload", [
    {"trades": [{"trade_id": "x", "created_time": "2026-11-08T10:00:01Z", "count": 1, "taker_side": "yes"}]},
    {"trades": [{"trade_id": "x", "created_time": "2026-11-08T10:00:01Z", "yes_price": 50, "count": 1,
                 "taker_side": "buy"}]},
    {"result": []},
])
def test_parse_trades_fails_loudly(payload):
    with pytest.raises(kalshi_api.KalshiSchemaError):
        kalshi_api.parse_trades(payload)


@pytest.mark.skipif(not (FIXTURES / "kalshi_orderbook.json").exists(),
                    reason="no captured orderbook yet: run `python edge_recorder.py --capture`")
def test_captured_orderbook_parses():
    q = kalshi_api.parse_orderbook(json.loads((FIXTURES / "kalshi_orderbook.json").read_text(encoding="utf-8")))
    assert set(q) == {"yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size"}
    if q["yes_bid"] is not None and q["yes_ask"] is not None:
        assert 0 < q["yes_bid"] < q["yes_ask"] < 1


@pytest.mark.skipif(not (FIXTURES / "kalshi_trades.json").exists(),
                    reason="no captured trades yet: run `python edge_recorder.py --capture`")
def test_captured_trades_parse():
    out = kalshi_api.parse_trades(json.loads((FIXTURES / "kalshi_trades.json").read_text(encoding="utf-8")))
    assert all(0 < t["yes_price"] < 1 and t["count"] > 0 for t in out)


# --- recorder --------------------------------------------------------------

T0 = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
CLOSE = T0 + timedelta(minutes=5)


def _market(ticker="KXBTC15M-TEST", strike=100000.0, close=CLOSE):
    return {"available": True, "ticker": ticker, "strike": strike, "close_time": close,
            "minutes_remaining": 5.0, "yes_bid": 0.5, "yes_ask": 0.52, "last_price": 0.51, "status": "open"}


def _candles(n=130, start=T0 - timedelta(minutes=135)):
    times = [start + timedelta(minutes=i) for i in range(n)]
    closes = [100000.0 * (1 + 0.0002 * ((i % 3) - 1)) for i in range(n)]
    return pd.DataFrame({"time": pd.to_datetime(times).tz_localize(None), "low": closes, "high": closes,
                         "open": closes, "close": closes, "volume": [1.0] * n})


def _book_snapshot():
    return {"available": True,
            "aggregate": {"imbalance": 0.25, "imbalance_near": 0.4, "microprice_bps": 1.5, "venue_count": 3},
            "persistence": {"seconds": 12.0, "direction": "bullish", "mean_imbalance": 0.2},
            "flow": {"available": True, "value": -0.1}}


def _trade_snapshot(now):
    return {"buy_notional": 600.0, "sell_notional": 400.0, "total_notional": 1000.0, "delta_ratio": 0.2,
            "status": {"coinbase": "connected", "binance_us": "connected", "binance_liq": "connected",
                       "bybit": "reconnecting", "bybit_liq": "connected"},
            "liquidation_events": [
                {"time": now - 10, "liquidated_side": "long", "notional": 5000.0},
                {"time": now - 100, "liquidated_side": "long", "notional": 99999.0},   # outside the window
                {"time": now - 3, "liquidated_side": "short", "notional": 700.0},
            ]}


class Fakes:
    def __init__(self, now):
        self.now = now
        self.market = _market()
        self.book = {"orderbook": {"yes": [[50, 10]], "no": [[48, 4]]}}
        self.trades = {"trades": []}
        self.spot = {"price": 100010.0, "source": "ticker"}
        self.calls = {"market": 0, "book": 0, "trades": 0, "vol": 0}
        self.book_error = None

    def clock(self):
        return self.now

    def fetch_market(self):
        self.calls["market"] += 1
        return self.market

    def fetch_orderbook(self, ticker):
        self.calls["book"] += 1
        if self.book_error:
            raise self.book_error
        self.now += 0.05                      # a 50 ms round trip
        return self.book

    def fetch_trades(self, ticker, min_ts):
        self.calls["trades"] += 1
        return self.trades

    def fetch_spot(self):
        return self.spot

    def fetch_candles(self):
        self.calls["vol"] += 1
        return _candles()

    def recorder(self, cfg, **kw):
        return Recorder(cfg, self.fetch_market, self.fetch_orderbook, self.fetch_trades, self.fetch_spot,
                        self.fetch_candles, clock=self.clock, log=lambda *_: None, **kw)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_tick_writes_fixed_schema_with_every_field_populated(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    rec = f.recorder(cfg, book_snapshot=_book_snapshot, trade_snapshot=lambda: _trade_snapshot(now))
    rec.tick(now)
    rows = _rows(cfg.EDGE_RECORD_PATH)
    assert list(rows[0].keys()) == RECORD_FIELDS
    r = rows[0]
    assert r["row_kind"] == "tick" and r["ticker"] == "KXBTC15M-TEST"
    assert float(r["secs_to_close"]) == pytest.approx(300.0)
    assert float(r["k_yes_bid"]) == 0.5 and float(r["k_yes_ask"]) == 0.52
    assert float(r["k_yes_bid_sz"]) == 10 and float(r["k_yes_ask_sz"]) == 4
    assert float(r["k_rtt_ms"]) == pytest.approx(50.0, abs=1.0)
    assert r["k_up"] == "1" and r["cb_up"] == "1" and r["book_up"] == "1" and r["agg_up"] == "1"
    assert float(r["cb_price"]) == 100010.0 and r["cb_source"] == "ticker"
    assert float(r["book_imb"]) == 0.25 and float(r["book_imb_near"]) == 0.4 and float(r["book_flow"]) == -0.1
    assert r["book_persist_dir"] == "bullish" and float(r["book_persist_s"]) == 12.0
    assert r["book_venues_up"] == "3" and r["agg_venues_up"] == "4"
    assert float(r["agg_delta"]) == 0.2
    assert float(r["liq_long_notional"]) == 5000.0 and float(r["liq_short_notional"]) == 700.0
    assert r["model_version"] == edge_recorder.MODEL_VERSION
    # The model's inputs are all on the row, and p_settle is recomputable from them exactly.
    vol = float(r["vol_ann"])
    expected = kalshi.settlement_probability(100010.0, 100000.0, float(r["secs_to_close"]) / 60, vol)
    assert float(r["p_settle"]) == pytest.approx(expected, abs=1e-6)
    assert float(r["p_point"]) == pytest.approx(
        kalshi.crossing_probability(100010.0, 100000.0, float(r["secs_to_close"]) / 60, vol), abs=1e-6)
    assert float(r["p_settle"]) > float(r["p_point"]) > 0.5      # gotcha 13: averaging is more extreme
    assert r["vol_candle_end"].startswith("2026-11-08T11:5")


def test_missing_sources_are_blank_not_zero(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    f.spot = {"price": None, "source": None, "reason": "down"}
    f.book_error = ConnectionError("kalshi down")
    rec = f.recorder(cfg)                      # no depth or flow feeds at all
    rec.tick(now)
    r = _rows(cfg.EDGE_RECORD_PATH)[0]
    assert r["k_up"] == "0" and r["cb_up"] == "0" and r["book_up"] == "0" and r["agg_up"] == "0"
    for k in ("k_yes_bid", "k_yes_ask", "k_rtt_ms", "cb_price", "book_imb", "book_flow", "agg_delta",
              "liq_long_notional", "p_settle", "p_point"):
        assert r[k] == "", k
    assert r["ticker"] == "KXBTC15M-TEST" and r["vol_ann"] != ""       # what IS known is still recorded


def test_quote_age_counts_from_last_good_book(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    rec = f.recorder(cfg)
    rec.tick(now)
    f.book_error = ConnectionError("blip")
    rec.tick(now + 1)
    rec.tick(now + 2)
    rows = _rows(cfg.EDGE_RECORD_PATH)
    assert rows[1]["k_up"] == "0" and rows[2]["k_up"] == "0"
    # The last good quote is carried with its age, so staleness is measurable, and never as fresh.
    assert float(rows[1]["k_yes_bid"]) == 0.5
    assert float(rows[2]["k_quote_age_s"]) == pytest.approx(2.0, abs=0.1)
    assert rows[2]["k_rtt_ms"] == ""


def test_unrecognised_kalshi_schema_stops_the_recorder(tmp_path):
    cfg = Cfg(tmp_path)
    f = Fakes(T0.timestamp())
    f.book = {"orderbook": {"levels": []}}
    with pytest.raises(kalshi_api.KalshiSchemaError):
        f.recorder(cfg).tick(T0.timestamp())
    assert not pathlib.Path(cfg.EDGE_RECORD_PATH).exists()


def test_gap_row_logged_when_ticks_stall(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    rec = f.recorder(cfg)
    rec.tick(now)
    rec.tick(now + 1)
    rec.tick(now + 1 + 42.5)
    rows = _rows(cfg.EDGE_RECORD_PATH)
    kinds = [r["row_kind"] for r in rows]
    assert kinds == ["tick", "tick", "gap", "tick"]
    gap = rows[2]
    assert float(gap["gap_seconds"]) == pytest.approx(42.5)
    assert all(gap[k] == "" for k in RECORD_FIELDS if k not in ("row_kind", "ts_utc", "gap_seconds"))
    assert rec.gaps_written == 1 and rec.rows_written == 3


def test_market_refresh_and_rollover_resets_quote(tmp_path):
    cfg = Cfg(tmp_path)
    cfg.EDGE_MARKET_REFRESH_SECONDS = 30.0
    now = T0.timestamp()
    f = Fakes(now)
    rec = f.recorder(cfg)
    for i in range(10):
        rec.tick(now + i)
    assert f.calls["market"] == 1                     # not re-fetched inside the refresh interval
    # The market closes: the next tick re-fetches even inside the interval, and the new
    # market starts with no carried quote.
    f.market = _market(ticker="KXBTC15M-NEXT", close=CLOSE + timedelta(minutes=15))
    f.book_error = ConnectionError("blip")
    t = CLOSE.timestamp() + 1
    rec.tick(t)
    rows = _rows(cfg.EDGE_RECORD_PATH)
    assert f.calls["market"] == 2
    assert rows[-1]["ticker"] == "KXBTC15M-NEXT" and rows[-1]["k_yes_bid"] == "" and rows[-1]["k_quote_age_s"] == ""


def test_trades_logged_once_each_in_fixed_schema(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    f.trades = {"trades": [
        {"trade_id": "t1", "ticker": "KXBTC15M-TEST", "created_time": "2026-11-08T11:59:58Z",
         "yes_price": 51, "count": 2, "taker_side": "yes"}]}
    rec = f.recorder(cfg)
    rec.tick(now)
    f.trades["trades"].append({"trade_id": "t2", "ticker": "KXBTC15M-TEST", "created_time": "2026-11-08T12:00:03Z",
                               "yes_price_dollars": "0.4900", "count_fp": "1.00", "taker_side": "no"})
    rec.tick(now + cfg.EDGE_TRADES_REFRESH_SECONDS)       # same t1 comes back, plus t2
    rec.tick(now + 2 * cfg.EDGE_TRADES_REFRESH_SECONDS)
    rows = _rows(cfg.EDGE_TRADES_PATH)
    assert list(rows[0].keys()) == TRADE_FIELDS
    assert [r["trade_id"] for r in rows] == ["t1", "t2"]
    assert rows[1]["taker_side"] == "no" and float(rows[1]["yes_price"]) == 0.49


def test_vol_refresh_is_rate_limited_and_blank_when_unavailable(tmp_path):
    cfg = Cfg(tmp_path)
    now = T0.timestamp()
    f = Fakes(now)
    rec = f.recorder(cfg)
    for i in range(5):
        rec.tick(now + i)
    assert f.calls["vol"] == 1
    flat = _candles()
    flat["close"] = 100000.0
    f.fetch_candles = lambda: flat                        # zero variance: no vol, no probability
    rec.fetch_candles = f.fetch_candles
    rec.tick(now + cfg.EDGE_VOL_REFRESH_SECONDS + 1)
    r = _rows(cfg.EDGE_RECORD_PATH)[-1]
    assert r["vol_ann"] == "" and r["p_settle"] == "" and r["cb_price"] != ""


def test_build_live_wires_without_network(monkeypatch, tmp_path):
    """The real wiring is constructed (feeds off) and every source is the project's own function."""
    cfg = Cfg(tmp_path)
    rec, handles = edge_recorder.build_live(cfg, with_feeds=False)
    assert handles == []
    assert rec.fetch_market is kalshi_api.current_btc15m_market
    assert rec.fetch_orderbook is kalshi_api.get_orderbook
    seen = {}
    monkeypatch.setattr(edge_recorder.data_fetch, "latest_price", lambda pid: seen.setdefault("pid", pid) or {"price": 1.0})
    rec.fetch_spot()
    assert seen["pid"] == cfg.PRODUCT_ID


def test_capture_writes_fixture_files(monkeypatch, tmp_path):
    monkeypatch.setattr(kalshi_api, "get_open_markets", lambda: [{"ticker": "KXBTC15M-X"}])
    monkeypatch.setattr(kalshi_api, "get_orderbook", lambda t, depth=10: {"orderbook": {"yes": [], "no": []}, "t": t})
    monkeypatch.setattr(kalshi_api, "get_trades", lambda t, limit=100, min_ts=None: {"trades": [], "t": t})
    out = edge_recorder.capture(tmp_path)
    names = sorted(p.name for p in out)
    assert names == ["kalshi_orderbook.json", "kalshi_orderbook_capture.txt", "kalshi_trades.json"]
    assert json.loads((tmp_path / "kalshi_orderbook.json").read_text(encoding="utf-8"))["t"] == "KXBTC15M-X"
