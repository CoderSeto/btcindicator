"""
Tests for dashboard_data.load_state (the fetch + compute pipeline behind the
dashboard) and charts.build_figure. Network calls are stubbed; the real
indicator, signal, divergence, weighting and outlook code runs.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest
import config as real_cfg
import dashboard_data
import data_fetch
from charts import build_figure


def make_cfg(tmp_path, **over):
    ns = SimpleNamespace(**{k: getattr(real_cfg, k) for k in dir(real_cfg) if k.isupper()})
    ns.BACKTEST_RESULTS_PATH = str(tmp_path / "bt.json")
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def candles(n=200, seed=1, last=75_700.0):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    close = close / close[-1] * last
    return pd.DataFrame({
        "time": pd.date_range("2026-11-07 15:00", periods=n, freq="min"),
        "low": close * 0.9995, "high": close * 1.0005,
        "open": np.r_[close[0], close[:-1]], "close": close,
        "volume": rng.uniform(5, 15, n),
    })


REST_BOOK = {"bids": [[75_699.0 - i, 1.0] for i in range(30)],
             "asks": [[75_701.0 + i, 0.5] for i in range(30)]}


class Net:
    """Stand-in for the Coinbase REST layer: set `.df`, read the call counters."""

    def __init__(self):
        self.df = candles()
        self.calls = {"candles": 0, "book": 0}


@pytest.fixture
def net(monkeypatch):
    n = Net()

    def get_candles(*a, **k):
        n.calls["candles"] += 1
        return n.df.copy()

    def get_order_book(*a, **k):
        n.calls["book"] += 1
        return REST_BOOK
    monkeypatch.setattr(data_fetch, "get_candles", get_candles)
    monkeypatch.setattr(data_fetch, "get_order_book", get_order_book)
    return n


# ---------- error paths ----------

def test_fetch_failure_becomes_an_error_state(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("coinbase is down")
    monkeypatch.setattr(data_fetch, "get_candles", boom)
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error == "coinbase is down" and st.df is None and st.snapshot is None


def test_order_book_fetch_failure_is_also_an_error_state(tmp_path, monkeypatch, net):
    monkeypatch.setattr(data_fetch, "get_order_book", lambda *a, **k: (_ for _ in ()).throw(ValueError("no book")))
    assert dashboard_data.load_state(make_cfg(tmp_path)).error == "no book"


@pytest.mark.parametrize("n", [0, 10, 27])
def test_too_little_history_is_an_error_state(tmp_path, net, n):
    net.df = candles(200).head(n)
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error and "Not enough candle history" in st.error


def test_exactly_the_minimum_history_is_enough(tmp_path, net):
    cfg = make_cfg(tmp_path)
    need = max(cfg.SMA_SLOW, cfg.BOLLINGER_PERIOD, cfg.MACD_SLOW, cfg.RSI_PERIOD) + 2
    net.df = candles(need)
    assert dashboard_data.load_state(cfg).error is None
    net.df = candles(need - 1)
    assert dashboard_data.load_state(cfg).error


# ---------- the happy path ----------

def test_state_is_fully_populated_without_feeds(tmp_path, net):
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error is None
    assert st.snapshot.price == pytest.approx(net.df["close"].iloc[-1])
    assert {"sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "orderbook"} <= set(st.snapshot.individual)
    assert "agg_delta" not in st.snapshot.individual                  # absent, not neutral
    assert st.imbalance is not None and -1.0 <= st.imbalance <= 1.0   # legacy REST reading was taken
    assert net.calls["book"] == 1
    assert "rsi" in st.df.columns and "bb_upper" in st.df.columns     # indicators were computed
    assert set(st.outlook) == {"headline", "detail", "caution", "book_note", "confidence_lines", "weighted_line"}
    assert st.divergence["type"] in (None, "bullish", "bearish")
    assert st.backtest_results is None                                # no backtest file in tmp_path
    assert st.weighted["available"] is True                           # weights default to 1.0, vote still weighs


def test_rest_imbalance_matches_the_pure_function(tmp_path, net):
    import orderbook
    cfg = make_cfg(tmp_path)
    st = dashboard_data.load_state(cfg)
    assert st.imbalance == pytest.approx(orderbook.compute_imbalance(REST_BOOK, cfg.ORDER_BOOK_DEPTH))
    assert st.imbalance > 0                                           # 2x the bid size -> buy pressure


def test_live_depth_feed_skips_the_rest_book_and_records_none(tmp_path, net):
    book_snap = {
        "available": True,
        "aggregate": {"imbalance": 0.4, "imbalance_near": 0.4, "imbalance_far": 0.4, "microprice_bps": 0.0,
                      "agreement": 1.0, "bid_notional": 2e6, "ask_notional": 1e6, "venues": {},
                      "venue_count": 2, "excluded": {}, "walls": [], "band_bps": 25.0},
        "persistence": {"seconds": 40.0, "direction": "bullish", "mean_imbalance": 0.4},
        "flow": {"available": False, "reason": "not enough history yet"},
        "walls": [], "samples": 100, "history_seconds": 100.0, "status": {}, "books": {},
    }
    st = dashboard_data.load_state(make_cfg(tmp_path), book_snapshot=book_snap)
    assert net.calls["book"] == 0                                           # no redundant request
    assert st.imbalance is None                                       # "never taken", not a balanced 0.0
    assert st.snapshot.individual["orderbook"] == "bullish"           # held 40s past the gate
    assert st.book_snapshot is book_snap


def test_unavailable_book_snapshot_still_uses_the_rest_fallback(tmp_path, net):
    st = dashboard_data.load_state(make_cfg(tmp_path), book_snapshot={"available": False, "reason": "connecting"})
    assert net.calls["book"] == 1 and st.imbalance is not None


def test_agg_delta_included_only_with_trade_volume(tmp_path, net):
    cfg = make_cfg(tmp_path)
    feed = {"total_notional": 1_000_000.0, "delta_ratio": 0.6}
    st = dashboard_data.load_state(cfg, feed_snapshot=feed)
    assert st.snapshot.individual["agg_delta"] == "bullish" and st.feed_snapshot is feed

    quiet = dashboard_data.load_state(cfg, feed_snapshot={"total_notional": 0, "delta_ratio": 0.0})
    assert "agg_delta" not in quiet.snapshot.individual               # no trades is silence, not "neutral"
    assert "agg_delta" not in dashboard_data.load_state(cfg, feed_snapshot=None).snapshot.individual


def test_price_only_snapshot_drives_backtest_confidence_but_not_the_live_snapshot(tmp_path, net, monkeypatch):
    seen = {}
    real = backtest.get_confidence

    def spy(results, price_snapshot, div, min_sample):
        seen["names"] = set(price_snapshot.individual)
        return real(results, price_snapshot, div, min_sample)
    monkeypatch.setattr(backtest, "get_confidence", spy)
    dashboard_data.load_state(make_cfg(tmp_path), feed_snapshot={"total_notional": 1e6, "delta_ratio": 0.9})
    assert seen["names"] == {"sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume"}   # backtestable only


def test_backtest_results_flow_into_confidence_and_weights(tmp_path, net):
    cfg = make_cfg(tmp_path)
    first = dashboard_data.load_state(cfg)
    snap = first.snapshot
    price_only_votes = {k: v for k, v in snap.individual.items() if k in
                        ("sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume")}
    # Pre-seed a results file whose buckets match whatever the live price-only state is.
    import json
    from signals import SignalSnapshot
    po = SignalSnapshot(price=0.0, individual=price_only_votes)
    buckets = {f"{n}_{v}": {"hits": 90, "total": 100} for n, v in price_only_votes.items() if v != "neutral"}
    results = {"product_id": cfg.PRODUCT_ID, "granularity": cfg.GRANULARITY, "lookahead_candles": 5,
               "strength_buckets": {f"{po.composite}_{po.strength}": {"hits": 3, "total": 4}} if po.composite != "neutral" else {},
               "divergence_buckets": {}, "indicator_buckets": buckets}
    (tmp_path / "bt.json").write_text(json.dumps(results))

    st = dashboard_data.load_state(cfg)
    assert st.backtest_results == results
    if buckets:
        assert st.weighted["available"] is True
        assert any(v["weight"] > 1.0 for v in st.weighted["breakdown"].values())   # 90% hit rate -> up-weighted
    if po.composite != "neutral":
        assert any("Historically" in line for line in st.outlook["confidence_lines"])


def test_results_for_a_different_pair_are_ignored(tmp_path, net):
    import json
    (tmp_path / "bt.json").write_text(json.dumps({"product_id": "ETH-USD", "granularity": 60,
                                                  "indicator_buckets": {}}))
    assert dashboard_data.load_state(make_cfg(tmp_path)).backtest_results is None


def test_weighting_does_not_change_the_raw_composite(tmp_path, net):
    # Same data with and without backtest results: composite/strength (which drive alerts) must match.
    import json
    cfg = make_cfg(tmp_path)
    without = dashboard_data.load_state(cfg).snapshot
    (tmp_path / "bt.json").write_text(json.dumps({
        "product_id": cfg.PRODUCT_ID, "granularity": cfg.GRANULARITY,
        "indicator_buckets": {f"{n}_{d}": {"hits": 99, "total": 100} for n in
                              ("rsi", "macd", "sma_cross", "ema_cross", "bollinger", "volume")
                              for d in ("bullish", "bearish")}}))
    with_bt = dashboard_data.load_state(cfg).snapshot
    assert (with_bt.composite, with_bt.strength, with_bt.individual) == \
           (without.composite, without.strength, without.individual)


# ---------- charts.build_figure ----------

def full_frame():
    import indicators
    return indicators.compute_all(candles(), real_cfg)


def test_figure_has_the_expected_traces_and_layout():
    fig = build_figure(full_frame(), real_cfg)
    names = [t.name for t in fig.data]
    assert names == ["Price", f"SMA {real_cfg.SMA_FAST}", f"SMA {real_cfg.SMA_SLOW}", "BB Upper", "BB Lower",
                     "RSI", "MACD Hist", "MACD", "Signal"]
    assert fig.layout.height == 750 and fig.layout.xaxis.rangeslider.visible is False
    assert [a.text for a in fig.layout.annotations] == ["Price", "RSI", "MACD"]
    assert tuple(fig.layout.yaxis2.range) == (0, 100)                 # RSI axis is pinned


def test_figure_data_matches_the_frame():
    df = full_frame()
    fig = build_figure(df, real_cfg)
    by_name = {t.name: t for t in fig.data}
    np.testing.assert_allclose(by_name["Price"].close, df["close"])
    np.testing.assert_allclose(by_name["RSI"].y, df["rsi"])
    np.testing.assert_allclose(by_name["MACD"].y, df["macd"])
    assert len(by_name["Price"].x) == len(df)


def test_rsi_threshold_lines_use_config_values():
    fig = build_figure(full_frame(), real_cfg)
    ys = sorted(s.y0 for s in fig.layout.shapes)
    assert ys == [real_cfg.RSI_OVERSOLD, real_cfg.RSI_OVERBOUGHT]


def test_macd_histogram_colours_follow_sign_and_tolerate_warmup_nans():
    df = full_frame()
    df.loc[df.index[:5], "macd_hist"] = np.nan                         # warmup gap
    hist = next(t for t in build_figure(df, real_cfg).data if t.name == "MACD Hist")
    want = ["#2ca02c" if v >= 0 else "#d62728" for v in df["macd_hist"].fillna(0)]
    assert list(hist.marker.color) == want
    assert list(hist.marker.color[:5]) == ["#2ca02c"] * 5              # NaN treated as flat, drawn green


def test_figure_renders_to_json():
    assert build_figure(full_frame(), real_cfg).to_json()



# ---------- the still-forming candle (REVIEW_FINDINGS D5) ----------
#
# Coinbase stamps a candle with its bucket's START, so the candle stamped t is still forming until
# t + granularity. Signals scored on it see a partial close and partial volume (a forming 1-minute
# candle rarely reaches 2x average volume), while the backtest scored completed candles -- so the
# hit rates described a different signal from the live one. Whether Coinbase returns the forming
# candle couldn't be checked from the sandbox; deciding by timestamp makes the answer irrelevant.

from datetime import datetime, timezone


def _at(hh, mm, ss=0):
    return datetime(2026, 11, 7, hh, mm, ss, tzinfo=timezone.utc)


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_drop_forming_candle_decides_by_bucket_start_plus_granularity(unit):
    df = candles(5)                                                     # 15:00 .. 15:04
    df["time"] = df["time"].astype(f"datetime64[{unit}]")               # resolution must not matter
    done, forming = data_fetch.drop_forming_candle(df, 60, now=_at(15, 4, 30))
    assert len(done) == 4 and forming["close"] == df["close"].iloc[-1]
    done, forming = data_fetch.drop_forming_candle(df, 60, now=_at(15, 5, 0))   # 15:04 bucket just closed
    assert len(done) == 5 and forming is None
    done, forming = data_fetch.drop_forming_candle(df, 300, now=_at(15, 8, 0))  # 5-min buckets: 15:04 open
    assert len(done) == 4 and forming is not None


def test_drop_forming_candle_handles_empty_frames():
    empty = pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])
    done, forming = data_fetch.drop_forming_candle(empty, 60)
    assert done.empty and forming is None


def test_signals_ignore_the_forming_candle_but_the_live_price_does_not(tmp_path, net):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    df = candles(200)
    df["time"] = pd.date_range(end=now, periods=200, freq="min")        # last bucket started this minute
    df.loc[199, ["close", "high"]] = df.loc[198, "close"] * 1.02        # a partial candle's wild print
    df.loc[199, "volume"] = 10_000.0                                    # ...and a "volume spike"
    net.df = df
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error is None
    assert len(st.df) == 199                                            # signals ran on completed candles
    assert st.snapshot.price == pytest.approx(df["close"].iloc[198])
    assert st.live_price == pytest.approx(df["close"].iloc[199])        # the header/Kalshi still see it
    assert len(st.price_df) == 200
    assert st.snapshot.individual["volume"] == "neutral"                # the partial candle cast no vote


def test_without_a_forming_candle_live_price_is_the_last_close(tmp_path, net):
    st = dashboard_data.load_state(make_cfg(tmp_path))                  # fixture candles end in the past
    assert len(st.df) == len(net.df)
    assert st.live_price == pytest.approx(net.df["close"].iloc[-1]) == pytest.approx(st.snapshot.price)


# ---------- the live price comes from /ticker (the candles are cached ~5 minutes) ----------
# Ticker payload: the shape of the "Get product ticker" example in Coinbase Exchange's REST docs
# (ask, bid, volume, trade_id, price, size, time; numbers as strings). `price` was read live from
# the user's machine on 2026-11-07; re-verify the rest against the docs when they're reachable.
TICKER = {"ask": "75712.35", "bid": "75712.33", "volume": "8021.11", "trade_id": 912345678,
          "price": "75712.34", "size": "0.004", "time": "2026-11-07T18:07:29.512Z"}


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise data_fetch.requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.payload


def test_latest_price_reads_the_ticker(monkeypatch):
    seen = []
    monkeypatch.setattr(data_fetch.requests, "get", lambda url, **k: seen.append(url) or _Resp(TICKER))
    got = data_fetch.latest_price("BTC-USD", candles(5))
    assert seen == ["https://api.exchange.coinbase.com/products/BTC-USD/ticker"]
    assert got["source"] == "ticker" and got["price"] == 75_712.34 and got["reason"] is None
    assert got["time"] == pd.Timestamp("2026-11-07T18:07:29.512Z")


@pytest.mark.parametrize("failure", [
    lambda url, **k: _Resp({"message": "rate limited"}, status=429),
    lambda url, **k: (_ for _ in ()).throw(data_fetch.requests.ConnectionError("down")),
    lambda url, **k: _Resp({**TICKER, "price": "0"}),
    lambda url, **k: _Resp({k: v for k, v in TICKER.items() if k != "price"}),
])
def test_latest_price_falls_back_to_the_newest_candle_and_says_why(monkeypatch, failure):
    monkeypatch.setattr(data_fetch.requests, "get", failure)
    df = candles(5)
    got = data_fetch.latest_price("BTC-USD", df)
    assert got["source"] == "candles" and got["price"] == pytest.approx(df["close"].iloc[-1])
    assert got["reason"].startswith("ticker unavailable")


def test_latest_price_with_nothing_to_fall_back_on_is_none(monkeypatch):
    monkeypatch.setattr(data_fetch, "get_ticker", lambda p: (_ for _ in ()).throw(RuntimeError("down")))
    assert data_fetch.latest_price("BTC-USD", None)["price"] is None
    assert data_fetch.latest_price("BTC-USD", candles(5).iloc[:0])["price"] is None


def test_load_state_takes_the_live_price_from_the_ticker(tmp_path, net, monkeypatch):
    monkeypatch.setattr(data_fetch, "get_ticker", lambda p: TICKER)
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error is None
    assert st.live_price == 75_712.34 and st.live_price_source == "ticker" and st.live_price_note is None
    assert st.snapshot.price == pytest.approx(net.df["close"].iloc[-1])  # signals still on candles
    assert len(st.price_df) == len(net.df)                              # vol input unchanged


def test_load_state_survives_a_failed_ticker_and_flags_the_fallback(tmp_path, net, monkeypatch):
    monkeypatch.setattr(data_fetch, "get_ticker", lambda p: (_ for _ in ()).throw(RuntimeError("429")))
    st = dashboard_data.load_state(make_cfg(tmp_path))
    assert st.error is None                                             # not an error state
    assert st.live_price == pytest.approx(net.df["close"].iloc[-1])
    assert st.live_price_source == "candles" and "429" in st.live_price_note
