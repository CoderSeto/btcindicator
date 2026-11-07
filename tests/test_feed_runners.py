"""
Tests for the WebSocket runner callbacks: what each feed subscribes to, and what it does with
every kind of message the socket can deliver. These are the closures inside
exchange_feeds._*_runner and orderbook_feeds._coinbase_book_runner / _bybit_book_runner; the
Binance.US depth runner has its own harness in test_orderbook_feeds.py.

No sockets and no threads: websocket.WebSocketApp is replaced by a recorder, run_with_reconnect by
a single call to the runner's create_app, and the test plays the server -- open, messages, error,
close -- by calling the real callbacks.

Payload sources (the exchanges' docs sites were unreachable from the sandbox; see also the
docstrings of test_exchange_feeds.py and test_orderbook_feeds.py):
  - Bybit v5 orderbook delta: VERBATIM from github.com/bybit-exchange/docs @ 6233a5a (2026-11-06),
    docs/v5/websocket/public/orderbook.mdx, "Response Example" > Delta. The snapshot example there
    elides levels with "...", so the snapshot here is test_orderbook_feeds.BYBIT_SNAPSHOT (the same
    example with the elisions dropped). The same page's rules: a size of 0 deletes the level, and
    "u"=1 "is a snapshot data due to the restart of the service. So please overwrite your local
    orderbook".
  - Bybit publicTrade / allLiquidation, Binance @trade / forceOrder: test_exchange_feeds.py.
  - Coinbase level2_batch snapshot / l2update: test_orderbook_feeds.py (Coinbase's documented
    examples). Coinbase `match`: test_exchange_feeds.py (schema from Coinbase's official client).
  - Server-side control messages (subscription acks, Coinbase's "error" message) are only checked
    for being harmless or surfacing as status, so their exact shape doesn't matter here.
"""
import json
import threading
from types import SimpleNamespace

import pytest
import websocket

import exchange_feeds as ef
import orderbook_feeds as obf
from test_exchange_feeds import (BINANCE_FORCE_ORDER, BINANCE_TRADE, BYBIT_LIQUIDATION,
                                 BYBIT_TRADE, COINBASE_MATCH)
from test_orderbook_feeds import (COINBASE_L2UPDATE, COINBASE_SNAPSHOT, BYBIT_SNAPSHOT, _Cfg)

BYBIT_DOC_DELTA = {
    "topic": "orderbook.50.BTCUSDT",
    "type": "delta",
    "ts": 1687940967466,
    "data": {
        "s": "BTCUSDT",
        "b": [["30247.20", "30.028"], ["30245.40", "0.224"], ["30242.10", "1.593"],
              ["30240.30", "1.305"], ["30240.00", "0"]],
        "a": [["30248.70", "0"], ["30249.30", "0.892"], ["30249.50", "1.778"], ["30249.60", "0"],
              ["30251.90", "2.947"], ["30252.20", "0.659"], ["30252.50", "4.591"]],
        "u": 177400507,
        "seq": 66544703342,
    },
    "cts": 1687940967464,
}


class Server:
    """Plays the exchange's side of one connection through the runner's real callbacks."""

    def __init__(self, app):
        self.app = app
        self.url = app.url
        self.sent = []
        self.ws = SimpleNamespace(send=lambda text: self.sent.append(json.loads(text)))

    def open(self):
        self.app.on_open(self.ws)

    def send(self, msg):
        self.app.on_message(self.ws, msg if isinstance(msg, str) else json.dumps(msg))

    def error(self, err):
        self.app.on_error(self.ws, err)

    def close(self):
        self.app.on_close(self.ws, 1000, "bye")


@pytest.fixture
def connect(monkeypatch):
    """connect(runner, *args) -> Server for the connection that runner opens."""
    apps = []

    class FakeApp:
        def __init__(self, url, on_open, on_message, on_error, on_close):
            self.url, self.on_open, self.on_message = url, on_open, on_message
            self.on_error, self.on_close = on_error, on_close
            apps.append(self)

    monkeypatch.setattr(websocket, "WebSocketApp", FakeApp)
    one_connection = lambda name, create, stop, sink, holder: create()   # noqa: E731
    monkeypatch.setattr(ef, "run_with_reconnect", one_connection)
    monkeypatch.setattr(obf, "run_with_reconnect", one_connection)

    def _connect(runner, *args):
        runner(*args, threading.Event(), {})
        return Server(apps[-1])
    return _connect


@pytest.fixture
def trades(monkeypatch):
    """A TradeAggregator whose clock reads `now["t"]` (payload times are years old)."""
    now = {"t": 0.0}
    monkeypatch.setattr(ef, "time", SimpleNamespace(time=lambda: now["t"], sleep=lambda s: None))
    agg = ef.TradeAggregator(60, 1_000, 50, 50)
    agg.now = now
    return agg


def _levels(book, side):
    return dict(getattr(book, side))


# ---------- Coinbase trades ----------

def test_coinbase_trades_subscribe_to_matches_and_count_the_taker_side(connect, trades):
    s = connect(ef._coinbase_runner, trades, "BTC-USD")
    assert s.url == "wss://ws-feed.exchange.coinbase.com"
    s.open()
    assert s.sent == [{"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]}]
    assert trades.snapshot()["status"]["coinbase"] == "connected"

    trades.now["t"] = 1415348367.0 + 1
    s.send({"type": "subscriptions", "channels": [{"name": "matches", "product_ids": ["BTC-USD"]}]})
    s.send("not json {")
    s.send(COINBASE_MATCH)                                   # maker side "sell" -> a taker BUY
    snap = trades.snapshot()
    assert snap["buy_notional"] == pytest.approx(400.23 * 5.23512) and snap["sell_notional"] == 0.0
    assert snap["whale_events"][0]["side"] == "buy"          # $2,095 >= the $1,000 test threshold


def test_coinbase_trade_feed_reports_errors_and_disconnects(connect, trades):
    s = connect(ef._coinbase_runner, trades, "BTC-USD")
    s.open()
    s.error(ConnectionResetError("reset by peer"))
    assert trades.snapshot()["status"]["coinbase"] == "error: reset by peer"
    s.close()
    assert trades.snapshot()["status"]["coinbase"] == "disconnected"


# ---------- Binance.US trades and Binance futures liquidations ----------

def test_binance_us_trades_stream_by_lowercase_symbol_and_count_the_taker_side(connect, trades):
    s = connect(ef._binance_us_trade_runner, trades, "BTCUSDT")
    assert s.url == "wss://stream.binance.us:9443/ws/btcusdt@trade"       # stream named in the URL
    s.open()
    assert s.sent == []                                                    # nothing to subscribe to
    trades.now["t"] = 1672515782.136 + 1
    s.send(BINANCE_TRADE)                                                  # m=true: taker sold
    snap = trades.snapshot()
    assert snap["sell_notional"] == pytest.approx(0.001 * 100) and snap["buy_notional"] == 0.0
    assert snap["per_exchange"] == {"binance_us": {"buy": 0.0, "sell": pytest.approx(0.1)}}


def test_binance_liquidations_keep_only_the_tracked_symbol(connect, trades):
    s = connect(ef._binance_liquidation_runner, trades, "BTCUSDT")
    s.open()
    s.send({**BINANCE_FORCE_ORDER, "o": {**BINANCE_FORCE_ORDER["o"], "s": "ETHUSDT"}})   # all-market stream
    s.send(BINANCE_FORCE_ORDER)
    liqs = trades.snapshot()["liquidation_events"]
    assert [(l["exchange"], l["liquidated_side"], l["notional"]) for l in liqs] == \
        [("binance", "long", pytest.approx(9910 * 0.014))]
    assert trades.snapshot()["status"]["binance_liq"] == "connected"


# ---------- Bybit trades and liquidations ----------

def test_bybit_trades_subscribe_on_the_spot_stream_and_count_the_taker_side(connect, trades):
    s = connect(ef._bybit_trade_runner, trades, "BTCUSDT")
    assert s.url == "wss://stream.bybit.com/v5/public/spot"
    s.open()
    assert s.sent == [{"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}]
    trades.now["t"] = 1672304486.865 + 1
    s.send({"success": True, "ret_msg": "subscribe", "op": "subscribe"})   # the ack: ignored
    s.send(BYBIT_TRADE)
    snap = trades.snapshot()
    assert snap["buy_notional"] == pytest.approx(16578.5 * 0.001)
    assert snap["per_exchange"] == {"bybit": {"buy": pytest.approx(16.5785), "sell": 0.0}}


def test_bybit_liquidations_subscribe_on_the_linear_stream(connect, trades):
    s = connect(ef._bybit_liquidation_runner, trades, "BTCUSDT")
    assert s.url == "wss://stream.bybit.com/v5/public/linear"             # USDT perps, not spot
    s.open()
    assert s.sent == [{"op": "subscribe", "args": ["allLiquidation.BTCUSDT"]}]
    s.send(BYBIT_LIQUIDATION)
    liqs = trades.snapshot()["liquidation_events"]
    assert [(l["exchange"], l["liquidated_side"]) for l in liqs] == [("bybit", "short")]
    s.close()
    assert trades.snapshot()["status"]["bybit_liq"] == "disconnected"


# ---------- Coinbase level-2 depth ----------

def test_coinbase_book_subscribes_to_level2_batch_not_the_authenticated_level2(connect):
    # Gotcha #4: "level2" needs auth since 2023-08-01 and would connect, then never send a snapshot.
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._coinbase_book_runner, agg, "BTC-USD")
    assert s.url == "wss://ws-feed.exchange.coinbase.com"
    s.open()
    assert s.sent == [{"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["level2_batch"]}]
    assert agg._status["coinbase"] == "connecting"                         # not "connected" until a book


def test_coinbase_book_is_built_from_the_snapshot_then_patched_by_updates(connect):
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._coinbase_book_runner, agg, "BTC-USD")
    s.open()
    s.send(COINBASE_L2UPDATE)                                              # before the snapshot: no book
    assert not agg.is_ready("coinbase")

    s.send(COINBASE_SNAPSHOT)
    assert agg.is_ready("coinbase") and agg._status["coinbase"] == "connected"
    book = agg._books["coinbase"]
    assert _levels(book, "bids") == {10101.10: 0.45054140}
    assert _levels(book, "asks") == {10102.55: 0.57753524}

    s.send(COINBASE_L2UPDATE)                                              # adds a bid at 10101.80
    assert _levels(book, "bids") == {10101.10: 0.45054140, 10101.80: 0.162567}
    s.send({"type": "l2update", "product_id": "BTC-USD", "changes": [["sell", "10102.55", "0"]]})
    assert _levels(book, "asks") == {}                                     # size 0 removes the level


def test_coinbase_book_error_message_surfaces_as_status(connect):
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._coinbase_book_runner, agg, "BTC-USD")
    s.open()
    s.send({"type": "error", "message": "Failed to subscribe", "reason": "level2 requires authentication"})
    assert agg._status["coinbase"] == "error: Failed to subscribe"
    assert not agg.is_ready("coinbase")


def test_coinbase_book_is_discarded_on_close_and_rebuilt_after_reconnect(connect):
    # A book kept across a disconnect would miss every change made while it was down.
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._coinbase_book_runner, agg, "BTC-USD")
    s.open()
    s.send(COINBASE_SNAPSHOT)
    s.close()
    assert not agg.is_ready("coinbase") and agg._status["coinbase"] == "disconnected"
    s.send(COINBASE_L2UPDATE)                                              # a straggler: not applied
    assert not agg.is_ready("coinbase")
    s.open()                                                               # reconnect
    assert len(s.sent) == 2 and not agg.is_ready("coinbase")
    s.send(COINBASE_SNAPSHOT)
    assert agg.is_ready("coinbase")


# ---------- Bybit level-2 depth ----------

def test_bybit_book_subscribes_at_the_configured_depth_on_the_spot_stream(connect):
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._bybit_book_runner, agg, "BTCUSDT", 200)
    assert s.url == "wss://stream.bybit.com/v5/public/spot"
    s.open()
    assert s.sent == [{"op": "subscribe", "args": ["orderbook.200.BTCUSDT"]}]


def test_bybit_book_applies_the_documented_delta_rules(connect):
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._bybit_book_runner, agg, "BTCUSDT", 50)
    s.open()
    s.send(BYBIT_DOC_DELTA)                                                # before any snapshot: ignored
    assert not agg.is_ready("bybit")

    s.send(BYBIT_SNAPSHOT)
    book = agg._books["bybit"]
    assert agg._status["bybit"] == "connected" and book.update_id == 18521288
    assert _levels(book, "bids") == {16493.50: 0.006, 16493.00: 0.100}

    s.send(BYBIT_DOC_DELTA)
    bids, asks = _levels(book, "bids"), _levels(book, "asks")
    # "If you receive an amount that does not exist, insert it"; size 0 deletes (none to delete here).
    assert bids == {16493.50: 0.006, 16493.00: 0.100, 30247.20: 30.028, 30245.40: 0.224,
                    30242.10: 1.593, 30240.30: 1.305}
    assert 30240.00 not in bids and 30248.70 not in asks and 30249.60 not in asks
    assert asks[30252.50] == 4.591 and len(asks) == 2 + 5
    assert book.update_id == 177400507

    # "If the entry exists, you simply update the value"; and 0 deletes an existing entry.
    s.send({**BYBIT_DOC_DELTA, "data": {**BYBIT_DOC_DELTA["data"], "u": 177400508,
                                        "b": [["30247.20", "1.000"], ["16493.50", "0"]], "a": []}})
    bids = _levels(book, "bids")
    assert bids[30247.20] == 1.0 and 16493.50 not in bids


def test_bybit_book_is_overwritten_when_the_service_restarts(connect):
    # u == 1 arrives typed "delta" but is a full snapshot after a restart: overwrite, don't patch.
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._bybit_book_runner, agg, "BTCUSDT", 50)
    s.open()
    s.send(BYBIT_SNAPSHOT)
    s.send({**BYBIT_DOC_DELTA, "data": {**BYBIT_DOC_DELTA["data"], "u": 1}})
    book = agg._books["bybit"]
    assert 16493.50 not in _levels(book, "bids")                           # the old book is gone
    assert _levels(book, "bids")[30247.20] == 30.028 and book.update_id == 1
    assert 30240.00 not in _levels(book, "bids")                           # size 0 isn't a level


def test_bybit_book_is_discarded_on_close(connect):
    agg = obf.BookAggregator(_Cfg)
    s = connect(obf._bybit_book_runner, agg, "BTCUSDT", 50)
    s.open()
    s.send(BYBIT_SNAPSHOT)
    s.error(TimeoutError("ping timeout"))
    assert agg._status["bybit"] == "error: ping timeout"
    s.close()
    assert not agg.is_ready("bybit") and agg._status["bybit"] == "disconnected"
