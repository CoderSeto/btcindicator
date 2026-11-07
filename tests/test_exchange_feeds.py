"""
Tests for exchange_feeds: the trade/liquidation parsers, TradeAggregator, and run_with_reconnect.

Payload sources. The exchanges' docs sites were unreachable from the sandbox these were written
in, so each payload comes from the exchange's own published source instead, pinned by commit so
it can be re-verified:

  - Binance spot <symbol>@trade and GET /api/v3/aggTrades: VERBATIM from the docs' own source,
    github.com/binance/binance-spot-api-docs @ 2bae3f5 (2026-11-06), web-socket-streams.md
    ("Trade Streams") and rest-api.md ("Compressed/Aggregate trades list"); `//` comments dropped.
  - Binance USD-M futures !forceOrder@arr: VERBATIM from Binance's official SDK test fixture,
    github.com/binance/binance-connector-python @ f75b4a5, clients/derivatives_trading_usds_futures/
    tests/unit/websocket_streams/test_market_api_ws_streams.py. The SDK documents `S` only as
    "Side"; "SELL = a long was liquidated" is the closing order's side (gotcha #3), not a quote.
  - Bybit v5 publicTrade and allLiquidation: VERBATIM from the docs' own source,
    github.com/bybit-exchange/docs @ 6233a5a (2026-11-06), docs/v5/websocket/public/trade.mdx and
    all-liquidation.mdx. The docs define trade `S` as "Side of taker" and liquidation `S` as
    "Position side ... When you receive a `Buy` update, this means that a long position has
    been liquidated".
  - Coinbase Exchange `match`: NOT a verbatim example -- none was reachable. Field names and types
    are from Coinbase's official client, github.com/coinbase/coinbase-pro-node @ 8bce637,
    index.d.ts (`Match`); the values are illustrative. What `side` MEANS is from the same client's
    lib/orderbook.js: match() looks the order up by `maker_order_id` in the tree for `side`, and
    asserts it is there -- so `side` is the MAKER's side, and the taker traded the other way.
    Re-check against https://docs.cdp.coinbase.com/exchange/websocket-feed/channels if reachable.

TradeAggregator runs on a fake clock. run_with_reconnect runs with no sockets: the WebSocketApp is
a stand-in whose run_forever() just advances a fake clock by how long that "connection" lasted.
"""
import threading
from types import SimpleNamespace

import pytest

import exchange_feeds


# ---------- parsers ----------

BINANCE_TRADE = {
    "e": "trade", "E": 1672515782136, "s": "BNBBTC", "t": 12345, "p": "0.001", "q": "100",
    "T": 1672515782136, "m": True, "M": True,
}

BINANCE_AGG_TRADE = {
    "a": 26129, "p": "0.01633102", "q": "4.70443515", "f": 27781, "l": 27781,
    "T": 1498793709153, "m": True, "M": True,
}

BINANCE_FORCE_ORDER = {
    "e": "forceOrder",
    "E": 1568014460893,
    "o": {
        "s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "0.014", "p": "9910",
        "ap": "9910", "X": "FILLED", "l": "0.014", "z": "0.014", "T": 1568014460893,
    },
    "ps": "BTCUSDT",
    "st": 1,
}

BYBIT_TRADE = {
    "topic": "publicTrade.BTCUSDT",
    "type": "snapshot",
    "ts": 1672304486868,
    "data": [
        {
            "T": 1672304486865, "s": "BTCUSDT", "S": "Buy", "v": "0.001", "p": "16578.50",
            "L": "PlusTick", "i": "20f43950-d8dd-5b31-9112-a178eb6023af", "BT": False,
            "seq": 1783284617,
        }
    ],
}

BYBIT_LIQUIDATION = {
    "topic": "allLiquidation.ROSEUSDT",
    "type": "snapshot",
    "ts": 1739502303204,
    "data": [{"T": 1739502302929, "s": "ROSEUSDT", "S": "Sell", "v": "20000", "p": "0.04499"}],
}

# Illustrative values on the `Match` schema from Coinbase's official client (see the docstring).
COINBASE_MATCH = {
    "type": "match", "trade_id": 10, "sequence": 50, "maker_order_id": "ac928c66-ca53-498f-9c13-a110027a60e8",
    "taker_order_id": "132fb6ae-456b-4654-b4e0-d681ac05cea1", "time": "2014-11-07T08:19:27.028459Z",
    "product_id": "BTC-USD", "size": "5.23512", "price": "400.23", "side": "sell",
}


def test_binance_trade_m_true_means_the_seller_was_the_taker():
    # "m": "Is the buyer the market maker?" -- true means the buy order was resting, so the
    # aggressor sold.
    assert exchange_feeds.parse_binance_trade(BINANCE_TRADE) == {
        "side": "sell", "price": 0.001, "size": 100.0, "time": 1672515782.136}
    assert exchange_feeds.parse_binance_trade({**BINANCE_TRADE, "m": False})["side"] == "buy"


def test_binance_agg_trade_rest_rows_parse_the_same_way():
    # backtest_agg_delta replays these REST rows through the same parser as the live stream.
    assert exchange_feeds.parse_binance_trade(BINANCE_AGG_TRADE) == {
        "side": "sell", "price": 0.01633102, "size": 4.70443515, "time": 1498793709.153}


def test_binance_force_order_sell_means_a_long_was_liquidated():
    out = exchange_feeds.parse_binance_liquidation(BINANCE_FORCE_ORDER)
    assert out == {"liquidated_side": "long", "price": 9910.0, "size": 0.014,
                   "time": 1568014460.893, "symbol": "BTCUSDT"}
    buy = {**BINANCE_FORCE_ORDER, "o": {**BINANCE_FORCE_ORDER["o"], "S": "BUY"}}
    assert exchange_feeds.parse_binance_liquidation(buy)["liquidated_side"] == "short"


def test_binance_force_order_symbol_filter_and_average_price_fallback():
    assert exchange_feeds.parse_binance_liquidation(BINANCE_FORCE_ORDER, symbol_filter="ETHUSDT") is None
    no_avg = {**BINANCE_FORCE_ORDER, "o": {**BINANCE_FORCE_ORDER["o"], "ap": "", "p": "9900"}}
    assert exchange_feeds.parse_binance_liquidation(no_avg)["price"] == 9900.0
    assert exchange_feeds.parse_binance_liquidation({"e": "trade"}) is None


def test_bybit_trade_side_is_the_taker_side():
    assert exchange_feeds.parse_bybit_trade(BYBIT_TRADE) == [
        {"side": "buy", "price": 16578.5, "size": 0.001, "time": 1672304486.865}]
    assert exchange_feeds.parse_bybit_trade({"topic": "orderbook.50.BTCUSDT", "data": []}) == []


def test_bybit_liquidation_side_is_the_position_side():
    # The opposite convention from Binance's field of the same letter (gotcha #3): "Sell" here is
    # a short position liquidated.
    assert exchange_feeds.parse_bybit_liquidation(BYBIT_LIQUIDATION) == [
        {"liquidated_side": "short", "price": 0.04499, "size": 20000.0, "time": 1739502302.929}]
    buy = {**BYBIT_LIQUIDATION, "data": [{**BYBIT_LIQUIDATION["data"][0], "S": "Buy"}]}
    assert exchange_feeds.parse_bybit_liquidation(buy)[0]["liquidated_side"] == "long"


def test_binance_and_bybit_liquidations_agree_once_normalized():
    # The same event -- a long liquidated -- arrives as S="SELL" from Binance and S="Buy" from Bybit.
    bybit_long = {**BYBIT_LIQUIDATION, "data": [{**BYBIT_LIQUIDATION["data"][0], "S": "Buy"}]}
    assert (exchange_feeds.parse_binance_liquidation(BINANCE_FORCE_ORDER)["liquidated_side"]
            == exchange_feeds.parse_bybit_liquidation(bybit_long)[0]["liquidated_side"] == "long")


def test_coinbase_match_side_is_the_makers_so_the_taker_traded_the_other_way():
    """
    A match with side "sell" filled a resting SELL order (Coinbase's own client removes the size
    from the maker's order in the ask tree), so the aggressor BOUGHT. The parser used to pass
    `side` straight through as the taker side, counting every Coinbase trade backwards in
    agg_delta and labelling Coinbase whale trades backwards.
    """
    out = exchange_feeds.parse_coinbase_match(COINBASE_MATCH)
    assert out["side"] == "buy"
    assert (out["price"], out["size"]) == (400.23, 5.23512)
    assert out["time"] == pytest.approx(1415348367.028459)
    assert exchange_feeds.parse_coinbase_match({**COINBASE_MATCH, "side": "buy"})["side"] == "sell"
    assert exchange_feeds.parse_coinbase_match({"type": "last_match", **{k: v for k, v in COINBASE_MATCH.items()
                                                                        if k != "type"}}) is None


def test_every_venue_reports_an_aggressive_buy_as_buy():
    # The property agg_delta rests on: one taker-buy, described in each venue's own convention,
    # normalizes to the same side.
    coinbase = exchange_feeds.parse_coinbase_match({**COINBASE_MATCH, "side": "sell"})       # maker sold
    binance = exchange_feeds.parse_binance_trade({**BINANCE_TRADE, "m": False})              # buyer not maker
    bybit = exchange_feeds.parse_bybit_trade(BYBIT_TRADE)[0]                                 # taker "Buy"
    assert coinbase["side"] == binance["side"] == bybit["side"] == "buy"


# ---------- TradeAggregator ----------

@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(exchange_feeds, "time", SimpleNamespace(time=lambda: t["now"], sleep=lambda s: None))
    return t


def _agg(**kw):
    return exchange_feeds.TradeAggregator(window_seconds=60, whale_usd_threshold=100_000,
                                          max_whale_events=3, max_liquidation_events=2, **kw)


def test_delta_ratio_and_per_exchange_split_hand_computed(clock):
    agg = _agg()
    now = clock["now"]
    agg.record_trade("coinbase", "buy", 100.0, 3.0, ts=now)          # 300 buy
    agg.record_trade("bybit", "sell", 50.0, 2.0, ts=now)             # 100 sell
    snap = agg.snapshot()
    assert (snap["buy_notional"], snap["sell_notional"], snap["total_notional"]) == (300.0, 100.0, 400.0)
    assert snap["delta_ratio"] == pytest.approx((300 - 100) / 400)
    assert snap["per_exchange"] == {"coinbase": {"buy": 300.0, "sell": 0.0},
                                    "bybit": {"buy": 0.0, "sell": 100.0}}


def test_trades_leave_the_window_by_the_local_clock(clock):
    agg = _agg()
    agg.record_trade("coinbase", "buy", 100.0, 1.0, ts=clock["now"] - 61)   # already outside
    agg.record_trade("coinbase", "sell", 100.0, 1.0, ts=clock["now"] - 30)
    assert agg.snapshot()["buy_notional"] == 0.0
    clock["now"] += 31                                                      # the sell is now 61s old
    snap = agg.snapshot()
    assert snap["total_notional"] == 0.0 and snap["delta_ratio"] == 0.0     # "no trades", callers check total


def test_whales_are_at_or_above_the_threshold_and_capped(clock):
    agg = _agg()
    agg.record_trade("coinbase", "buy", 100_000.0, 0.999, ts=clock["now"])  # $99,900: not a whale
    for i in range(4):
        agg.record_trade("coinbase", "buy", 100_000.0, 1.0, ts=clock["now"] + i)   # exactly $100k
    whales = agg.snapshot()["whale_events"]
    assert len(whales) == 3 and [w["time"] for w in whales] == [clock["now"] + i for i in (1, 2, 3)]


def test_callbacks_run_outside_the_lock_and_cannot_kill_the_feed(clock):
    # A callback that reads the aggregator would deadlock if it ran under the lock.
    seen = []
    agg = _agg(on_whale_trade=lambda e: seen.append(agg.snapshot()["total_notional"]),
               on_liquidation=lambda e: 1 / 0)
    t = threading.Thread(target=agg.record_trade, args=("coinbase", "buy", 200_000.0, 1.0, clock["now"]))
    t.start()
    t.join(2)
    assert not t.is_alive() and seen == [200_000.0]
    agg.record_liquidation("bybit", "long", 75_000.0, 1.0, ts=clock["now"])   # raises inside: swallowed
    assert agg.snapshot()["liquidation_events"][0]["notional"] == 75_000.0


def test_liquidations_are_capped_and_a_missing_timestamp_means_now(clock):
    agg = _agg()
    for side in ("long", "short", "long"):
        agg.record_liquidation("binance", side, 100.0, 1.0)
    liqs = agg.snapshot()["liquidation_events"]
    assert [l["liquidated_side"] for l in liqs] == ["short", "long"]        # maxlen 2
    assert liqs[-1]["time"] == clock["now"]


# ---------- run_with_reconnect ----------

class _Stop:
    """A stop_event stand-in that records every wait (the backoff) and stops after N connections."""

    def __init__(self, connections):
        self.remaining = connections
        self.waits = []

    def is_set(self):
        return self.remaining <= 0

    def wait(self, seconds):
        self.waits.append(seconds)
        return self.is_set()


class _Sink:
    def __init__(self):
        self.states = []

    def set_status(self, name, state):
        self.states.append(state)


def _run(monkeypatch, lifetimes):
    clock = {"t": 0.0}
    slept = []
    monkeypatch.setattr(exchange_feeds, "time", SimpleNamespace(
        time=lambda: clock["t"], sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s))))
    stop = _Stop(len(lifetimes))
    queue = list(lifetimes)

    def create_app():
        def run_forever(**kwargs):
            clock["t"] += queue.pop(0)
            stop.remaining -= 1
        return SimpleNamespace(run_forever=run_forever)

    exchange_feeds.run_with_reconnect("x", create_app, stop, _Sink(), {})
    return stop.waits + slept


def test_backoff_resets_after_a_connection_that_stayed_up(monkeypatch):
    """
    The backoff doubled on every disconnect for the life of the process and never came back down,
    so after a handful of ordinary drops (exchanges routinely cycle connections) every reconnect
    cost the full 30s of missing data -- long enough to drop a venue from the book aggregate.
    A connection that held for a while proves the path is healthy; the next drop starts at 1s.
    """
    # three instant failures, then an hour up, then two more instant failures
    waits = _run(monkeypatch, [0.1, 0.1, 0.1, 3600.0, 0.1, 0.1, 0.1])
    assert waits == [1, 2, 4, 1, 2, 4]


def test_backoff_still_grows_and_caps_while_connections_keep_failing(monkeypatch):
    assert _run(monkeypatch, [0.1] * 8) == [1, 2, 4, 8, 16, 30, 30]


def test_backoff_waits_on_the_stop_event_so_stop_is_not_delayed_up_to_30s(monkeypatch):
    # time.sleep(backoff) ignored stop(); waiting on the event returns as soon as it's set.
    clock = {"t": 0.0}
    monkeypatch.setattr(exchange_feeds, "time", SimpleNamespace(
        time=lambda: clock["t"], sleep=lambda s: (_ for _ in ()).throw(AssertionError("time.sleep used"))))
    stop = _Stop(2)
    exchange_feeds.run_with_reconnect(
        "x", lambda: SimpleNamespace(run_forever=lambda **k: setattr(stop, "remaining", stop.remaining - 1)),
        stop, _Sink(), {})
    assert stop.waits == [1]
