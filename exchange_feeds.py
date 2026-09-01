"""
Multi-exchange live data feeds: connects to Coinbase, Binance.US, and
Bybit's public trade WebSocket streams (plus Binance's and Bybit's
liquidation streams) and aggregates them into:

  - a rolling buy/sell "delta" across exchanges (feeds the agg_delta signal)
  - a running list of large ("whale") individual trades
  - a running list of liquidation events

This is the piece that makes the bot see more than one exchange at once,
inspired by tools like aggr.trade -- but built directly against each
exchange's own public API rather than depending on a third-party service,
since aggr.trade doesn't expose one for external use.

Everything here is read-only public market data. No API keys are used or
needed.

Run `python exchange_feeds.py` standalone to sanity-check connectivity --
it connects for a short window, prints what it receives from each
exchange, and exits. Useful for confirming your network can actually reach
these exchanges before relying on the dashboard or console bot.
"""
import json
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import requests
import websocket

# Bounds the initial connection attempt (not just post-connect pings) -- without
# this, a network path that silently drops packets rather than rejecting the
# connection can hang a feed thread indefinitely. 15s is generous for a normal
# handshake and still short enough that a genuinely dead path gets retried soon.
websocket.setdefaulttimeout(15)

# --- Symbol mapping -------------------------------------------------------
# The bot's config uses Coinbase-style product IDs (e.g. "BTC-USD"). Binance
# and Bybit use a different convention (e.g. "BTCUSDT"). Pairs with no clean
# USDT equivalent (e.g. "BTC-EUR") are simply not fed to those exchanges --
# the aggregator falls back to whichever exchanges DO support the pair.
_USDT_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
}


def to_binance_symbol(product_id: str) -> Optional[str]:
    return _USDT_SYMBOL_MAP.get(product_id)


def to_bybit_symbol(product_id: str) -> Optional[str]:
    return _USDT_SYMBOL_MAP.get(product_id)


# --- Historical trade fetching (Binance.US only) -----------------------------
# Of the three exchanges the live feeds use, only Binance.US has a REST
# endpoint that supports genuine time-range historical queries:
#   - Coinbase's /trades endpoint is cursor/trade-ID paginated, not
#     time-range -- covering multiple days would take an impractical number
#     of requests for a liquid pair.
#   - Bybit's recent-trade endpoint caps at the last 60 trades, no
#     pagination at all.
#   - Binance.US's aggTrades endpoint (public, no auth) supports
#     startTime/endTime range queries, up to 1000 trades per request,
#     paginated forward in time.
# See backtest_agg_delta.py, which uses this to reconstruct historical
# agg_delta readings.

_BINANCE_US_REST_BASE = "https://api.binance.us"


def fetch_binance_us_agg_trades(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
    """
    One page of historical aggregate trades from Binance.US's public REST
    API for [start_ms, end_ms] (both Unix milliseconds), capped at `limit`
    (max 1000). Returns raw trade dicts -- same p/q/T/m schema as the live
    WebSocket trade stream, so parse_binance_trade() works on them
    unchanged. For a range that might contain more than `limit` trades,
    use fetch_binance_us_trade_history() to paginate.
    """
    url = f"{_BINANCE_US_REST_BASE}/api/v3/aggTrades"
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_binance_us_trade_history(symbol: str, start_ms: int, end_ms: int,
                                    pause: float = 0.3, max_requests: int = 300) -> list:
    """
    Paginates fetch_binance_us_agg_trades() forward in time to cover the
    full [start_ms, end_ms] range. Returns a time-sorted list of
    normalized trade dicts (as parsed by parse_binance_trade).

    max_requests is a safety cap -- a very active window could otherwise
    need an unpredictable number of requests; pagination stops early and
    prints a warning if it's hit, returning whatever was collected so far.
    """
    all_trades = []
    cursor = start_ms
    requests_made = 0

    while cursor <= end_ms and requests_made < max_requests:
        try:
            raw = fetch_binance_us_agg_trades(symbol, cursor, end_ms)
        except requests.exceptions.RequestException as e:
            print(f"[exchange_feeds] stopping Binance.US trade history fetch after a request error: {e}")
            break
        requests_made += 1

        if not raw:
            break

        for item in raw:
            trade = parse_binance_trade(item)
            if trade:
                all_trades.append(trade)

        last_ms = raw[-1]["T"]
        if last_ms < cursor:
            break  # shouldn't happen, but guards against an infinite loop either way
        cursor = last_ms + 1

        if len(raw) < 1000:
            break  # a partial page means we've reached end_ms

        time.sleep(pause)

    if requests_made >= max_requests:
        print(
            f"[exchange_feeds] stopped Binance.US trade history pagination after "
            f"{max_requests} requests (safety cap) -- results may not cover the full requested window."
        )

    all_trades.sort(key=lambda t: t["time"])
    return all_trades


def _parse_iso_time(s) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


# --- Message parsers --------------------------------------------------------
# Each parser takes a raw decoded message and returns a normalized dict (or
# a list of them, for exchanges that batch multiple trades per message), or
# None/[] if the message isn't a type we care about. Kept as pure functions
# so they can be tested directly against real example payloads without a
# live connection.

def parse_coinbase_match(msg: dict) -> Optional[dict]:
    """Coinbase Exchange 'matches' channel. 'side' is already the taker side."""
    if msg.get("type") != "match":
        return None
    try:
        return {
            "side": msg["side"],  # "buy" or "sell"
            "price": float(msg["price"]),
            "size": float(msg["size"]),
            "time": _parse_iso_time(msg.get("time")),
        }
    except (KeyError, ValueError, TypeError):
        return None


def parse_binance_trade(msg: dict) -> Optional[dict]:
    """
    Binance spot @trade stream. 'm' (isBuyerMaker) is the INVERSE of the
    taker side: m=True means the buyer was resting (maker), so the SELLER
    was the aggressor -- i.e. the trade was taker-sell.
    """
    if "p" not in msg or "q" not in msg or "m" not in msg:
        return None
    try:
        side = "sell" if msg["m"] else "buy"
        return {
            "side": side,
            "price": float(msg["p"]),
            "size": float(msg["q"]),
            "time": msg.get("T", 0) / 1000.0,
        }
    except (KeyError, ValueError, TypeError):
        return None


def parse_binance_liquidation(msg: dict, symbol_filter: Optional[str] = None) -> Optional[dict]:
    """
    Binance all-market forceOrder stream. 'S' is the SIDE OF THE CLOSING
    ORDER, which is the *opposite* convention from Bybit's liquidation feed:
    S="SELL" means the system sold to close a long (a long was liquidated);
    S="BUY" means it bought to close a short (a short was liquidated).
    Normalized here to liquidated_side "long"/"short" either way.
    """
    if msg.get("e") != "forceOrder":
        return None
    o = msg.get("o", {})
    try:
        symbol = o["s"]
        if symbol_filter and symbol != symbol_filter:
            return None
        liquidated_side = "long" if o["S"] == "SELL" else "short"
        price = float(o["ap"]) if o.get("ap") else float(o["p"])
        return {
            "liquidated_side": liquidated_side,
            "price": price,
            "size": float(o["q"]),
            "time": o.get("T", msg.get("E", 0)) / 1000.0,
            "symbol": symbol,
        }
    except (KeyError, ValueError, TypeError):
        return None


def parse_bybit_trade(msg: dict) -> list:
    """Bybit v5 publicTrade.{symbol}. 'S' is already the taker side."""
    if not str(msg.get("topic", "")).startswith("publicTrade."):
        return []
    out = []
    for item in msg.get("data", []) or []:
        try:
            out.append({
                "side": "buy" if item["S"] == "Buy" else "sell",
                "price": float(item["p"]),
                "size": float(item["v"]),
                "time": item.get("T", 0) / 1000.0,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def parse_bybit_liquidation(msg: dict) -> list:
    """
    Bybit v5 allLiquidation.{symbol}. 'S' here is the POSITION side that
    was liquidated directly -- S="Buy" means a long was liquidated, which
    is the OPPOSITE convention from Binance's field of the same letter.
    Normalized here to liquidated_side "long"/"short" either way.
    """
    if not str(msg.get("topic", "")).startswith("allLiquidation."):
        return []
    out = []
    for item in msg.get("data", []) or []:
        try:
            out.append({
                "liquidated_side": "long" if item["S"] == "Buy" else "short",
                "price": float(item["p"]),
                "size": float(item["v"]),
                "time": item.get("T", 0) / 1000.0,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


# --- Aggregator --------------------------------------------------------------

class TradeAggregator:
    """
    Thread-safe rolling aggregator. record_trade()/record_liquidation() are
    called from WebSocket callback threads (one per exchange); snapshot()
    is called from the main thread (console loop or Streamlit rerun).

    on_whale_trade / on_liquidation are optional callbacks invoked with the
    event dict the moment a whale trade or liquidation is recorded -- for
    real-time console alerts. They're called OUTSIDE the internal lock, so
    a slow or re-entrant callback can't block other threads or deadlock.
    """

    def __init__(self, window_seconds: int, whale_usd_threshold: float,
                 max_whale_events: int, max_liquidation_events: int,
                 on_whale_trade: Optional[Callable] = None, on_liquidation: Optional[Callable] = None):
        self.window_seconds = window_seconds
        self.whale_usd_threshold = whale_usd_threshold
        self.on_whale_trade = on_whale_trade
        self.on_liquidation = on_liquidation

        self._lock = threading.Lock()
        self._trades = deque()  # (ts, exchange, side, notional)
        self._whale_events = deque(maxlen=max_whale_events)
        self._liquidation_events = deque(maxlen=max_liquidation_events)
        self._status = {}

    def set_status(self, exchange: str, status: str):
        with self._lock:
            self._status[exchange] = status

    def record_trade(self, exchange: str, side: str, price: float, size: float, ts: Optional[float] = None):
        ts = ts or time.time()
        notional = price * size
        is_whale = notional >= self.whale_usd_threshold
        event = None
        with self._lock:
            self._trades.append((ts, exchange, side, notional))
            self._prune(ts)
            if is_whale:
                event = {
                    "time": ts, "exchange": exchange, "side": side,
                    "price": price, "size": size, "notional": notional,
                }
                self._whale_events.append(event)
        if event and self.on_whale_trade:
            try:
                self.on_whale_trade(event)
            except Exception:
                pass  # a bug in the caller's callback must never kill a feed thread

    def record_liquidation(self, exchange: str, liquidated_side: str, price: float, size: float,
                            ts: Optional[float] = None):
        ts = ts or time.time()
        event = {
            "time": ts, "exchange": exchange, "liquidated_side": liquidated_side,
            "price": price, "size": size, "notional": price * size,
        }
        with self._lock:
            self._liquidation_events.append(event)
        if self.on_liquidation:
            try:
                self.on_liquidation(event)
            except Exception:
                pass

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            self._prune(now)
            buy = sum(n for (_, _, side, n) in self._trades if side == "buy")
            sell = sum(n for (_, _, side, n) in self._trades if side == "sell")
            per_exchange: dict = {}
            for (_, ex, side, n) in self._trades:
                d = per_exchange.setdefault(ex, {"buy": 0.0, "sell": 0.0})
                d[side] += n
            whales = list(self._whale_events)
            liqs = list(self._liquidation_events)
            status = dict(self._status)

        total = buy + sell
        delta_ratio = (buy - sell) / total if total > 0 else 0.0
        return {
            "window_seconds": self.window_seconds,
            "buy_notional": buy,
            "sell_notional": sell,
            "total_notional": total,
            "delta_ratio": delta_ratio,
            "per_exchange": per_exchange,
            "whale_events": whales,
            "liquidation_events": liqs,
            "status": status,
        }


# --- WebSocket runners ---------------------------------------------------

def _reconnecting_run(name: str, create_app_fn, stop_event: threading.Event,
                       aggregator: TradeAggregator, holder: dict):
    backoff = 1
    while not stop_event.is_set():
        aggregator.set_status(name, "connecting")
        ws_app = create_app_fn()
        holder["ws"] = ws_app
        try:
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            aggregator.set_status(name, f"error: {e}")
        if stop_event.is_set():
            break
        aggregator.set_status(name, "reconnecting")
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)
    aggregator.set_status(name, "stopped")


def _coinbase_runner(aggregator: TradeAggregator, product_id: str, stop_event: threading.Event, holder: dict):
    name = "coinbase"
    url = "wss://ws-feed.exchange.coinbase.com"

    def on_open(ws):
        aggregator.set_status(name, "connected")
        ws.send(json.dumps({"type": "subscribe", "product_ids": [product_id], "channels": ["matches"]}))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        trade = parse_coinbase_match(msg)
        if trade:
            aggregator.record_trade(name, trade["side"], trade["price"], trade["size"], trade["time"])

    def on_error(ws, error):
        aggregator.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        aggregator.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    _reconnecting_run(name, create_app, stop_event, aggregator, holder)


def _binance_us_trade_runner(aggregator: TradeAggregator, symbol: str, stop_event: threading.Event, holder: dict):
    """
    Binance.US is a separate, independently-licensed exchange from
    Binance.com with its own distinct order book -- not just a mirror.
    Binance.com's spot API/WebSocket blocks connections from the US for
    regulatory reasons; Binance.US exists specifically to serve that
    market instead, unaffected by that block.

    This is the only Binance spot source connected (binance.com's own spot
    feed was dropped -- it can't be reached from the US, so there's no
    point paying the reconnect overhead for it here). If you ever want the
    deeper, more liquid binance.com feed back (e.g. running from outside
    the US), its runner is straightforward to re-add: same shape as this
    one, just wss://stream.binance.com:443/ws/{symbol}@trade instead.

    Binance.US's WebSocket API mirrors Binance.com's format exactly (same
    stream naming, same /ws/ path, and their REST API confirms identical
    isBuyerMaker field semantics) -- so the same parser applies unchanged.
    Binance.US doesn't offer retail derivatives, so there's no liquidation
    feed equivalent to add here.
    """
    name = "binance_us"
    url = f"wss://stream.binance.us:9443/ws/{symbol.lower()}@trade"

    def on_open(ws):
        aggregator.set_status(name, "connected")

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        trade = parse_binance_trade(msg)
        if trade:
            aggregator.record_trade(name, trade["side"], trade["price"], trade["size"], trade["time"])

    def on_error(ws, error):
        aggregator.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        aggregator.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    _reconnecting_run(name, create_app, stop_event, aggregator, holder)


def _binance_liquidation_runner(aggregator: TradeAggregator, symbol: str, stop_event: threading.Event, holder: dict):
    # Confirmed current (post 2026-04-23 migration) endpoint: liquidations
    # are routed under /market, not the legacy unrouted /ws path.
    name = "binance_liq"
    url = "wss://fstream.binance.com/market/ws/!forceOrder@arr"

    def on_open(ws):
        aggregator.set_status(name, "connected")

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        liq = parse_binance_liquidation(msg, symbol_filter=symbol)
        if liq:
            aggregator.record_liquidation("binance", liq["liquidated_side"], liq["price"], liq["size"], liq["time"])

    def on_error(ws, error):
        aggregator.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        aggregator.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    _reconnecting_run(name, create_app, stop_event, aggregator, holder)


def _bybit_trade_runner(aggregator: TradeAggregator, symbol: str, stop_event: threading.Event, holder: dict):
    name = "bybit"
    url = "wss://stream.bybit.com/v5/public/spot"

    def on_open(ws):
        aggregator.set_status(name, "connected")
        ws.send(json.dumps({"op": "subscribe", "args": [f"publicTrade.{symbol}"]}))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        for trade in parse_bybit_trade(msg):
            aggregator.record_trade(name, trade["side"], trade["price"], trade["size"], trade["time"])

    def on_error(ws, error):
        aggregator.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        aggregator.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    _reconnecting_run(name, create_app, stop_event, aggregator, holder)


def _bybit_liquidation_runner(aggregator: TradeAggregator, symbol: str, stop_event: threading.Event, holder: dict):
    name = "bybit_liq"
    url = "wss://stream.bybit.com/v5/public/linear"

    def on_open(ws):
        aggregator.set_status(name, "connected")
        ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{symbol}"]}))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        for liq in parse_bybit_liquidation(msg):
            aggregator.record_liquidation("bybit", liq["liquidated_side"], liq["price"], liq["size"], liq["time"])

    def on_error(ws, error):
        aggregator.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        aggregator.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    _reconnecting_run(name, create_app, stop_event, aggregator, holder)


class FeedHandle:
    """Handle to a running set of exchange feed threads."""

    def __init__(self, aggregator: TradeAggregator, stop_event: threading.Event, holders: list):
        self.aggregator = aggregator
        self._stop_event = stop_event
        self._holders = holders

    def snapshot(self) -> dict:
        return self.aggregator.snapshot()

    def stop(self):
        self._stop_event.set()
        for holder in self._holders:
            ws = holder.get("ws")
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


def start_feeds(cfg, on_whale_trade: Optional[Callable] = None, on_liquidation: Optional[Callable] = None) -> FeedHandle:
    """
    Starts background WebSocket threads for every exchange that supports
    cfg.PRODUCT_ID, and returns a FeedHandle to read snapshots from. Safe
    to call even if cfg.PRODUCT_ID has no Binance/Bybit equivalent (e.g.
    "BTC-EUR") -- those exchanges are simply skipped, Coinbase still runs.

    on_whale_trade / on_liquidation are optional callbacks fired in
    real time as events are recorded (e.g. for console alerts) -- see
    TradeAggregator for details. Leave them unset to just poll snapshot().
    """
    aggregator = TradeAggregator(
        window_seconds=cfg.AGG_WINDOW_SECONDS,
        whale_usd_threshold=cfg.WHALE_TRADE_USD_THRESHOLD,
        max_whale_events=cfg.WHALE_LIST_MAXLEN,
        max_liquidation_events=cfg.LIQUIDATION_LIST_MAXLEN,
        on_whale_trade=on_whale_trade,
        on_liquidation=on_liquidation,
    )
    stop_event = threading.Event()
    holders = []

    def spawn(target, *args):
        holder = {}
        holders.append(holder)
        t = threading.Thread(target=target, args=(*args, holder), daemon=True)
        t.start()

    spawn(_coinbase_runner, aggregator, cfg.PRODUCT_ID, stop_event)

    binance_symbol = to_binance_symbol(cfg.PRODUCT_ID)
    if binance_symbol:
        spawn(_binance_us_trade_runner, aggregator, binance_symbol, stop_event)
        spawn(_binance_liquidation_runner, aggregator, binance_symbol, stop_event)

    bybit_symbol = to_bybit_symbol(cfg.PRODUCT_ID)
    if bybit_symbol:
        spawn(_bybit_trade_runner, aggregator, bybit_symbol, stop_event)
        spawn(_bybit_liquidation_runner, aggregator, bybit_symbol, stop_event)

    return FeedHandle(aggregator, stop_event, holders)


if __name__ == "__main__":
    import config as cfg

    print(f"Connecting to live feeds for {cfg.PRODUCT_ID} -- watching for 20 seconds...\n")
    handle = start_feeds(cfg)
    try:
        for _ in range(20):
            time.sleep(1)
            snap = handle.snapshot()
            print(
                f"status={snap['status']}  "
                f"buy=${snap['buy_notional']:,.0f}  sell=${snap['sell_notional']:,.0f}  "
                f"delta={snap['delta_ratio']:+.2f}  "
                f"whales={len(snap['whale_events'])}  liquidations={len(snap['liquidation_events'])}"
            )
    finally:
        handle.stop()
    print("\nDone. If any exchange shows an 'error' or never reached 'connected', "
          "check that domain isn't blocked by your network/firewall.")
