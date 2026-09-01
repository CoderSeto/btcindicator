"""
Turns raw indicator values into discrete bullish/bearish/neutral signals,
then combines them into a composite read on market movement.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalSnapshot:
    price: float
    individual: dict = field(default_factory=dict)  # name -> "bullish"/"bearish"/"neutral"
    # Legacy single-snapshot Coinbase REST book imbalance. None means it
    # was never computed (the live depth feeds were up, so the extra REST
    # request was skipped) -- which is a different thing from 0.0, a real
    # reading of a balanced book. The loggers write a blank cell for None.
    imbalance: Optional[float] = None
    book: dict = field(default_factory=dict)  # summarize_book() result, {} when unavailable

    @property
    def bullish_count(self) -> int:
        return sum(1 for v in self.individual.values() if v == "bullish")

    @property
    def bearish_count(self) -> int:
        return sum(1 for v in self.individual.values() if v == "bearish")

    @property
    def composite(self) -> str:
        if self.bullish_count == self.bearish_count:
            return "neutral"
        return "bullish" if self.bullish_count > self.bearish_count else "bearish"

    @property
    def strength(self) -> int:
        """How many individual signals agree with the composite direction."""
        return max(self.bullish_count, self.bearish_count)


def _evaluate_price_signals(last, prev, cfg) -> dict:
    """
    The 6 signals derivable purely from candle data (no order book).
    Split out so the same logic can be reused for backtesting, where only
    historical candles are available -- Coinbase doesn't expose historical
    order-book snapshots.
    """
    signals = {}

    # SMA crossover
    if last["sma_9"] > last["sma_21"] and prev["sma_9"] <= prev["sma_21"]:
        signals["sma_cross"] = "bullish"
    elif last["sma_9"] < last["sma_21"] and prev["sma_9"] >= prev["sma_21"]:
        signals["sma_cross"] = "bearish"
    else:
        signals["sma_cross"] = "bullish" if last["sma_9"] > last["sma_21"] else "bearish"

    # EMA crossover
    if last["ema_12"] > last["ema_26"] and prev["ema_12"] <= prev["ema_26"]:
        signals["ema_cross"] = "bullish"
    elif last["ema_12"] < last["ema_26"] and prev["ema_12"] >= prev["ema_26"]:
        signals["ema_cross"] = "bearish"
    else:
        signals["ema_cross"] = "bullish" if last["ema_12"] > last["ema_26"] else "bearish"

    # RSI
    if last["rsi"] < cfg.RSI_OVERSOLD:
        signals["rsi"] = "bullish"  # oversold -> potential bounce
    elif last["rsi"] > cfg.RSI_OVERBOUGHT:
        signals["rsi"] = "bearish"  # overbought -> potential pullback
    else:
        signals["rsi"] = "neutral"

    # MACD
    if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
        signals["macd"] = "bullish"
    elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
        signals["macd"] = "bearish"
    else:
        signals["macd"] = "bullish" if last["macd"] > last["macd_signal"] else "bearish"

    # Bollinger Bands
    if last["close"] >= last["bb_upper"]:
        signals["bollinger"] = "bearish"  # stretched to the upside
    elif last["close"] <= last["bb_lower"]:
        signals["bollinger"] = "bullish"  # stretched to the downside
    else:
        signals["bollinger"] = "neutral"

    # Volume spike (direction depends on which way price moved on the spike)
    if last["volume_spike"]:
        signals["volume"] = "bullish" if last["close"] >= last["open"] else "bearish"
    else:
        signals["volume"] = "neutral"

    return signals


def summarize_book(book_snapshot: dict) -> dict:
    """
    Flattens an `orderbook_feeds.BookFeedHandle.snapshot()` result into the
    handful of scalars the signal, the logs and the UI all want, so none of
    them has to reach three levels into a nested dict and guess at defaults.

    Returns {"available": False} when the depth feeds aren't up or haven't
    produced a usable book yet -- callers must treat that as "no data",
    which is a different thing from a balanced book.
    """
    if not book_snapshot or not book_snapshot.get("available"):
        return {"available": False}

    agg = book_snapshot["aggregate"]
    persistence = book_snapshot["persistence"]
    flow = book_snapshot["flow"]

    return {
        "available": True,
        "imbalance": agg["imbalance"],
        "imbalance_near": agg["imbalance_near"],
        "imbalance_far": agg["imbalance_far"],
        "microprice_bps": agg["microprice_bps"],
        "agreement": agg["agreement"],
        "bid_notional": agg["bid_notional"],
        "ask_notional": agg["ask_notional"],
        "venue_count": agg["venue_count"],
        "venues": agg["venues"],
        "excluded": agg.get("excluded", {}),
        "persistence_seconds": persistence["seconds"],
        "persistence_direction": persistence["direction"],
        "flow_available": flow.get("available", False),
        "flow_value": flow.get("value"),
        "flow_reason": flow.get("reason"),
        "flow_window_seconds": flow.get("window_seconds"),
        "bid_change_pct": flow.get("bid_change_pct"),
        "ask_change_pct": flow.get("ask_change_pct"),
        "walls": book_snapshot.get("walls", []),
        "held_walls": [w for w in book_snapshot.get("walls", []) if w.get("held")],
        "samples": book_snapshot.get("samples", 0),
        "history_seconds": book_snapshot.get("history_seconds", 0.0),
        "status": book_snapshot.get("status", {}),
    }


def _orderbook_signal(book: dict, imbalance: Optional[float], cfg) -> str:
    """
    The order-book vote.

    With the live depth feeds up, a lean only becomes a vote once it has
    held the same side past BOOK_IMBALANCE_THRESHOLD continuously for
    BOOK_PERSISTENCE_SECONDS. That gate is the whole point of maintaining
    a live book: resting size is placed and pulled constantly, so any
    single reading of a lopsided book is mostly noise, and "it has looked
    like this for the last half a minute" is a far stronger statement than
    "it looks like this in the frame I happened to sample".

    Without the feeds, this falls back to the original behavior -- one
    Coinbase REST snapshot against IMBALANCE_THRESHOLD, with no time
    dimension available to gate on.
    """
    if book.get("available"):
        direction = book["persistence_direction"]
        if direction and book["persistence_seconds"] >= cfg.BOOK_PERSISTENCE_SECONDS:
            return direction
        return "neutral"

    reading = imbalance or 0.0
    if reading > cfg.IMBALANCE_THRESHOLD:
        return "bullish"
    if reading < -cfg.IMBALANCE_THRESHOLD:
        return "bearish"
    return "neutral"


def _book_flow_signal(book: dict, cfg) -> str:
    """
    Resting liquidity being added to one side relative to the other:
    bids stacking and/or asks being pulled reads bullish, and vice versa.

    Distinct from the imbalance vote above -- imbalance is about the shape
    of the book right now, flow is about which way that shape is moving,
    and they routinely disagree (a heavily bid book whose bids are being
    pulled is a different situation from one that's still filling in).
    """
    if not book.get("available") or not book.get("flow_available"):
        return "neutral"
    value = book["flow_value"]
    if value > cfg.BOOK_FLOW_THRESHOLD:
        return "bullish"
    if value < -cfg.BOOK_FLOW_THRESHOLD:
        return "bearish"
    return "neutral"


def evaluate(df, imbalance: Optional[float], cfg, agg_delta: float = None,
             book_snapshot: dict = None) -> SignalSnapshot:
    """
    The live path: candle signals plus the order-book and cross-exchange
    trade-flow signals that only exist in real time.

    `imbalance` is the legacy single-snapshot Coinbase REST reading, still
    used as the fallback when `book_snapshot` (from
    orderbook_feeds.BookFeedHandle.snapshot()) is absent or not yet warmed
    up. Pass None when it wasn't computed at all.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signals = _evaluate_price_signals(last, prev, cfg)
    book = summarize_book(book_snapshot)

    signals["orderbook"] = _orderbook_signal(book, imbalance, cfg)

    # Only voted on when the feeds are actually up and have enough history
    # to measure a change -- an absent signal is left out of the snapshot
    # entirely rather than logged as a neutral it didn't earn (the loggers
    # write a blank cell for it; see ALL_SIGNAL_NAMES in main.py).
    if cfg.ENABLE_BOOK_FLOW_SIGNAL and book.get("available") and book.get("flow_available"):
        signals["book_flow"] = _book_flow_signal(book, cfg)

    # Multi-exchange aggregated taker buy/sell delta (only present when
    # exchange_feeds is enabled and has collected at least some trades)
    if agg_delta is not None:
        if agg_delta > cfg.AGG_DELTA_THRESHOLD:
            signals["agg_delta"] = "bullish"
        elif agg_delta < -cfg.AGG_DELTA_THRESHOLD:
            signals["agg_delta"] = "bearish"
        else:
            signals["agg_delta"] = "neutral"

    return SignalSnapshot(price=float(last["close"]), individual=signals,
                          imbalance=imbalance, book=book)


def evaluate_price_only(df, cfg) -> SignalSnapshot:
    """
    Same as evaluate(), but without the order-book signal. Used for
    backtesting (no historical order book available) and for looking up
    backtest confidence against the live state on a matching scale.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    signals = _evaluate_price_signals(last, prev, cfg)
    return SignalSnapshot(price=float(last["close"]), individual=signals, imbalance=None)
