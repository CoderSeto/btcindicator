"""
Turns raw indicator values into discrete bullish/bearish/neutral signals,
then combines them into a composite read on market movement.
"""
from dataclasses import dataclass, field


@dataclass
class SignalSnapshot:
    price: float
    individual: dict = field(default_factory=dict)  # name -> "bullish"/"bearish"/"neutral"
    imbalance: float = 0.0

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


def evaluate(df, imbalance: float, cfg, agg_delta: float = None) -> SignalSnapshot:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signals = _evaluate_price_signals(last, prev, cfg)

    # Order book imbalance
    if imbalance > cfg.IMBALANCE_THRESHOLD:
        signals["orderbook"] = "bullish"
    elif imbalance < -cfg.IMBALANCE_THRESHOLD:
        signals["orderbook"] = "bearish"
    else:
        signals["orderbook"] = "neutral"

    # Multi-exchange aggregated taker buy/sell delta (only present when
    # exchange_feeds is enabled and has collected at least some trades)
    if agg_delta is not None:
        if agg_delta > cfg.AGG_DELTA_THRESHOLD:
            signals["agg_delta"] = "bullish"
        elif agg_delta < -cfg.AGG_DELTA_THRESHOLD:
            signals["agg_delta"] = "bearish"
        else:
            signals["agg_delta"] = "neutral"

    return SignalSnapshot(price=float(last["close"]), individual=signals, imbalance=imbalance)


def evaluate_price_only(df, cfg) -> SignalSnapshot:
    """
    Same as evaluate(), but without the order-book signal. Used for
    backtesting (no historical order book available) and for looking up
    backtest confidence against the live state on a matching scale.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    signals = _evaluate_price_signals(last, prev, cfg)
    return SignalSnapshot(price=float(last["close"]), individual=signals, imbalance=0.0)
