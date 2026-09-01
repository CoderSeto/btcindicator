"""
Turns the current signal snapshot (+ divergence read) into a short,
plain-language read on where things might be leaning.

This is commentary on the current pattern of indicators, not a forecast --
it describes what's showing up right now and what that combination has
tended to precede, with hedged language on purpose. It is not trading
advice and shouldn't be treated as one.
"""

# Ordered roughly by how much weight each indicator typically carries for
# spotting a near-term turn (vs. just confirming an established trend).
_PRIORITY = ["rsi", "bollinger", "agg_delta", "orderbook", "macd", "volume", "sma_cross", "ema_cross"]

_DRIVER_PHRASES = {
    "rsi": {
        "bullish": "RSI is in oversold territory, which has often preceded a short-term bounce",
        "bearish": "RSI is in overbought territory, which has often preceded a short-term pullback",
    },
    "bollinger": {
        "bullish": "price is pressing the lower Bollinger Band, suggesting it may be overextended to the downside",
        "bearish": "price is pressing the upper Bollinger Band, suggesting it may be overextended to the upside",
    },
    "agg_delta": {
        "bullish": "taker buying is outpacing selling across exchanges right now",
        "bearish": "taker selling is outpacing buying across exchanges right now",
    },
    "orderbook": {
        "bullish": "the order book is currently buy-heavy",
        "bearish": "the order book is currently sell-heavy",
    },
    "macd": {
        "bullish": "MACD just crossed above its signal line",
        "bearish": "MACD just crossed below its signal line",
    },
    "volume": {
        "bullish": "a volume spike accompanied the last up move",
        "bearish": "a volume spike accompanied the last down move",
    },
    "sma_cross": {
        "bullish": "the fast moving average is above the slow one",
        "bearish": "the fast moving average is below the slow one",
    },
    "ema_cross": {
        "bullish": "the fast EMA is above the slow EMA",
        "bearish": "the fast EMA is below the slow EMA",
    },
}


def _headline(snapshot) -> str:
    total = len(snapshot.individual)
    if snapshot.composite == "neutral":
        return "Indicators are split -- no clear lean either way right now."

    lean = "upside" if snapshot.composite == "bullish" else "downside"
    if snapshot.strength >= total - 1:
        conviction = "Most indicators agree"
    elif snapshot.strength > total / 2:
        conviction = "A majority of indicators agree"
    else:
        conviction = "A slim majority of indicators agree"
    return f"{conviction} on {lean} pressure ({snapshot.strength}/{total})."


def _drivers(snapshot) -> list:
    # "Neutral" isn't a lean -- there's no phrase for it and nothing to
    # highlight as "contributing" when the composite itself is split.
    if snapshot.composite == "neutral":
        return []
    return [name for name in _PRIORITY if snapshot.individual.get(name) == snapshot.composite]


def _confidence_lines(confidence: dict) -> list:
    """
    Turns a backtest.get_confidence() result into readable lines. Returns
    an empty list if there's no backtest yet -- callers should treat that
    as "no confidence data available" rather than an error.
    """
    if not confidence or not confidence.get("available"):
        return []

    lookahead = confidence.get("lookahead_candles")
    lines = []

    sig = confidence.get("signal")
    if sig:
        pct = f"{sig['rate'] * 100:.0f}%"
        caveat = " (small sample -- treat with caution)" if sig["low_sample"] else ""
        lines.append(
            f"Historically, this signal pattern has been followed by a {sig['direction']} move "
            f"within {lookahead} candles {pct} of the time (n={sig['n']}){caveat}."
        )

    div = confidence.get("divergence")
    if div:
        pct = f"{div['rate'] * 100:.0f}%"
        caveat = " (small sample -- treat with caution)" if div["low_sample"] else ""
        lines.append(
            f"{div['direction'].capitalize()} RSI/price divergence patterns like this have preceded a "
            f"{div['direction']} move within {lookahead} candles {pct} of the time (n={div['n']}){caveat}."
        )

    return lines


def _weighted_line(weighted: dict, snapshot) -> str:
    """
    Turns a weighting.weighted_conviction() result into one readable line.
    Returns "" if unavailable -- callers should treat that as "nothing to
    show" rather than an error.
    """
    if not weighted or not weighted.get("available"):
        return ""

    pct = f"{weighted['weighted_pct'] * 100:.0f}%"
    direction = weighted["direction"]

    if direction == snapshot.composite:
        return (
            f"Weighted by each indicator's historical accuracy, conviction is {pct} "
            f"{direction} -- agrees with the raw count above."
        )
    elif direction == "neutral":
        return (
            f"Weighted by each indicator's historical accuracy, this is closer to a "
            f"toss-up than the raw count above suggests."
        )
    else:
        return (
            f"Weighted by each indicator's historical accuracy, conviction actually leans "
            f"{pct} {direction} -- the opposite of the raw count above. Worth a second look."
        )


def build_outlook(snapshot, divergence: dict, confidence: dict = None, weighted: dict = None) -> dict:
    """
    Returns {"headline": str, "detail": str, "caution": str | None,
    "confidence_lines": list[str], "weighted_line": str}.
    `caution` is set when a reversal-type divergence pattern was found,
    especially when it runs counter to the current composite lean.
    `confidence_lines` holds backtested historical hit-rate readouts, if a
    matching backtest has been run (empty list otherwise).
    `weighted_line` holds a one-line read from weighting.weighted_conviction(),
    a second opinion that weighs each indicator by its own backtested
    accuracy rather than counting all of them equally (empty string if no
    backtest is available yet). It never changes `headline` or anything
    else here -- the raw composite still drives everything else unchanged.
    """
    headline = _headline(snapshot)

    top_drivers = _drivers(snapshot)[:3]
    phrases = [
        _DRIVER_PHRASES[d][snapshot.composite]
        for d in top_drivers
        if snapshot.composite in _DRIVER_PHRASES.get(d, {})
    ]
    detail = ("Contributing: " + "; ".join(phrases) + ".") if phrases else ""

    caution = None
    div_type = divergence.get("type")
    if div_type and div_type != snapshot.composite:
        caution = f"Reversal watch ({div_type.upper()}): {divergence['detail']}"
    elif div_type and div_type == snapshot.composite:
        detail = (detail + " " if detail else "") + (
            f"This also lines up with a recent {div_type} RSI/price divergence: {divergence['detail']}"
        )

    return {
        "headline": headline,
        "detail": detail,
        "caution": caution,
        "confidence_lines": _confidence_lines(confidence),
        "weighted_line": _weighted_line(weighted, snapshot),
    }
