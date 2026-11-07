"""
Tests for outlook.py -- the plain-language layer between the numbers and the
screen. The properties that matter most are the ones the project's design
principles promise: hedged wording, no blending of probabilities, the weighted
read never overriding the raw headline, and a resting wall never described as
a floor or ceiling.
"""
import pytest

import main
import outlook
import signals


def snap(book=None, **votes):
    return signals.SignalSnapshot(price=100.0, individual=votes, book=book or {})


NO_DIV = {"type": None, "detail": "No clear divergence in the recent window."}


# ---------- headline ----------

@pytest.mark.parametrize("votes,expected", [
    # strength >= total - 1 -> "Most"
    (dict(a="bullish", b="bullish", c="bullish"), "Most indicators agree on upside pressure (3/3)."),
    (dict(a="bullish", b="bullish", c="bearish"), "Most indicators agree on upside pressure (2/3)."),
    (dict(a="bearish", b="bearish", c="bearish", d="bearish", e="neutral"),
     "Most indicators agree on downside pressure (4/5)."),
    # more than half but more than one dissenter -> "A majority"
    (dict(a="bullish", b="bullish", c="bullish", d="bearish", e="neutral", f="neutral"),
     "A slim majority of indicators agree on upside pressure (3/6)."),
    (dict(a="bullish", b="bullish", c="bullish", d="bullish", e="bearish", f="neutral", g="neutral"),
     "A majority of indicators agree on upside pressure (4/7)."),
])
def test_headline_wording_and_counts(votes, expected):
    assert outlook.build_outlook(snap(**votes), NO_DIV)["headline"] == expected


@pytest.mark.parametrize("votes", [
    dict(),                                                    # nothing at all
    dict(a="neutral", b="neutral"),
    dict(a="bullish", b="bearish"),                            # tie
    dict(a="bullish", b="bullish", c="bearish", d="bearish"),
])
def test_split_or_empty_is_reported_as_no_lean(votes):
    assert outlook.build_outlook(snap(**votes), NO_DIV)["headline"] == \
        "Indicators are split -- no clear lean either way right now."


# ---------- detail / drivers ----------

def test_drivers_follow_priority_order_and_cap_at_three():
    s = snap(sma_cross="bullish", rsi="bullish", macd="bullish", volume="bullish", ema_cross="bearish")
    assert outlook._drivers(s) == ["rsi", "macd", "volume", "sma_cross"]       # priority, not dict, order
    detail = outlook.build_outlook(s, NO_DIV)["detail"]
    assert detail == (
        "Contributing: RSI is in oversold territory, which has often preceded a short-term bounce; "
        "MACD is above its signal line; "
        "a volume spike accompanied the last up move."
    )
    assert "moving average" not in detail                                       # 4th driver is cut off


def test_driver_phrases_follow_the_composite_direction():
    s = snap(rsi="bearish", bollinger="bearish", macd="bullish")
    detail = outlook.build_outlook(s, NO_DIV)["detail"]
    assert "overbought" in detail and "upper Bollinger Band" in detail
    assert "MACD" not in detail                                                 # dissenters are not drivers


def test_no_drivers_when_split_and_unknown_names_are_ignored():
    assert outlook._drivers(snap(rsi="bullish", macd="bearish")) == []
    s = snap(mystery_signal="bullish")
    assert outlook._drivers(s) == []
    assert outlook.build_outlook(s, NO_DIV)["detail"] == ""


def test_every_signal_the_bot_can_emit_has_a_priority_slot_and_phrases():
    # A signal missing from _PRIORITY would silently never be mentioned as a driver.
    assert set(main.ALL_SIGNAL_NAMES) == set(outlook._PRIORITY)
    for name in outlook._PRIORITY:
        assert set(outlook._DRIVER_PHRASES[name]) == {"bullish", "bearish"}
    assert not (set(outlook._DRIVER_PHRASES) - set(outlook._PRIORITY))


# ---------- divergence: caution vs corroboration ----------

def test_counter_divergence_raises_a_caution_and_leaves_detail_alone():
    s = snap(rsi="bullish", macd="bullish", sma_cross="bearish")
    div = {"type": "bearish", "detail": "Price made a higher high."}
    out = outlook.build_outlook(s, div)
    assert out["caution"] == "Reversal watch (BEARISH): Price made a higher high."
    assert "divergence" not in out["detail"]


def test_matching_divergence_is_appended_to_detail_without_a_caution():
    s = snap(rsi="bullish", macd="bullish", sma_cross="bearish")
    div = {"type": "bullish", "detail": "Price made a lower low."}
    out = outlook.build_outlook(s, div)
    assert out["caution"] is None
    assert out["detail"].endswith("This also lines up with a recent bullish RSI/price divergence: Price made a lower low.")
    assert out["detail"].startswith("Contributing:")


def test_matching_divergence_with_no_drivers_has_no_leading_space():
    out = outlook.build_outlook(snap(mystery="bullish"), {"type": "bullish", "detail": "x"})
    assert out["detail"] == "This also lines up with a recent bullish RSI/price divergence: x"


def test_divergence_while_split_is_a_caution():
    out = outlook.build_outlook(snap(a="bullish", b="bearish"), {"type": "bullish", "detail": "d"})
    assert out["caution"] == "Reversal watch (BULLISH): d"


@pytest.mark.parametrize("div", [NO_DIV, {}, {"type": None}])
def test_no_divergence_no_caution(div):
    assert outlook.build_outlook(snap(rsi="bullish"), div)["caution"] is None


# ---------- backtest confidence lines ----------

def test_confidence_lines_format_and_small_sample_caveat():
    conf = {
        "available": True, "lookahead_candles": 5,
        "signal": {"direction": "bullish", "rate": 0.7549, "n": 40, "low_sample": False},
        "divergence": {"direction": "bearish", "rate": 0.3, "n": 10, "low_sample": True},
    }
    lines = outlook._confidence_lines(conf)
    assert lines == [
        "Historically, this signal pattern has been followed by a bullish move within 5 candles 75% of the time (n=40).",
        "Bearish RSI/price divergence patterns like this have preceded a bearish move within 5 candles 30% of the time "
        "(n=10) (small sample -- treat with caution).",
    ]
    conf["signal"]["low_sample"] = True
    assert "(small sample -- treat with caution)" in outlook._confidence_lines(conf)[0]


@pytest.mark.parametrize("conf", [None, {}, {"available": False}])
def test_confidence_lines_empty_without_a_backtest(conf):
    assert outlook._confidence_lines(conf) == []


def test_confidence_lines_available_but_nothing_matched():
    assert outlook._confidence_lines({"available": True, "lookahead_candles": 5}) == []


# ---------- weighted line ----------

def w(direction, pct):
    return {"available": True, "direction": direction, "weighted_pct": pct}


def test_weighted_line_agrees_disagrees_or_tosses_up():
    bull = snap(rsi="bullish", macd="bullish", sma_cross="bearish")
    assert outlook._weighted_line(w("bullish", 0.683), bull) == (
        "Weighted by each indicator's historical accuracy, conviction is 68% bullish -- agrees with the raw count above.")
    assert outlook._weighted_line(w("bearish", 0.58), bull) == (
        "Weighted by each indicator's historical accuracy, conviction actually leans 58% bearish -- "
        "the opposite of the raw count above. Worth a second look.")
    assert "closer to a toss-up" in outlook._weighted_line(w("neutral", 0.5), bull)


@pytest.mark.parametrize("weighted", [None, {}, {"available": False}])
def test_weighted_line_empty_when_unavailable(weighted):
    assert outlook._weighted_line(weighted, snap(rsi="bullish")) == ""


def test_weighted_read_never_changes_the_headline_or_anything_else():
    s = snap(rsi="bullish", macd="bullish", sma_cross="bearish")
    base = outlook.build_outlook(s, NO_DIV)
    with_w = outlook.build_outlook(s, NO_DIV, weighted=w("bearish", 0.9))
    for key in ("headline", "detail", "caution", "book_note", "confidence_lines"):
        assert with_w[key] == base[key]
    assert base["weighted_line"] == "" and with_w["weighted_line"] != ""


# ---------- order-book note ----------

def wall(side="bid", notional=500_000, bps=12, age=45, venue="coinbase"):
    w = {"side": side, "notional": notional, "distance_bps": bps, "age_seconds": age}
    if venue:
        w["venue"] = venue
    return w


def book(*walls, available=True):
    return {"available": available, "held_walls": list(walls)}


@pytest.mark.parametrize("b", [None, {}, {"available": False, "held_walls": [wall()]},
                               {"available": True}, {"available": True, "held_walls": []}])
def test_book_note_empty_without_a_held_wall(b):
    assert outlook.build_outlook(snap(rsi="bullish", book=b), NO_DIV)["book_note"] == ""


def test_book_note_describes_the_largest_held_wall():
    s = snap(rsi="bullish", book=book(wall(notional=300_000, side="ask"), wall(notional=1_250_000, bps=7.6, age=61.4)))
    note = outlook.build_outlook(s, NO_DIV)["book_note"]
    assert note.startswith("Order book: $1,250,000 of resting size sits 8bps below mid on coinbase and has been there 61s.")
    assert "continue down" in note


def test_book_note_ask_wall_is_above_and_blocks_upside():
    note = outlook._book_note(snap(book=book(wall(side="ask", venue="bybit"))))
    assert "above mid on bybit" in note and "continue up" in note


def test_book_note_missing_venue_falls_back():
    assert "on an exchange" in outlook._book_note(snap(book=book(wall(venue=None))))


def test_book_note_never_calls_a_wall_a_floor_or_ceiling():
    note = outlook._book_note(snap(book=book(wall())))
    assert "not a floor or a ceiling" in note and "can be pulled" in note
    for word in ("support", "resistance", "guarantee"):
        assert word not in note.lower()


# ---------- shape ----------

def test_build_outlook_always_returns_every_key():
    for s in (snap(), snap(rsi="bullish"), snap(rsi="bullish", macd="bearish")):
        out = outlook.build_outlook(s, NO_DIV)
        assert set(out) == {"headline", "detail", "caution", "book_note", "confidence_lines", "weighted_line"}
        assert isinstance(out["confidence_lines"], list)


def test_macd_phrase_describes_a_state_not_a_fresh_cross():
    # signals.py votes MACD by which side of its signal line it is on (the crossover branch
    # returns the same answer as the state check), so a MACD that crossed 100 candles ago still
    # votes. The narrative must not tell the reader it "just crossed".
    for direction in ("bullish", "bearish"):
        detail = outlook.build_outlook(snap(macd=direction), NO_DIV)["detail"]
        assert "just crossed" not in detail
        assert "MACD is " + ("above" if direction == "bullish" else "below") + " its signal line" in detail


def test_confidence_lines_show_the_base_rate_and_the_effective_sample():
    # A hit rate means little without what a no-skill call would have scored in the same window,
    # and n overstates the evidence when outcome windows overlap (REVIEW_FINDINGS D2/D3).
    conf = {
        "available": True, "lookahead_candles": 5,
        "signal": {"direction": "bullish", "rate": 0.53, "n": 1980, "n_eff": 571.6,
                   "base_rate": 0.512, "low_sample": False},
    }
    assert outlook._confidence_lines(conf) == [
        "Historically, this signal pattern has been followed by a bullish move within 5 candles 53% "
        "of the time, against a 51% base rate for any window (n=1,980, about 572 independent)."
    ]


def test_weighted_line_says_when_no_weight_carries_a_measured_edge():
    s = snap(rsi="bullish", macd="bullish")
    weighted = {"available": True, "direction": "bullish", "weighted_pct": 1.0,
                "breakdown": {"rsi": {"vote": "bullish", "weight": 1.0},
                              "macd": {"vote": "bullish", "weight": 1.0}}}
    line = outlook._weighted_line(weighted, s)
    assert "no indicator voting now has a measured edge over the base rate" in line
    assert "%" not in line                                                  # no conviction figure to quote
