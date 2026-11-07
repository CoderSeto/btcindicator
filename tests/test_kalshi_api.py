"""
Tests for kalshi_api.py.

Schema fixtures are VERBATIM responses from Kalshi's public, unauthenticated
market-data API, captured 2026-11-07 from
    GET https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXBTC15M&status=open
    GET https://api.elections.kalshi.com/trade-api/v2/markets/KXBTC15M-26NOV071245-45
(tests/fixtures/kalshi_open_markets.json, kalshi_market_finalized.json), so
field names and value formats are the real ones, not guessed. Re-capture them
with those two requests if Kalshi changes the schema. Things the real payloads
show that the code depends on:
  - prices are *_dollars strings ("0.3200"), strikes are plain floats
  - an open market reports status "active" (the `status=open` query filter
    is not the same word as the status field)
  - `result` is "" until settlement, then "yes"/"no" with status "finalized"
  - a finalized market's quotes collapse to bid 0.0000 / ask 1.0000
The network is mocked; nothing here touches the internet.
"""
import json
import pathlib
from datetime import datetime, timezone

import pytest
import requests

import kalshi
import kalshi_api
import kalshi_tracking

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


OPEN = load("kalshi_open_markets.json")            # {"cursor": "", "markets": [ {...} ]}
FINAL = load("kalshi_market_finalized.json")       # {"market": {...}}
OPEN_MARKET = OPEN["markets"][0]
FINAL_MARKET = FINAL["market"]

# 7.5 minutes before the captured open market's 18:15:00Z close.
NOW = datetime(2026, 11, 7, 18, 7, 30, tzinfo=timezone.utc)


class FakeResp:
    def __init__(self, payload=None, status=200, bad_json=False):
        self._payload, self.status_code, self._bad_json = payload, status, bad_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        if self._bad_json:
            raise requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Replaces requests.get; records calls; the test sets `http.respond`."""
    class Http:
        calls = []
        respond = None
    h = Http()
    h.calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        h.calls.append({"url": url, "params": params, "timeout": timeout})
        r = h.respond
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(kalshi_api.requests, "get", fake_get)
    return h


# ---------- the captured payloads are what the code expects ----------

def test_fixtures_look_like_the_series_we_track():
    assert kalshi_api.SERIES_TICKER == "KXBTC15M" and kalshi_api.SERIES_TICKER != "KXBTCD"
    assert OPEN_MARKET["event_ticker"].startswith(kalshi_api.SERIES_TICKER)
    assert OPEN_MARKET["ticker"].startswith(OPEN_MARKET["event_ticker"])
    assert OPEN_MARKET["status"] == "active" and OPEN_MARKET["result"] == ""
    assert FINAL_MARKET["status"] == "finalized" and FINAL_MARKET["result"] in ("yes", "no")
    assert isinstance(OPEN_MARKET["floor_strike"], float)
    assert isinstance(OPEN_MARKET["yes_bid_dollars"], str)


def test_base_url_shape():
    assert kalshi_api.BASE_URL.startswith("https://") and kalshi_api.BASE_URL.endswith("/trade-api/v2")


# ---------- parse_market on real payloads ----------

def test_parse_open_market():
    out = kalshi_api.parse_market(OPEN_MARKET, now=NOW)
    assert out["available"] is True
    assert out["ticker"] == OPEN_MARKET["ticker"]
    assert out["strike"] == OPEN_MARKET["floor_strike"] == 75743.66
    assert out["close_time"] == datetime(2026, 11, 7, 18, 15, tzinfo=timezone.utc)
    assert out["close_time"].tzinfo is not None
    assert out["minutes_remaining"] == pytest.approx(7.5)
    assert out["yes_bid"] == float(OPEN_MARKET["yes_bid_dollars"])
    assert out["yes_ask"] == float(OPEN_MARKET["yes_ask_dollars"])
    assert out["last_price"] == float(OPEN_MARKET["last_price_dollars"])
    assert 0.0 <= out["yes_bid"] <= out["yes_ask"] <= 1.0
    assert out["status"] == "active"


def test_parse_finalized_market_zero_quote_stays_zero_and_time_clamps():
    out = kalshi_api.parse_market(FINAL_MARKET, now=NOW)           # NOW is after its 17:45Z close
    assert out["available"] is True
    assert out["minutes_remaining"] == 0.0                         # clamped, never negative
    assert out["yes_bid"] == 0.0 and out["yes_bid"] is not None    # a real 0.0000 is not "missing"
    assert out["yes_ask"] == 1.0
    assert out["last_price"] == pytest.approx(0.999)
    assert out["status"] == "finalized"


def test_parse_market_does_not_mutate_input():
    snap = json.dumps(OPEN_MARKET, sort_keys=True)
    kalshi_api.parse_market(OPEN_MARKET, now=NOW)
    assert json.dumps(OPEN_MARKET, sort_keys=True) == snap


def test_parse_market_defaults_now_to_the_real_clock():
    out = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2999-01-01T00:00:00Z"})
    assert out["minutes_remaining"] > 1_000_000


# ---------- parse_market edge cases ----------

@pytest.mark.parametrize("patch,why", [
    ({"floor_strike": None}, "strike"),
    ({"close_time": None}, "close"),
    ({"close_time": ""}, "close"),
])
def test_parse_market_missing_required_fields(patch, why):
    out = kalshi_api.parse_market({**OPEN_MARKET, **patch}, now=NOW)
    assert out["available"] is False and "missing strike or close time" in out["reason"]


def test_parse_market_absent_keys_and_empty_record():
    no_strike = {k: v for k, v in OPEN_MARKET.items() if k != "floor_strike"}
    assert kalshi_api.parse_market(no_strike, now=NOW)["available"] is False
    assert kalshi_api.parse_market({}, now=NOW)["available"] is False


def test_parse_market_unparseable_close_time():
    out = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "soon"}, now=NOW)
    assert out["available"] is False and "'soon'" in out["reason"]


def test_parse_market_accepts_z_and_explicit_offset_equally():
    a = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2026-11-07T18:15:00Z"}, now=NOW)
    b = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2026-11-07T18:15:00+00:00"}, now=NOW)
    c = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2026-11-07T13:15:00-05:00"}, now=NOW)
    assert a["close_time"] == b["close_time"] == c["close_time"]
    assert a["minutes_remaining"] == b["minutes_remaining"] == c["minutes_remaining"] == pytest.approx(7.5)


def test_parse_market_fractional_seconds_in_close_time():
    out = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2026-11-07T18:15:00.500000Z"}, now=NOW)
    assert out["minutes_remaining"] == pytest.approx(7.5 + 0.5 / 60)


def test_parse_market_strike_may_arrive_as_a_string():
    assert kalshi_api.parse_market({**OPEN_MARKET, "floor_strike": "75743.66"}, now=NOW)["strike"] == 75743.66


@pytest.mark.parametrize("raw,expected", [
    ("0.3200", 0.32), (0.32, 0.32), (1, 1.0), ("0.0000", 0.0), ("1", 1.0),
    (None, None), ("", None), ("n/a", None), ([], None), ({}, None),
])
def test_to_float(raw, expected):
    assert kalshi_api._to_float(raw) == expected


def test_parse_market_missing_or_bad_quotes_become_none_not_zero():
    rec = {k: v for k, v in OPEN_MARKET.items() if k not in ("yes_bid_dollars", "last_price_dollars")}
    rec["yes_ask_dollars"] = "garbage"
    out = kalshi_api.parse_market(rec, now=NOW)
    assert out["available"] is True
    assert out["yes_bid"] is None and out["yes_ask"] is None and out["last_price"] is None


# ---------- get_open_markets / get_market ----------

def test_get_open_markets_request_and_result(http):
    http.respond = FakeResp(OPEN)
    out = kalshi_api.get_open_markets()
    assert out == OPEN["markets"]
    (call,) = http.calls
    assert call["url"] == f"{kalshi_api.BASE_URL}/markets"
    assert call["params"] == {"series_ticker": "KXBTC15M", "status": "open"}
    assert call["timeout"]                                          # never hangs forever


def test_get_open_markets_other_series_and_missing_key(http):
    http.respond = FakeResp({"cursor": ""})
    assert kalshi_api.get_open_markets("KXOTHER") == []
    assert http.calls[0]["params"]["series_ticker"] == "KXOTHER"


def test_get_open_markets_propagates_http_errors(http):
    http.respond = FakeResp({"error": {"code": "too_many_requests"}}, status=429)
    with pytest.raises(requests.exceptions.HTTPError):
        kalshi_api.get_open_markets()


def test_get_market_request_and_unwrap(http):
    http.respond = FakeResp(FINAL)
    out = kalshi_api.get_market("KXBTC15M-26NOV071245-45")
    assert out == FINAL_MARKET and out["result"] == "yes"
    (call,) = http.calls
    assert call["url"] == f"{kalshi_api.BASE_URL}/markets/KXBTC15M-26NOV071245-45"
    assert call["timeout"]


def test_get_market_missing_key_and_errors(http):
    http.respond = FakeResp({})
    assert kalshi_api.get_market("X") == {}
    http.respond = FakeResp(None, status=404)
    with pytest.raises(requests.exceptions.HTTPError):
        kalshi_api.get_market("X")


# ---------- current_btc15m_market ----------

def test_current_market_happy_path(http):
    http.respond = FakeResp(OPEN)
    out = kalshi_api.current_btc15m_market()
    assert out["available"] is True and out["ticker"] == OPEN_MARKET["ticker"]
    assert out["strike"] == 75743.66


def test_current_market_uses_the_first_listed_market(http):
    second = {**OPEN_MARKET, "ticker": "KXBTC15M-26NOV071330-30", "floor_strike": 1.0}
    http.respond = FakeResp({"markets": [OPEN_MARKET, second]})
    assert kalshi_api.current_btc15m_market()["ticker"] == OPEN_MARKET["ticker"]


def test_current_market_none_open(http):
    http.respond = FakeResp({"markets": [], "cursor": ""})
    out = kalshi_api.current_btc15m_market()
    assert out["available"] is False and "No open KXBTC15M market" in out["reason"]


@pytest.mark.parametrize("failure", [
    requests.exceptions.ConnectionError("dns"),
    requests.exceptions.Timeout("slow"),
    requests.exceptions.SSLError("tls"),
    FakeResp({"error": "rate"}, status=429),
    FakeResp({"error": "oops"}, status=500),
    FakeResp(bad_json=True),                                       # an HTML error page, not JSON
])
def test_current_market_network_failures_become_unavailable(http, failure):
    http.respond = failure
    out = kalshi_api.current_btc15m_market()
    assert out["available"] is False and "Couldn't reach Kalshi's API" in out["reason"]


def test_current_market_with_incomplete_record(http):
    http.respond = FakeResp({"markets": [{"ticker": "KXBTC15M-X"}]})
    assert kalshi_api.current_btc15m_market()["available"] is False


# ---------- hand-offs to the rest of the bot, using the real payloads ----------

def test_parsed_market_feeds_the_fair_value_math():
    m = kalshi_api.parse_market(OPEN_MARKET, now=NOW)
    p = kalshi.crossing_probability(m["strike"] + 10, m["strike"], m["minutes_remaining"], 0.6)
    assert 0.5 < p < 1.0                                           # price a bit above strike -> favours YES


def test_outcome_resolution_reads_the_real_finalized_payload(http):
    http.respond = FakeResp(FINAL)
    cache = kalshi_tracking.resolve_outcomes([FINAL_MARKET["ticker"]], {})
    assert cache == {FINAL_MARKET["ticker"]: FINAL_MARKET["result"]}
    # ...and the still-open market (result == "") is left unresolved for next time.
    http.respond = FakeResp({"market": OPEN_MARKET})
    assert kalshi_tracking.resolve_outcomes([OPEN_MARKET["ticker"]], {}) == {}


def test_a_close_time_without_a_timezone_is_read_as_utc_not_a_crash():
    # Real payloads end in "Z" (both fixtures, and the live API on 2026-11-07), but a naive
    # timestamp used to raise TypeError (naive minus aware) straight out of
    # current_btc15m_market(), which only catches network errors -- an exception on the page.
    parsed = kalshi_api.parse_market({**OPEN_MARKET, "close_time": "2026-11-07T18:15:00"},
                                     now=datetime(2026, 11, 7, 18, 7, 30, tzinfo=timezone.utc))
    assert parsed["available"] is True
    assert parsed["minutes_remaining"] == pytest.approx(7.5)
    assert parsed["close_time"].tzinfo is not None


# ---------- the shared poll cache (REVIEW_FINDINGS D10) ----------
#
# The dashboard polled Kalshi on every rerun of every tab (12 requests a minute per tab at a 5s
# refresh), and Kalshi has answered 429 before. cached_btc15m_market() fetches at most once per
# max_age across the whole process, but recomputes minutes_remaining from the clock on every call,
# so the countdown never freezes on a cached value.

@pytest.fixture
def poll(monkeypatch):
    from datetime import timedelta
    state = {"calls": 0, "result": kalshi_api.parse_market(OPEN_MARKET, now=NOW), "now": NOW}

    def fetch():
        state["calls"] += 1
        return state["result"]
    monkeypatch.setattr(kalshi_api, "current_btc15m_market", fetch)
    monkeypatch.setattr(kalshi_api, "_now", lambda: state["now"])
    monkeypatch.setattr(kalshi_api, "_poll_cache", {"at": None, "result": None})
    state["advance"] = lambda s: state.__setitem__("now", state["now"] + timedelta(seconds=s))
    return state


def test_poll_is_shared_within_max_age_but_the_countdown_keeps_moving(poll):
    assert kalshi_api.cached_btc15m_market(10)["minutes_remaining"] == pytest.approx(7.5)
    poll["advance"](6)
    again = kalshi_api.cached_btc15m_market(10)
    assert poll["calls"] == 1                                              # served from the cache
    assert again["minutes_remaining"] == pytest.approx(7.4)                # but 6s later on the clock
    poll["advance"](5)                                                     # 11s after the fetch
    kalshi_api.cached_btc15m_market(10)
    assert poll["calls"] == 2


def test_poll_refetches_once_the_cached_window_has_closed(poll):
    kalshi_api.cached_btc15m_market(600)
    poll["advance"](7.5 * 60)                                              # exactly at close
    kalshi_api.cached_btc15m_market(600)
    assert poll["calls"] == 2                                              # the next window, not a stale 0:00


def test_poll_caches_failures_too_so_a_429_is_not_hammered(poll):
    poll["result"] = {"available": False, "reason": "Couldn't reach Kalshi's API: 429"}
    kalshi_api.cached_btc15m_market(10)
    poll["advance"](3)
    out = kalshi_api.cached_btc15m_market(10)
    assert poll["calls"] == 1 and out["available"] is False


def test_poll_max_age_zero_always_fetches_and_callers_cannot_mutate_the_cache(poll):
    first = kalshi_api.cached_btc15m_market(0)
    first["strike"] = -1.0
    second = kalshi_api.cached_btc15m_market(0)
    assert poll["calls"] == 2 and second["strike"] == 75_743.66
