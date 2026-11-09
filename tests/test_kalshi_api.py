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


# ---------- the shared, adaptive quote poll (kalshi_api.QuotePoller) ----------
#
# One poll for the whole process; quote from the orderbook (the market record's quote lagged the
# live book in the 2026-11-08 probe); backoff on 429 / 5xx / timeout, Retry-After honoured; the
# interval shrinks 20% after 30 straight successes. Clock and network are faked.

def _book(yes_bid, yes_ask):
    return {"orderbook_fp": {"yes_dollars": [[f"{yes_bid:.4f}", "10"]], "no_dollars": [[f"{1 - yes_ask:.4f}", "10"]]}}


def _http_error(code, retry_after=None):
    resp = requests.Response()
    resp.status_code = code
    if retry_after is not None:
        resp.headers["Retry-After"] = str(retry_after)
    return requests.exceptions.HTTPError(f"{code} error", response=resp)


@pytest.fixture
def qp(monkeypatch):
    st = {"t": 1000.0, "market_calls": 0, "book_calls": 0, "book": _book(0.43, 0.44), "error": None,
          "market": kalshi_api.parse_market(OPEN_MARKET, now=NOW)}

    def market():
        st["market_calls"] += 1
        return st["market"]

    def book(ticker, depth=10):
        st["book_calls"] += 1
        assert depth == 1
        if st["error"] is not None:
            raise st["error"]
        return st["book"]
    monkeypatch.setattr(kalshi_api, "current_btc15m_market", market)
    monkeypatch.setattr(kalshi_api, "get_orderbook", book)
    monkeypatch.setattr(kalshi_api, "_now", lambda: NOW)
    st["poller"] = kalshi_api.QuotePoller(0.5, 10.0, 30.0, clock=lambda: st["t"])
    return st


def test_poller_quotes_from_the_orderbook_and_caches_within_the_interval(qp):
    p = qp["poller"]
    out = p.get()
    assert (out["yes_bid"], out["yes_ask"], out["quote_source"]) == (0.43, 0.44, "orderbook")
    assert out["quote_age_seconds"] == 0.0 and out["minutes_remaining"] == pytest.approx(7.5)
    qp["t"] += 0.3
    again = p.get()
    assert qp["book_calls"] == 1 and again["quote_age_seconds"] == pytest.approx(0.3)
    qp["t"] += 0.2                                                    # interval (0.5s) reached
    p.get()
    assert qp["book_calls"] == 2 and qp["market_calls"] == 1          # record re-read only every 30s
    qp["t"] += 30
    p.get()
    assert qp["market_calls"] == 2


def test_poller_doubles_the_interval_on_429_and_pauses_until_it_passes(qp):
    p = qp["poller"]
    p.get()
    qp["error"] = _http_error(429)
    qp["t"] += 0.5
    out = p.get()
    assert p.interval == 1.0 and p.last_429 == NOW and qp["book_calls"] == 2
    assert out["yes_bid"] == 0.43 and out["quote_age_seconds"] == pytest.approx(0.5)   # old quote, aged
    qp["t"] += 0.9
    p.get(force=True)                                                 # paused: even force sends nothing
    assert qp["book_calls"] == 2
    qp["t"] += 0.1
    p.get()
    assert qp["book_calls"] == 3 and p.interval == 2.0                # still 429: doubled again
    for _ in range(5):
        qp["t"] += p.interval
        p.get()
    assert p.interval == 10.0                                         # capped at the max


def test_poller_honours_retry_after(qp):
    p = qp["poller"]
    qp["error"] = _http_error(429, retry_after=7)
    p.get()
    assert p.interval == 7.0
    qp["t"] += 6.9
    p.get()
    assert qp["book_calls"] == 1
    qp["t"] += 0.1
    p.get()
    assert qp["book_calls"] == 2


@pytest.mark.parametrize("err", [_http_error(503), requests.exceptions.Timeout("read timed out"),
                                 requests.exceptions.ConnectionError("reset")])
def test_poller_backs_off_on_5xx_and_timeouts_too(qp, err):
    qp["error"] = err
    qp["poller"].get()
    assert qp["poller"].interval == 1.0 and qp["poller"].last_429 is None


def test_poller_shrinks_20_percent_after_30_straight_successes(qp):
    p = qp["poller"]
    p.interval = 8.0
    for i in range(29):
        p.get()
        qp["t"] += p.interval
    assert p.interval == 8.0
    p.get()
    assert p.interval == pytest.approx(6.4)
    p.interval = 0.55
    p.successes = 29
    qp["t"] += 10
    p.get()
    assert p.interval == 0.5                                          # never below the minimum


def test_one_poll_is_shared_across_sessions(qp, monkeypatch):
    import config
    kalshi_api.reset_shared_poller()
    monkeypatch.setattr(kalshi_api, "_poller", None)
    clock = qp["t"]
    a = kalshi_api.shared_poller(0.5, 10, 30)
    a.clock = lambda: clock
    b = kalshi_api.shared_poller(0.5, 10, 30)                         # a second session / tab
    assert a is b
    kalshi_api.live_market(config)
    kalshi_api.live_market(config)
    assert qp["book_calls"] == 1


def test_a_closed_window_drops_the_old_quote_and_reads_the_next_market(qp):
    from datetime import timedelta
    p = qp["poller"]
    p.get()
    later = NOW + timedelta(minutes=7.5)
    nxt = dict(qp["market"], ticker="KXBTC15M-NEXT", close_time=later + timedelta(minutes=15))
    qp["market"] = nxt
    qp["error"] = requests.exceptions.Timeout("x")
    out = p.get(wall=later)
    assert out["ticker"] == "KXBTC15M-NEXT" and out["yes_bid"] is None   # never the last window's quote


def test_orderbook_top_and_market_record_use_the_same_convention():
    # The fixture's market record: YES bid 0.43, NO bid 0.56 -> YES ask 0.44. The same top of book
    # expressed as Kalshi's bids-only orderbook must parse to the same YES bid/ask. (Live, the two
    # endpoints disagree because the record lags -- DASHBOARD_NOTES.md -- not because of convention.)
    rec = kalshi_api.parse_market(OPEN_MARKET, now=NOW)
    book = {"orderbook_fp": {"yes_dollars": [["0.4200", "5"], [OPEN_MARKET["yes_bid_dollars"], "9"]],
                             "no_dollars": [[OPEN_MARKET["no_bid_dollars"], "7"], ["0.5500", "3"]]}}
    top = kalshi_api.parse_orderbook(book)
    assert (top["yes_bid"], top["yes_ask"]) == (rec["yes_bid"], rec["yes_ask"])


def test_saved_live_headers_carry_no_rate_limit_fields():
    cap = json.loads((pathlib.Path(__file__).parent / "fixtures" / "kalshi_response_headers.json").read_text(encoding="utf-8"))
    for part in ("markets_list", "orderbook_depth1", "market_record"):
        assert cap[part]["status"] == 200
        assert not [h for h in cap[part]["headers"] if "ratelimit" in h.lower().replace("-", "")]
    assert cap["orderbook_depth1"]["bytes"] < cap["market_record"]["bytes"] / 10
