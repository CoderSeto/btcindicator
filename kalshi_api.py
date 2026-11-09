"""
Public (no-auth) Kalshi market data client for the KXBTC15M series --
auto-discovers the currently open 15-minute BTC contract and pulls its
exact strike, close time, and Kalshi's own quoted market price, instead
of typing them in by hand each time.

Confirmed via Kalshi's official API docs (docs.kalshi.com):
  - Base URL: https://external-api.kalshi.com/trade-api/v2 -- no API key
    needed for these read-only market-data endpoints.
  - Series ticker for the 15-minute up/down BTC series is KXBTC15M.
    NOT to be confused with KXBTCD, which is Kalshi's separate HOURLY
    directional BTC series -- easy to mix up (some sources conflate the
    two), confirmed distinct by cross-checking multiple sources.

Verified live against external-api.kalshi.com on 2026-11-07, from the user's
machine (the build sandbox can't reach it): it returned the open market with a
real strike, close time and quote. The tests run on verbatim payloads in
tests/fixtures/. Re-run the live check any time:

    python kalshi_api.py
"""
import threading
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC15M"


def get_open_markets(series_ticker: str = SERIES_TICKER) -> list:
    """
    Returns the list of currently open markets in the given series.
    For KXBTC15M there's normally exactly one open market at a time --
    the current 15-minute window. No authentication required.
    """
    url = f"{BASE_URL}/markets"
    params = {"series_ticker": series_ticker, "status": "open"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("markets", [])


def get_market(ticker: str) -> dict:
    """Full details for one market by its exact ticker. No auth required."""
    url = f"{BASE_URL}/markets/{ticker}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("market", {})


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_market(market: dict, now: datetime = None) -> dict:
    """
    Normalizes a raw Kalshi market record (as returned by get_open_markets
    or get_market) into exactly what kalshi.fair_value() needs, plus
    Kalshi's own quoted YES/NO price for comparison. Pure function, no
    network call -- kept separate from current_btc15m_market() so the
    parsing logic can be tested against a hand-built record without a
    live connection.

    Returns {"available": False, "reason": ...} if the record is missing
    fields this needs.
    """
    strike = market.get("floor_strike")
    close_time_str = market.get("close_time")
    if strike is None or not close_time_str:
        return {"available": False, "reason": "Market record is missing strike or close time."}

    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except ValueError:
        return {"available": False, "reason": f"Couldn't parse close_time: {close_time_str!r}"}
    if close_time.tzinfo is None:
        # Kalshi's timestamps are UTC ("...Z"); a naive one would otherwise raise
        # TypeError (naive minus aware) below, which no caller catches.
        close_time = close_time.replace(tzinfo=timezone.utc)

    now = now or datetime.now(timezone.utc)
    minutes_remaining = max(0.0, (close_time - now).total_seconds() / 60.0)

    return {
        "available": True,
        "ticker": market.get("ticker"),
        "strike": float(strike),
        "close_time": close_time,
        "minutes_remaining": minutes_remaining,
        "yes_bid": _to_float(market.get("yes_bid_dollars")),
        "yes_ask": _to_float(market.get("yes_ask_dollars")),
        "last_price": _to_float(market.get("last_price_dollars")),
        "status": market.get("status"),
    }


def current_btc15m_market() -> dict:
    """
    Fetches and normalizes the currently open KXBTC15M market. Returns
    {"available": False, "reason": ...} if no market is open right now
    (e.g. briefly between windows) or the request fails.
    """
    try:
        markets = get_open_markets()
    except requests.exceptions.RequestException as e:
        return {"available": False, "reason": f"Couldn't reach Kalshi's API: {e}"}

    if not markets:
        return {"available": False, "reason": "No open KXBTC15M market found right now."}

    return parse_market(markets[0])


def _now() -> datetime:
    """The poll cache's clock (a function so tests can move it)."""
    return datetime.now(timezone.utc)


# --- The shared, adaptive quote poll (the dashboard's live strip) -----------
#
# Measured from the user's machine on 2026-11-08 (DASHBOARD_NOTES.md,
# review/kalshi_poll_probe.py): the market record's yes_bid/yes_ask -- both
# /markets?series_ticker= and /markets/{ticker} -- lagged the live book by
# tens of seconds (0 of 15 back-to-back reads agreed; the list sat on one
# quote for ~20s while the book moved 8c), while /markets/{ticker}/orderbook
# ?depth=1 is live and 91 bytes against ~2.1 KB. So the quote comes from the
# orderbook; the market record is read only for the ticker, strike and close
# time (once per KALSHI_MARKET_REFRESH_SECONDS, and as soon as the window
# closes). No rate-limit headers were returned, and 120 requests a minute
# drew no 429, so nothing here assumes a limit: it backs off when Kalshi says so.

class RateLimited(Exception):
    """Kalshi answered 429 / 5xx or timed out; `retry_after` is its Retry-After in seconds, if any."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(resp):
    v = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return max(0.0, float(v)) if v is not None else None
    except ValueError:
        return None


def _backoff_error(e):
    """A RateLimited for errors that call for backing off, else None (re-raise)."""
    if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return RateLimited(f"{type(e).__name__}: {e}")
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        code = e.response.status_code
        if code == 429 or code >= 500:
            return RateLimited(f"HTTP {code}", _retry_after(e.response))
    return None


class QuotePoller:
    """
    One Kalshi poll for the whole process (Streamlit runs every tab and session in
    one process), at an adaptive interval:

      - starts at `min_s`; a call within the interval gets the cached quote;
      - 429, 5xx, timeout or connection error: honour Retry-After if given, else
        double the interval (capped at `max_s`), and send nothing until it passes;
      - after `shrink_after` consecutive successes, shrink the interval by
        `shrink` (x0.8) toward `min_s`.

    Only one caller fetches at a time; the others get the cached quote rather than
    queueing a second request. Each result carries `quote_age_seconds` (since the
    book was read), so a stale quote can never pass for a live one.
    """

    def __init__(self, min_s, max_s, market_refresh_s, shrink_after=30, shrink=0.8, clock=None):
        self.min_s, self.max_s, self.market_refresh_s = float(min_s), float(max_s), float(market_refresh_s)
        self.shrink_after, self.shrink = int(shrink_after), float(shrink)
        self.clock = clock or time.monotonic
        self.interval = self.min_s
        self.next_at = None              # monotonic time the next request may go out
        self.successes = 0
        self.requests = 0
        self.last_429 = None             # wall-clock datetime of the last 429
        self.last_error = None
        self.market = None               # parse_market() of the open window
        self.market_at = None
        self.quote = None                # parse_orderbook() top of book for self.market
        self.quote_at = None
        self._lock = threading.Lock()
        self._in_flight = False

    def _fail(self, err: RateLimited, wall):
        self.successes = 0
        self.last_error = str(err)
        if "429" in str(err):
            self.last_429 = wall
        self.interval = min(self.max_s, err.retry_after if err.retry_after is not None else self.interval * 2)
        self.next_at = self.clock() + self.interval

    def _ok(self):
        self.successes += 1
        self.last_error = None
        if self.successes >= self.shrink_after:
            self.interval = max(self.min_s, self.interval * self.shrink)
            self.successes = 0
        self.next_at = self.clock() + self.interval

    def _fetch(self, wall):
        mono = self.clock()
        need_market = (self.market is None or not self.market.get("available")
                       or self.market["close_time"] <= wall
                       or self.market_at is None or mono - self.market_at >= self.market_refresh_s)
        if need_market:
            self.requests += 1
            m = current_btc15m_market()
            if not m.get("available"):
                reason = m.get("reason", "")
                if reason.startswith("Couldn't reach"):          # a failed request, not "no market open"
                    raise RateLimited(reason)
                self.market, self.market_at, self.quote, self.quote_at = m, mono, None, None
                return
            if self.market is None or m.get("ticker") != self.market.get("ticker"):
                self.quote, self.quote_at = None, None
            self.market, self.market_at = m, mono
        self.requests += 1
        try:
            top = parse_orderbook(get_orderbook(self.market["ticker"], depth=1))
        except requests.exceptions.RequestException as e:
            err = _backoff_error(e)
            if err is None:
                raise
            raise err
        self.quote, self.quote_at = top, self.clock()

    def get(self, force: bool = False, wall=None) -> dict:
        """The current market with the freshest quote (fetching if due, or `force` and not paused)."""
        wall = wall or _now()
        with self._lock:
            mono = self.clock()
            paused = self.next_at is not None and mono < self.next_at and self.last_error is not None
            closed = self.market is not None and self.market.get("available") and self.market["close_time"] <= wall
            due = self.next_at is None or mono >= self.next_at or closed or (force and not paused)
            go = due and not self._in_flight
            if go:
                self._in_flight = True
        if go:
            try:
                self._fetch(wall)
            except RateLimited as e:
                with self._lock:
                    self._fail(e, wall)
            except (requests.exceptions.RequestException, KalshiSchemaError) as e:
                with self._lock:
                    self._fail(RateLimited(f"{type(e).__name__}: {e}"), wall)
            else:
                with self._lock:
                    self._ok()
            finally:
                with self._lock:
                    self._in_flight = False
        return self.snapshot(wall)

    def snapshot(self, wall=None) -> dict:
        wall = wall or _now()
        with self._lock:
            info = {"poll_interval": self.interval, "last_429": self.last_429, "requests": self.requests,
                    "poll_error": self.last_error}
            if self.market is None:
                return {"available": False, "reason": f"Couldn't reach Kalshi's API: {self.last_error}"
                        if self.last_error else "Kalshi not polled yet.", **info}
            out = dict(self.market, **info)
        if not out.get("available"):
            return out
        out["minutes_remaining"] = max(0.0, (out["close_time"] - wall).total_seconds() / 60.0)
        if self.quote is None:
            out.update(yes_bid=None, yes_ask=None, yes_bid_size=None, yes_ask_size=None,
                       quote_age_seconds=None, quote_source=None)
        else:
            out.update(yes_bid=self.quote["yes_bid"], yes_ask=self.quote["yes_ask"],
                       yes_bid_size=self.quote.get("yes_bid_size"), yes_ask_size=self.quote.get("yes_ask_size"),
                       quote_age_seconds=max(0.0, self.clock() - self.quote_at), quote_source="orderbook")
        return out


_poller = None
_poller_lock = threading.Lock()


def shared_poller(min_s, max_s, market_refresh_s) -> QuotePoller:
    """The process-wide QuotePoller (created on first use; later settings are ignored)."""
    global _poller
    with _poller_lock:
        if _poller is None:
            _poller = QuotePoller(min_s, max_s, market_refresh_s)
        return _poller


def reset_shared_poller():
    """Drop the process-wide poller (tests)."""
    global _poller
    with _poller_lock:
        _poller = None


def live_market(cfg, force: bool = False) -> dict:
    """The dashboard's quote: the shared poll, at cfg's KALSHI_POLL_* settings."""
    return shared_poller(cfg.KALSHI_POLL_MIN_SECONDS, cfg.KALSHI_POLL_MAX_SECONDS,
                         cfg.KALSHI_MARKET_REFRESH_SECONDS).get(force=force)


if __name__ == "__main__":
    print(f"Fetching the currently open {SERIES_TICKER} market from Kalshi's public API...\n")
    result = current_btc15m_market()
    if not result["available"]:
        print(f"[error] {result['reason']}")
        print(
            "\nIf this keeps failing, double check SERIES_TICKER is still correct -- Kalshi "
            "occasionally renames or restructures series. You can also try get_open_markets() "
            "with a different guess, or browse kalshi.com/markets directly to find the right one."
        )
    else:
        print(f"Ticker:             {result['ticker']}")
        print(f"Strike:             ${result['strike']:,.2f}")
        print(f"Close time (UTC):   {result['close_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Minutes remaining:  {result['minutes_remaining']:.1f}")
        if result["yes_bid"] is not None and result["yes_ask"] is not None:
            print(f"Kalshi YES bid/ask: ${result['yes_bid']:.2f} / ${result['yes_ask']:.2f}")
        if result["last_price"] is not None:
            print(f"Last traded price:  ${result['last_price']:.2f}")
        print("\nIf these numbers look sensible (a real strike near current BTC price, a close "
              "time in the next 15 minutes), the integration is working.")


# --- Order book and prints (edge_recorder.py) ------------------------------
#
# UNTRUSTED PARSERS. Both endpoints are public and need no auth, but this
# code was written where Kalshi's API was unreachable, so neither has yet
# been checked against a real payload. `python edge_recorder.py --capture`
# saves verbatim responses to tests/fixtures/; until those exist, treat the
# field names below as Kalshi's documented schema, not as verified. Anything
# the parsers don't recognise raises KalshiSchemaError rather than guessing.

class KalshiSchemaError(ValueError):
    """A Kalshi payload did not have the shape this code expects."""


def get_orderbook(ticker: str, depth: int = 10) -> dict:
    """Raw GET /markets/{ticker}/orderbook response (bids on each side). No auth."""
    url = f"{BASE_URL}/markets/{ticker}/orderbook"
    resp = requests.get(url, params={"depth": depth}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_trades(ticker: str, limit: int = 100, min_ts=None) -> dict:
    """Raw GET /markets/trades response: the latest executed prints for one market. No auth."""
    url = f"{BASE_URL}/markets/trades"
    params = {"ticker": ticker, "limit": limit}
    if min_ts is not None:
        params["min_ts"] = int(min_ts)
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _levels(raw, scale: float) -> list:
    """[[price, quantity], ...] -> [(price_dollars, quantity)] sorted best (highest bid) first."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise KalshiSchemaError(f"orderbook side is {type(raw).__name__}, expected a list")
    out = []
    for level in raw:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise KalshiSchemaError(f"orderbook level {level!r} is not [price, quantity]")
        try:
            price = float(level[0]) * scale
            qty = float(level[1])
        except (TypeError, ValueError) as e:
            raise KalshiSchemaError(f"orderbook level {level!r}: {e}")
        if not (0.0 <= price <= 1.0) or qty < 0:
            raise KalshiSchemaError(f"orderbook level {level!r} out of range")
        out.append((price, qty))
    return sorted(out, key=lambda l: -l[0])


def parse_orderbook(payload: dict) -> dict:
    """
    Kalshi's book lists only BIDS, per side: `yes` is the resting YES bids,
    `no` the resting NO bids. A YES ask is the mirror of a NO bid, so
    yes_ask = 1 - best NO bid, with the NO bid's quantity as its size.

    Payload may be wrapped as `orderbook_fp` (seen live, 2026-11-08) or `orderbook`.
    Accepts either the dollar-string form (`yes_dollars` / `no_dollars`,
    [["0.56", "100"], ...]) or the integer-cent form (`yes` / `no`,
    [[56, 100], ...]), preferring dollars. Returns
    {"yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size"} with None for an
    empty side. Raises KalshiSchemaError for anything else.
    """
    if not isinstance(payload, dict):
        raise KalshiSchemaError("orderbook payload is not an object")
    book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    if not isinstance(book, dict):
        raise KalshiSchemaError("'orderbook' is not an object")
    if "yes_dollars" in book or "no_dollars" in book:
        yes, no = _levels(book.get("yes_dollars"), 1.0), _levels(book.get("no_dollars"), 1.0)
    elif "yes" in book or "no" in book:
        yes, no = _levels(book.get("yes"), 0.01), _levels(book.get("no"), 0.01)
    else:
        raise KalshiSchemaError(f"orderbook has neither yes/no nor yes_dollars/no_dollars: {sorted(book)}")
    out = {"yes_bid": None, "yes_bid_size": None, "yes_ask": None, "yes_ask_size": None}
    if yes:
        out["yes_bid"], out["yes_bid_size"] = yes[0]
    if no:
        out["yes_ask"], out["yes_ask_size"] = round(1.0 - no[0][0], 6), no[0][1]
    return out


def parse_trades(payload: dict) -> list:
    """
    GET /markets/trades -> [{"trade_id", "ticker", "created_time" (UTC datetime),
    "yes_price", "count", "taker_side"}], oldest first. Accepts
    `yes_price_dollars` or integer-cent `yes_price`, and `count_fp` or `count`.
    Raises KalshiSchemaError when a required field is missing.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("trades"), list):
        raise KalshiSchemaError("trades payload has no 'trades' list")
    out = []
    for t in payload["trades"]:
        try:
            if "yes_price_dollars" in t:
                price = float(t["yes_price_dollars"])
            else:
                price = float(t["yes_price"]) / 100.0
            count = float(t["count_fp"]) if "count_fp" in t else float(t["count"])
            created = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            side = t["taker_side"]
        except (KeyError, TypeError, ValueError) as e:
            raise KalshiSchemaError(f"trade {t!r}: {e}")
        if side not in ("yes", "no"):
            raise KalshiSchemaError(f"trade taker_side {side!r} is neither 'yes' nor 'no'")
        out.append({"trade_id": t.get("trade_id"), "ticker": t.get("ticker"), "created_time": created,
                    "yes_price": price, "count": count, "taker_side": side})
    return sorted(out, key=lambda x: x["created_time"])
