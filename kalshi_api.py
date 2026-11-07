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

I can't test this against Kalshi's live API myself (external-api.kalshi.com
isn't reachable from the sandbox this was built in), so this is built
carefully against the documented response schema with mocked tests -- it
needs a live sanity check. Run standalone to do that:

    python kalshi_api.py
"""
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
