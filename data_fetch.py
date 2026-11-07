"""
Thin wrapper around Coinbase Exchange's public market data REST API.
No API key is required for these endpoints (they're read-only market data).
"""
from datetime import datetime, timezone

import requests
import pandas as pd

BASE_URL = "https://api.exchange.coinbase.com"
HEADERS = {"User-Agent": "btc-indicator-bot/1.0"}


def get_candles(product_id: str, granularity: int, limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent OHLCV candles.
    Coinbase returns candles as [time, low, high, open, close, volume].
    """
    url = f"{BASE_URL}/products/{product_id}/candles"
    params = {"granularity": granularity}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.tail(limit).reset_index(drop=True)


def drop_forming_candle(df: pd.DataFrame, granularity: int, now: datetime = None):
    """
    Splits off the last candle if its bucket hasn't closed yet. Returns
    (completed_candles, forming_row_or_None).

    Coinbase stamps each candle with its bucket's START time, so the candle
    stamped t covers [t, t + granularity) and is still forming until
    t + granularity. A forming candle has a partial close and partial volume:
    scoring signals on it (a 1-minute candle 20 seconds in rarely reaches
    2x average volume) means the live votes are a different signal from the
    completed candles the backtest scored. Deciding by timestamp makes this
    right whether or not the API returns the forming bucket.
    """
    if df.empty:
        return df, None
    now = now or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert("UTC").tz_localize(None)       # candle times are naive UTC
    last_start = pd.Timestamp(df["time"].iloc[-1])
    if last_start.tzinfo is not None:
        last_start = last_start.tz_convert("UTC").tz_localize(None)
    if last_start + pd.Timedelta(seconds=granularity) > now_ts:
        return df.iloc[:-1].reset_index(drop=True).copy(), df.iloc[-1]
    return df, None


def get_candles_range(product_id: str, granularity: int, start, end) -> pd.DataFrame:
    """
    Fetch candles for a specific start/end range (used for backtesting).
    The live bot uses get_candles() instead, which just returns the most
    recent candles without needing a range.
    `start`/`end` accept anything pandas.Timestamp can parse.
    """
    url = f"{BASE_URL}/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])

    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def get_order_book(product_id: str, level: int = 2) -> dict:
    """
    Fetch the order book. Level 2 gives aggregated price levels
    (top ~50 per side) -- enough for an imbalance signal without
    needing the full level-3 feed.
    """
    url = f"{BASE_URL}/products/{product_id}/book"
    params = {"level": level}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_ticker(product_id: str) -> dict:
    """Fetch the current best bid/ask and last trade price."""
    url = f"{BASE_URL}/products/{product_id}/ticker"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


# The bare `/candles?granularity=N` URL is served from a cache that Coinbase
# refreshes about every 5 minutes: measured on 2026-11-07, the newest candle
# went 23:17 -> 23:22 -> 23:27 and its close stayed frozen in between, so a
# price read off it was up to ~5 minutes old and $48 (~6bp) off the ticker.
# That's fine for indicators on 120-minute windows and fatal for a Kalshi
# contract a minute from settling. The price people read and the Kalshi model
# use comes from the ticker instead.
CANDLE_CACHE_NOTE = ("Coinbase serves the candles endpoint from a cache refreshed about every "
                     "5 minutes, so this price can be up to ~5 minutes old")


def latest_price(product_id: str, candles: pd.DataFrame = None) -> dict:
    """
    The freshest price available: the last trade from /ticker.

    Returns {"price", "source", "time", "reason"}. "source" is "ticker", or
    "candles" when the ticker request failed and the newest candle's close is
    used instead (then "reason" says why, and the price may be minutes stale --
    CANDLE_CACHE_NOTE). With no ticker and no candles, "price" is None.
    Never raises: a missing ticker shouldn't take the page down.
    """
    reason = None
    try:
        t = get_ticker(product_id)
        price = float(t["price"])
        if not price > 0:
            raise ValueError(f"ticker price {t['price']!r} is not a positive number")
        ts = t.get("time")
        return {"price": price, "source": "ticker",
                "time": pd.Timestamp(ts) if ts else None, "reason": None}
    except Exception as e:                       # network, HTTP, schema: all mean "no ticker"
        reason = f"ticker unavailable ({e})"
    if candles is None or candles.empty:
        return {"price": None, "source": None, "time": None, "reason": reason}
    return {"price": float(candles["close"].iloc[-1]), "source": "candles",
            "time": pd.Timestamp(candles["time"].iloc[-1]), "reason": reason}
