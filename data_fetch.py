"""
Thin wrapper around Coinbase Exchange's public market data REST API.
No API key is required for these endpoints (they're read-only market data).
"""
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
