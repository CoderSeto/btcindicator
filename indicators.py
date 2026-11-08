"""
Classic technical indicators computed on an OHLCV DataFrame.
Each function adds column(s) to the DataFrame and returns it.
"""
import numpy as np
import pandas as pd


def add_sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    df[f"sma_{period}"] = df[col].rolling(period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = df[col].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # No losses in the window: gains only is the most overbought a window can
    # be (RSI 100, the limit as losses -> 0), not neutral. It used to read 50,
    # so a one-way run cast no overbought vote at all. A flat window (no gains
    # either) and the warm-up candles stay neutral at 50.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    df["rsi"] = rsi.fillna(50)
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int, col: str = "close") -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int, num_std: float, col: str = "close") -> pd.DataFrame:
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    return df


def add_volume_spike(df: pd.DataFrame, window: int, multiplier: float) -> pd.DataFrame:
    avg_vol = df["volume"].rolling(window).mean()
    df["volume_avg"] = avg_vol
    df["volume_spike"] = df["volume"] > (avg_vol * multiplier)
    return df


def compute_all(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = add_sma(df, cfg.SMA_FAST)
    df = add_sma(df, cfg.SMA_SLOW)
    df = add_ema(df, cfg.EMA_FAST)
    df = add_ema(df, cfg.EMA_SLOW)
    df = add_rsi(df, cfg.RSI_PERIOD)
    df = add_macd(df, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    df = add_bollinger_bands(df, cfg.BOLLINGER_PERIOD, cfg.BOLLINGER_STD)
    df = add_volume_spike(df, cfg.VOLUME_SPIKE_WINDOW, cfg.VOLUME_SPIKE_MULTIPLIER)
    return df
