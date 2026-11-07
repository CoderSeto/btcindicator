"""
Fair-value calculator for Kalshi's KXBTC15M-style binary BTC contracts:
"will price be above strike K at expiry".

Uses the standard lognormal/geometric-Brownian-motion crossing probability
-- the same math underlying the N(d2) term in Black-Scholes -- given
current price, strike, time remaining, and recent realized volatility.
This is the textbook "fair" risk-neutral probability the contract
resolves YES, assuming no drift (a pure random walk) and no jumps.

This answers a genuinely different question than the rest of this bot.
Everything else here answers "is price leaning up or down" -- Kalshi's
contract asks "what's the actual probability price is above a FIXED
number at a FIXED future time". A qualitative lean isn't a probability;
this tries to give an actual one.

Known limitations, stated plainly rather than glossed over:
  - Coinbase-vs-settlement-source basis risk: Kalshi settles on CF
    Benchmarks' Real-Time Index, a multi-exchange composite -- not
    Coinbase alone. This uses Coinbase's price as a proxy for the
    current price input, which can diverge from the actual settlement
    source by a few dollars, especially during a volatile close. On a
    contract decided by a small margin, that gap matters.
  - Settlement is actually a 60-SECOND AVERAGE of the reference index at
    the window's close, not a single point-in-time price. This model
    treats it as a single point-in-time draw -- a reasonable
    approximation for most of the window, but it overstates how extreme
    the true probability gets in the final minute or two (the real
    distribution has slightly lower variance near expiry than modeled
    here, since averaging smooths out some of the terminal noise).
  - Assumes driftless geometric Brownian motion (no jumps, no trend) --
    a standard simplifying assumption for pricing this kind of contract,
    not a claim that BTC actually moves this way. It deliberately ignores
    whatever directional lean the rest of this bot's signals suggest.
    For a complementary, SEPARATELY-computed empirical read, see
    backtest.py's hit-rate buckets (set BACKTEST_LOOKAHEAD_CANDLES to
    match your time horizon first) -- shown alongside this, not blended
    into it, since there's no validated evidence for the right way to
    combine the two into one number.
  - Not investment advice, and this file is not a licensed financial
    advisor. It's a calculator, not a guarantee.

Run standalone for a quick check:
    python kalshi.py --strike 105000 --minutes 8
"""
import math
from datetime import datetime, timezone

import pandas as pd


def realized_volatility(df: pd.DataFrame, lookback_minutes: int) -> dict:
    """
    Annualized realized volatility from close-to-close log returns over
    the last `lookback_minutes` of 1-minute candle data.

    Returns {"annual_vol": float, "samples_used": int, "samples_requested": int}
    or {"annual_vol": None, "reason": str} if there isn't enough data or
    price never moved (zero variance -- can't annualize a zero).
    `samples_used` vs `samples_requested` lets a caller notice when less
    data was actually available than asked for, rather than silently
    computing a shorter-than-intended window.
    """
    recent = df.tail(lookback_minutes + 1)
    if len(recent) < 3:
        return {"annual_vol": None, "reason": "Not enough candle history to estimate volatility."}

    closes = recent["close"].to_numpy()
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    samples_used = len(log_returns)
    if samples_used < 2:
        return {"annual_vol": None, "reason": "Not enough valid price observations to estimate volatility."}

    mean_r = sum(log_returns) / samples_used
    variance = sum((r - mean_r) ** 2 for r in log_returns) / (samples_used - 1)
    per_minute_std = math.sqrt(variance)
    if per_minute_std <= 0:
        return {"annual_vol": None, "reason": "Price didn't move at all in this window -- can't estimate volatility."}

    minutes_per_year = 365.25 * 24 * 60
    annual_vol = per_minute_std * math.sqrt(minutes_per_year)

    return {
        "annual_vol": annual_vol,
        "samples_used": samples_used,
        "samples_requested": lookback_minutes,
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the math.erf identity -- no scipy dependency needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def crossing_probability(current_price: float, strike: float, minutes_remaining: float,
                          annual_vol: float) -> float:
    """
    P(price finishes ABOVE strike at expiry), under driftless geometric
    Brownian motion with the given annualized volatility -- the same math
    as the N(d2) term in Black-Scholes.

    Explicitly handles minutes_remaining <= 0 (avoids a division by zero):
    with no time left, the outcome is already decided by where price sits
    relative to the strike right now. KXBTC15M is "greater_or_equal", so
    finishing exactly on the strike is YES.
    """
    if current_price <= 0 or strike <= 0:
        raise ValueError("current_price and strike must be positive")

    if minutes_remaining <= 0:
        return 1.0 if current_price >= strike else 0.0

    if annual_vol is None or annual_vol <= 0:
        raise ValueError("annual_vol must be a positive number")

    minutes_per_year = 365.25 * 24 * 60
    tau = minutes_remaining / minutes_per_year

    d2 = (math.log(current_price / strike) - 0.5 * annual_vol ** 2 * tau) / (annual_vol * math.sqrt(tau))
    return _norm_cdf(d2)


def fair_value(df: pd.DataFrame, strike: float, minutes_remaining: float, lookback_minutes: int) -> dict:
    """
    Convenience wrapper: pulls current price from the latest candle close,
    computes realized volatility over lookback_minutes, and returns the
    full picture -- not just the final probability, so every input is
    visible and auditable rather than a number appearing from nowhere.

    Returns {"available": False, "reason": ...} if there isn't enough
    data to compute a trustworthy estimate.
    """
    if df.empty:
        return {"available": False, "reason": "No candle data available."}
    if strike <= 0:
        return {"available": False, "reason": "Strike must be a positive price."}

    current_price = float(df["close"].iloc[-1])
    vol_result = realized_volatility(df, lookback_minutes)
    if vol_result["annual_vol"] is None:
        return {"available": False, "reason": vol_result["reason"]}

    annual_vol = vol_result["annual_vol"]
    p_yes = crossing_probability(current_price, strike, minutes_remaining, annual_vol)

    minutes_per_year = 365.25 * 24 * 60
    tau = max(minutes_remaining, 0) / minutes_per_year
    price_std = current_price * annual_vol * math.sqrt(tau) if tau > 0 else 0.0

    return {
        "available": True,
        "current_price": current_price,
        "strike": strike,
        "minutes_remaining": minutes_remaining,
        "annual_vol": annual_vol,
        "vol_samples_used": vol_result["samples_used"],
        "vol_samples_requested": vol_result["samples_requested"],
        "price_std": price_std,
        "p_yes": p_yes,
        "p_no": 1.0 - p_yes,
    }


def minutes_until_next_kalshi_close(now: datetime = None) -> float:
    """
    Kalshi's 15-minute BTC contracts settle on quarter-hour UTC boundaries
    (:00, :15, :30, :45). Returns minutes remaining until the next one --
    a convenience default for the fair-value calculator's time-remaining
    input. This assumes you're looking at the currently-open window; if
    you're checking a specific contract, its own listed close time is
    always the source of truth, not this.
    """
    now = now or datetime.now(timezone.utc)
    minutes_into_hour = now.minute + now.second / 60 + now.microsecond / 60_000_000
    next_quarter = math.ceil(minutes_into_hour / 15) * 15
    if next_quarter <= minutes_into_hour:
        next_quarter += 15
    return next_quarter - minutes_into_hour


if __name__ == "__main__":
    import argparse

    import config as cfg
    import data_fetch

    parser = argparse.ArgumentParser(description="Kalshi 15m BTC fair-value calculator")
    parser.add_argument("--strike", type=float, required=True, help="Kalshi contract strike price")
    parser.add_argument(
        "--minutes", type=float, default=None,
        help="Minutes remaining until settlement (default: time until the next 15-min UTC boundary)",
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Volatility lookback window in minutes (default: config.KALSHI_VOL_LOOKBACK_MINUTES)",
    )
    args = parser.parse_args()

    minutes_remaining = args.minutes if args.minutes is not None else minutes_until_next_kalshi_close()
    lookback = args.lookback if args.lookback is not None else cfg.KALSHI_VOL_LOOKBACK_MINUTES

    print(f"Fetching recent {cfg.PRODUCT_ID} price history...")
    df = data_fetch.get_candles(cfg.PRODUCT_ID, cfg.GRANULARITY, max(cfg.CANDLE_HISTORY, lookback + 10))

    result = fair_value(df, args.strike, minutes_remaining, lookback)
    if not result["available"]:
        print(f"[error] {result['reason']}")
    else:
        print(f"\nCurrent price:       ${result['current_price']:,.2f}")
        print(f"Strike:              ${result['strike']:,.2f}")
        print(f"Minutes remaining:   {result['minutes_remaining']:.1f}")
        print(f"Annualized vol:      {result['annual_vol'] * 100:.1f}%  "
              f"(from {result['vol_samples_used']} of {result['vol_samples_requested']} requested samples)")
        print(f"Est. price std dev:  ${result['price_std']:,.2f}")
        print(f"\nFair value P(YES / above strike): {result['p_yes'] * 100:.1f}%")
        print(f"Fair value P(NO / below strike):  {result['p_no'] * 100:.1f}%")
        print(
            "\n(Model-only estimate: Coinbase price used as a proxy for Kalshi's actual "
            "multi-exchange\nsettlement source; assumes driftless GBM; ignores the final "
            "60-second settlement averaging.\nSee README for full limitations. Not "
            "financial advice.)"
        )
