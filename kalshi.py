"""
Fair-value calculator for Kalshi's KXBTC15M-style binary BTC contracts:
"will price be above strike K at expiry".

Prices what the contract actually settles on -- the simple average of the
60 one-second index prints before close, compared with the strike -- under
driftless geometric Brownian motion, given current price, strike, time
remaining and recent realized volatility (settlement_probability). The
textbook point-in-time version, the N(d2) term from Black-Scholes, is kept
as crossing_probability and reported alongside for comparison.

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
  - Settlement is a 60-SECOND AVERAGE of the reference index at the
    window's close, and P(YES) models that average. Averaging lowers the
    variance (to sigma^2 * (T - ~39.5s) instead of sigma^2 * T), so it is
    always at least as far from 50% as the point-in-time N(d2): under a
    point apart at 14 minutes, ~13 points at 1 minute with spot 5bp above
    the strike (79% point-in-time vs 92% averaged). What remains
    approximate is the FINAL MINUTE: some of the 60 prints have already
    been taken, their real values aren't available here, and they're
    assumed equal to the current price. Treat P(YES) inside the last
    minute as rough.
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


def realized_volatility(df: pd.DataFrame, lookback_minutes: int, candle_seconds: int = 60) -> dict:
    """
    Annualized realized volatility from close-to-close log returns over
    the last `lookback_minutes` of candle data whose candles are
    `candle_seconds` long. The candle size matters twice: it sets how many
    candles cover the lookback, and how many such periods make a year.
    Annualizing 5-minute returns as if they were 1-minute returns
    overstates vol by sqrt(5).

    Returns {"annual_vol": float, "samples_used": int, "samples_requested": int}
    or {"annual_vol": None, "reason": str} if there isn't enough data or
    price never moved (zero variance -- can't annualize a zero).
    `samples_used` vs `samples_requested` lets a caller notice when less
    data was actually available than asked for, rather than silently
    computing a shorter-than-intended window.
    """
    lookback_candles = max(int(math.ceil(lookback_minutes * 60 / candle_seconds)), 1)
    recent = df.tail(lookback_candles + 1)
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
    per_candle_std = math.sqrt(variance)
    if per_candle_std <= 0:
        return {"annual_vol": None, "reason": "Price didn't move at all in this window -- can't estimate volatility."}

    candles_per_year = 365.25 * 24 * 60 * 60 / candle_seconds
    annual_vol = per_candle_std * math.sqrt(candles_per_year)

    return {
        "annual_vol": annual_vol,
        "samples_used": samples_used,
        "samples_requested": lookback_candles,
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


SETTLEMENT_PRINTS = 60          # KXBTC15M: simple average of 60 BRTI prints...
SETTLEMENT_PRINT_SECONDS = 1.0  # ...one per second, the last one at close


def settlement_variance_seconds(seconds_remaining: float,
                                prints: int = SETTLEMENT_PRINTS,
                                spacing: float = SETTLEMENT_PRINT_SECONDS) -> float:
    """
    Variance, in seconds of Brownian time, of the settlement average
    (1/m) * sum W(u_k) over the m prints at u_k = T - (m - k) * spacing,
    k = 1..m, seen from now (W(0) = 0 known).

    Only prints still in the future are random. For those, sorted ascending
    as u_1 < ... < u_f, the i-th one is the minimum in 2(f - i) + 1 of the
    pairs in sum_{j,k} min(u_j, u_k), so

        var = (1/m^2) * sum_i u_i * (2(f - i) + 1)

    With every print still ahead this is (T - m*spacing) + spacing*(m+1)(2m+1)/(6m):
    for the 60 one-second prints, T minus ~39.5 seconds. Prints already taken
    contribute nothing here (see settlement_probability for how their level
    is treated).
    """
    future = sorted(u for u in (seconds_remaining - (prints - k) * spacing for k in range(1, prints + 1))
                    if u > 0)
    f = len(future)
    return sum(u * (2 * (f - i) + 1) for i, u in enumerate(future, start=1)) / prints ** 2


def settlement_probability(current_price: float, strike: float, minutes_remaining: float,
                           annual_vol: float) -> float:
    """
    P(YES) for what KXBTC15M actually settles on: YES iff the simple average of
    the 60 one-second BRTI prints before close is >= the strike
    (rules_primary; strike_type "greater_or_equal").

    Driftless GBM, so every print is a martingale and the average's mean is
    the current price. Its log is treated as normal with variance
    sigma^2 * settlement_variance_seconds(T) -- a moment-matched lognormal,
    which at these variances (~1e-5) is indistinguishable from the exact
    distribution (checked against a second-by-second Monte Carlo of the rule
    in tests/test_kalshi.py).

    Versus crossing_probability(), the point-in-time N(d2): the average has
    less variance than the last print, so this is always at least as far from
    50% -- about a point at 14 minutes, ~13 points at 1 minute with spot 5bp
    above the strike.

    Limitation, stated in the output too: inside the final minute some prints
    have already been taken. Their actual BRTI values aren't available here,
    so they're taken to equal the current price.
    """
    if current_price <= 0 or strike <= 0:
        raise ValueError("current_price and strike must be positive")
    if minutes_remaining <= 0:
        return 1.0 if current_price >= strike else 0.0
    if annual_vol is None or annual_vol <= 0:
        raise ValueError("annual_vol must be a positive number")

    seconds_per_year = 365.25 * 24 * 60 * 60
    var = annual_vol ** 2 * settlement_variance_seconds(minutes_remaining * 60.0) / seconds_per_year
    if var <= 0:
        return 1.0 if current_price >= strike else 0.0
    d = (math.log(current_price / strike) - 0.5 * var) / math.sqrt(var)
    return _norm_cdf(d)


MAX_STRIKE_DISTANCE = 0.05     # config.KALSHI_MAX_STRIKE_DISTANCE; see strike_check


def strike_check(strike: float, current_price: float, max_distance: float = MAX_STRIKE_DISTANCE):
    """
    None if the strike is plausible, else the reason it isn't. A strike more
    than `max_distance` (a fraction, 0.05 = 5%) from spot is refused, not
    priced: BTC doesn't move that far within one 15-minute window, and
    Kalshi's own records have held strikes divided by 10,000 (7.2437 for
    72,437.29). The model prices those at P(YES) = 100%, which looks like a
    sure thing rather than an error. The reason's first words carry no
    dollar sign, so it can be cut short for a metric label.
    """
    if max_distance is None or not (strike > 0 and current_price > 0):
        return None
    distance = abs(strike / current_price - 1.0)
    if distance <= max_distance + 1e-9:          # exactly at the limit passes despite float residue
        return None
    return (f"Strike {distance * 100:.1f}% from spot -- likely bad. ${strike:,.2f} vs spot "
            f"${current_price:,.2f} is far more than BTC moves in 15 minutes, so the strike is "
            f"probably wrong (Kalshi's own records have held strikes divided by 10,000). Not "
            f"priced; check the strike. The limit is {max_distance * 100:.0f}% "
            f"(KALSHI_MAX_STRIKE_DISTANCE).")


def fair_value(df: pd.DataFrame, strike: float, minutes_remaining: float, lookback_minutes: int,
               candle_seconds: int = 60, current_price: float = None,
               max_strike_distance: float = MAX_STRIKE_DISTANCE) -> dict:
    """
    Convenience wrapper: takes the current price (`current_price`, or the
    latest candle close if it isn't given), computes realized volatility
    over lookback_minutes from the candles, and returns the
    full picture -- not just the final probability, so every input is
    visible and auditable rather than a number appearing from nowhere.

    Pass `current_price` from the ticker (data_fetch.latest_price): a candle's
    close lags the last trade -- by up to 5 minutes when Coinbase's candle
    cache was hit (gotcha #19) -- and near settlement a few basis points move
    P(YES) by tens of points. Volatility over a 120-minute window barely
    notices.

    Returns {"available": False, "reason": ...} if there isn't enough
    data to compute a trustworthy estimate, or if the strike fails
    strike_check (then "strike_rejected" is True). Pass
    max_strike_distance=None to skip that check.
    """
    if df.empty:
        return {"available": False, "reason": "No candle data available."}
    if strike <= 0:
        return {"available": False, "reason": "Strike must be a positive price."}

    current_price = float(df["close"].iloc[-1]) if current_price is None else float(current_price)
    if not current_price > 0:
        return {"available": False, "reason": "Current price must be a positive number."}
    bad_strike = strike_check(strike, current_price, max_strike_distance)
    if bad_strike:
        return {"available": False, "reason": bad_strike, "strike_rejected": True}
    vol_result = realized_volatility(df, lookback_minutes, candle_seconds)
    if vol_result["annual_vol"] is None:
        return {"available": False, "reason": vol_result["reason"]}

    annual_vol = vol_result["annual_vol"]
    p_yes = settlement_probability(current_price, strike, minutes_remaining, annual_vol)
    p_yes_point = crossing_probability(current_price, strike, minutes_remaining, annual_vol)

    # Standard deviation of the settlement average, not of the last print.
    seconds_per_year = 365.25 * 24 * 60 * 60
    var = settlement_variance_seconds(max(minutes_remaining, 0) * 60.0) / seconds_per_year
    price_std = current_price * annual_vol * math.sqrt(var) if var > 0 else 0.0

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
        # The point-in-time N(d2), kept for comparison and logged alongside so
        # calibration can show whether modelling the average actually helped.
        "p_yes_point": p_yes_point,
        "final_minute": 0 < minutes_remaining < SETTLEMENT_PRINTS * SETTLEMENT_PRINT_SECONDS / 60.0,
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

    spot = data_fetch.latest_price(cfg.PRODUCT_ID, df)
    result = fair_value(df, args.strike, minutes_remaining, lookback, candle_seconds=cfg.GRANULARITY,
                        max_strike_distance=cfg.KALSHI_MAX_STRIKE_DISTANCE,
                        current_price=spot["price"])
    if not result["available"]:
        print(f"[error] {result['reason']}")
    else:
        print(f"\nCurrent price:       ${result['current_price']:,.2f}  (from {spot['source']})")
        if spot["source"] != "ticker":
            print(f"  !! {spot['reason']}; using the newest candle instead. "
                  f"{data_fetch.CANDLE_CACHE_NOTE}.")
        print(f"Strike:              ${result['strike']:,.2f}")
        print(f"Minutes remaining:   {result['minutes_remaining']:.1f}")
        print(f"Annualized vol:      {result['annual_vol'] * 100:.1f}%  "
              f"(from {result['vol_samples_used']} of {result['vol_samples_requested']} requested samples)")
        print(f"Settlement-avg sd:   ${result['price_std']:,.2f}")
        print(f"\nFair value P(YES / above strike): {result['p_yes'] * 100:.1f}%  "
              f"(point-in-time N(d2), ignoring the averaging: {result['p_yes_point'] * 100:.1f}%)")
        print(f"Fair value P(NO / below strike):  {result['p_no'] * 100:.1f}%")
        print(
            "\n(Model-only estimate: Coinbase price used as a proxy for Kalshi's actual "
            "multi-exchange\nsettlement source; assumes driftless GBM; prices the 60-second "
            "settlement average, but inside the\nfinal minute takes the already-elapsed prints "
            "at the current price. See README for full\nlimitations. Not financial advice.)"
        )
