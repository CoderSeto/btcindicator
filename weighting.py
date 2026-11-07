"""
Turns per-indicator historical hit rates (from backtest.py's
indicator_buckets) into weights, and combines them with the live signal
snapshot into a "weighted conviction" score.

This is a SEPARATE, additive read -- it does not modify or replace
snapshot.composite/strength, which continues to drive the alert threshold,
CSV logging, and the outlook headline exactly as before. Think of it as a
second opinion: "if you'd trusted each indicator in proportion to how
often it's actually been right, which way would that lean, and by how
much?"

Weights are measured as edge over the window's base rate, on the
overlap-corrected effective sample size, and stay at 1.0 unless that edge
clears INDICATOR_WEIGHT_MIN_Z standard errors -- see weight_details().

Methodological caveat, worth keeping in mind: the weights come from the
same historical window backtest.py covers. That's a real historical edge,
grounded in actual frequency -- but it is NOT an independent, out-of-sample
validation that weighting beats the raw unweighted count. Markets change
regime; a weight that made sense in the backtested window isn't guaranteed
to keep making sense. Treat the weighted score as one more input to
consider alongside the raw composite, not a replacement for it.
"""

_WEIGHTABLE_INDICATORS = ["sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "agg_delta"]


def weight_details(bucket, cfg) -> dict:
    """
    The weight for one indicator/direction bucket, and why it is what it is.

        edge   = hit_rate - base_rate
        z      = edge / sqrt(base_rate * (1 - base_rate) / n_eff)
        weight = 1.0 + SCALE * edge, clipped to [MIN, MAX]

    `base_rate` is how often the backtested window moved in this direction at
    all, so it's what a call with no skill scores. Measuring edge against 0.5
    instead rewarded every bullish call in a window that drifted up
    (REVIEW_FINDINGS D3). `n_eff` is the overlap-corrected sample size from
    backtest.effective_sample_size(); the raw row count overstates the
    evidence ~3.5x for the persistent state votes.

    The weight stays at 1.0 -- the same "no evidence" answer as a missing
    bucket -- when n_eff < BACKTEST_MIN_SAMPLE, or when |z| <
    INDICATOR_WEIGHT_MIN_Z, i.e. the edge is within sampling noise. On
    backtests of a pure random walk, ungated weights still spread +/-0.08 to
    +/-0.16 from noise alone.

    Results saved before n_eff/base_rate existed fall back to n_eff = total and
    base_rate = 0.5, the old behaviour (re-run the backtest to get the honest
    numbers).

    Returns {"weight", "reason", "edge", "z", "n_eff", "base_rate"}, with
    reason one of "no_data", "small_sample", "within_noise", "measured".
    """
    out = {"weight": 1.0, "reason": "no_data", "edge": None, "z": None, "n_eff": None, "base_rate": None}
    if not bucket or not bucket.get("total"):
        return out
    n_eff = bucket.get("n_eff", bucket["total"])
    base = bucket.get("base_rate")
    base = 0.5 if base is None else base
    hit_rate = bucket["hits"] / bucket["total"]
    edge = hit_rate - base
    se = (base * (1.0 - base) / n_eff) ** 0.5 if n_eff > 0 else float("inf")
    z = edge / se if se > 0 else (0.0 if edge == 0 else float("inf"))
    out.update(edge=edge, z=z, n_eff=n_eff, base_rate=base)

    if n_eff < cfg.BACKTEST_MIN_SAMPLE:
        out["reason"] = "small_sample"
        return out
    if abs(z) < getattr(cfg, "INDICATOR_WEIGHT_MIN_Z", 0.0):
        out["reason"] = "within_noise"
        return out
    weight = 1.0 + cfg.INDICATOR_WEIGHT_SCALE * edge
    out["weight"] = max(cfg.INDICATOR_WEIGHT_MIN, min(cfg.INDICATOR_WEIGHT_MAX, weight))
    out["reason"] = "measured"
    return out


def _weight_from_bucket(bucket, cfg) -> float:
    """Just the weight from weight_details()."""
    return weight_details(bucket, cfg)["weight"]


def compute_weights(results, cfg) -> dict:
    """
    Returns {indicator_name: {"bullish": weight, "bearish": weight}} for
    the 6 backtestable candle-based indicators, plus agg_delta if
    backtest_agg_delta.py has been run (its buckets are keyed the same
    way -- "agg_delta_bullish"/"agg_delta_bearish" -- so this picks them
    up automatically, no special-casing needed). Neither order-book signal
    has an entry here: no exchange publishes historical level-2 snapshots,
    so there is no way to backtest either the imbalance vote or book flow
    at all. Callers should default to weight 1.0 for them and for any
    other name not present.
    """
    return {
        name: {d: detail["weight"] for d, detail in dirs.items()}
        for name, dirs in compute_weight_details(results, cfg).items()
    }


def compute_weight_details(results, cfg) -> dict:
    """Same shape as compute_weights(), with weight_details() dicts as the values."""
    buckets = (results or {}).get("indicator_buckets", {})
    return {
        name: {d: weight_details(buckets.get(f"{name}_{d}"), cfg) for d in ("bullish", "bearish")}
        for name in _WEIGHTABLE_INDICATORS
    }


def weighted_conviction(snapshot, weights: dict) -> dict:
    """
    Applies per-indicator weights to the live signal snapshot. Indicators
    with no entry in `weights` -- the two order-book signals (`orderbook`
    and `book_flow`) -- get weight 1.0, counted the same as in the raw
    composite, just without any historical up/down-weighting applied.

    Returns {"available": False} if every individual signal is neutral
    (nothing directional to weigh). Otherwise:
      {
        "available": True,
        "direction": "bullish" | "bearish" | "neutral",
        "weighted_pct": float in (0.5, 1.0] -- winning side's share of weighted mass,
        "bullish_weight": float, "bearish_weight": float,
        "breakdown": {indicator_name: {"vote": ..., "weight": ...}},
      }
    """
    breakdown = {}
    bullish_w = 0.0
    bearish_w = 0.0

    for name, vote in snapshot.individual.items():
        if vote == "neutral":
            continue
        w = weights.get(name, {}).get(vote, 1.0)
        breakdown[name] = {"vote": vote, "weight": w}
        if vote == "bullish":
            bullish_w += w
        else:
            bearish_w += w

    total_w = bullish_w + bearish_w
    if total_w == 0:
        return {"available": False}

    if bullish_w > bearish_w:
        direction = "bullish"
    elif bearish_w > bullish_w:
        direction = "bearish"
    else:
        direction = "neutral"

    weighted_pct = max(bullish_w, bearish_w) / total_w

    return {
        "available": True,
        "direction": direction,
        "weighted_pct": weighted_pct,
        "bullish_weight": bullish_w,
        "bearish_weight": bearish_w,
        "breakdown": breakdown,
    }
