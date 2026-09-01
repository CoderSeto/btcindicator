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

Methodological caveat, worth keeping in mind: the weights come from the
same historical window backtest.py covers. That's a real historical edge,
grounded in actual frequency -- but it is NOT an independent, out-of-sample
validation that weighting beats the raw unweighted count. Markets change
regime; a weight that made sense in the backtested window isn't guaranteed
to keep making sense. Treat the weighted score as one more input to
consider alongside the raw composite, not a replacement for it.
"""

_WEIGHTABLE_INDICATORS = ["sma_cross", "ema_cross", "rsi", "macd", "bollinger", "volume", "agg_delta"]


def _weight_from_bucket(bucket, cfg) -> float:
    """
    weight = 1.0 + SCALE * (hit_rate - 0.5), clipped to [MIN, MAX].
    A coin-flip (50%) indicator gets weight 1.0 -- same as unweighted.
    No bucket, or fewer than BACKTEST_MIN_SAMPLE occurrences, also falls
    back to 1.0 (neutral) rather than trusting a noisy small-sample edge.
    """
    if not bucket or bucket.get("total", 0) < cfg.BACKTEST_MIN_SAMPLE:
        return 1.0
    hit_rate = bucket["hits"] / bucket["total"]
    edge = hit_rate - 0.5
    weight = 1.0 + cfg.INDICATOR_WEIGHT_SCALE * edge
    return max(cfg.INDICATOR_WEIGHT_MIN, min(cfg.INDICATOR_WEIGHT_MAX, weight))


def compute_weights(results, cfg) -> dict:
    """
    Returns {indicator_name: {"bullish": weight, "bearish": weight}} for
    the 6 backtestable candle-based indicators, plus agg_delta if
    backtest_agg_delta.py has been run (its buckets are keyed the same
    way -- "agg_delta_bullish"/"agg_delta_bearish" -- so this picks them
    up automatically, no special-casing needed). Order-book imbalance has
    no entry here -- Coinbase exposes no historical order-book data, so
    there's no way to backtest it at all; callers should default to
    weight 1.0 for it and any other name not present.
    """
    buckets = (results or {}).get("indicator_buckets", {})
    weights = {}
    for name in _WEIGHTABLE_INDICATORS:
        weights[name] = {
            "bullish": _weight_from_bucket(buckets.get(f"{name}_bullish"), cfg),
            "bearish": _weight_from_bucket(buckets.get(f"{name}_bearish"), cfg),
        }
    return weights


def weighted_conviction(snapshot, weights: dict) -> dict:
    """
    Applies per-indicator weights to the live signal snapshot. Indicators
    with no entry in `weights` (order-book imbalance, aggregated exchange
    delta) get weight 1.0 -- counted the same as in the raw composite,
    just without any historical up/down-weighting applied.

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
