"""
Runs the pre-registered forward edge evaluation (EDGE_HYPOTHESES.md,
kalshi_forward_study.py) on edge_record.csv and prints the report.

    python review/kalshi_forward_edge.py            # resolves outcomes from Kalshi, then evaluates
    python review/kalshi_forward_edge.py --offline  # cached outcomes only

Each hypothesis is evaluated ONCE under its stopping rule. Until a
hypothesis has reached its stop (n_eff >= STOP_N_EFF or STOP_DATE) the report
labels its line "INTERIM, NOT A VERDICT"; that is a pipeline check, not a
result, and must not be quoted as one. Outcomes are cached in
config.KALSHI_OUTCOMES_CACHE_PATH, the same cache kalshi_tracking uses.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import edge_recorder  # noqa: E402
import kalshi_forward_study as fs  # noqa: E402
import kalshi_tracking  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=config.EDGE_RECORD_PATH)
    ap.add_argument("--offline", action="store_true", help="do not query Kalshi for outcomes; use the cache")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.record):
        raise SystemExit(f"{args.record} not found: run `python edge_recorder.py` first")
    record = fs.load_record(args.record)
    cache = kalshi_tracking.load_outcomes_cache(config.KALSHI_OUTCOMES_CACHE_PATH)
    if not args.offline:
        tickers = sorted(t for t in record["ticks"]["ticker"].unique() if t)
        cache = kalshi_tracking.resolve_outcomes(tickers, cache)
        kalshi_tracking.save_outcomes_cache(config.KALSHI_OUTCOMES_CACHE_PATH, cache)
    ev = fs.evaluate(record, cache, datetime.now(timezone.utc), edge_recorder.MODEL_VERSION)
    print(fs.report(ev))


if __name__ == "__main__":
    main()
