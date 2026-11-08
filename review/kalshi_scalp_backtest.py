"""Run from the repo root.

    python review/kalshi_scalp_backtest.py [--cache-dir study_data]

The scalping backtest for the KXBTC15M contracts: does buying a side and selling it back
1-3 minutes later, on a simple entry rule, earn more than the round trip costs? The design
-- entry rules, horizons, holds, costs, selection -- is fixed in kalshi_scalp_study.py and was
committed before any result was computed.

It reads the cache the edge study built, so run that first if you haven't:

    python review/kalshi_market_study.py fetch      # settled markets + their 1-minute quotes
    python review/kalshi_market_study.py step3      # (optional) caches Coinbase minutes; this script
                                                    #  fetches any day step3 hasn't

Nothing here changes the cache and nothing is traded.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kalshi_scalp_study as sc
import kalshi_study as ks

FILL_NOTE = ("Fills are assumed at the quoted top of book, in AND out, with zero latency and decisions made "
             "on one-minute closing quotes. Candlesticks carry no depth, so the size really available "
             "may be smaller -- a scalp, which has to trade twice, is the most exposed to that.")


def _c(x):
    return f"{100 * x:+.1f}c" if x == x else "   n/a"


def load_inputs(cache_dir: str = ks.CACHE_DIR, coinbase_fetch=None, progress=None):
    """(markets, parsed candles, Coinbase MinuteGrid, split cutoff) from the study cache."""
    markets_path, _ = ks._paths(cache_dir)
    if not os.path.exists(markets_path):
        raise SystemExit(f"No cached markets at {markets_path}. Run "
                         "`python review/kalshi_market_study.py fetch` first.")
    with open(markets_path, encoding="utf-8") as f:
        parsed = [ks.parse_settled_market(m) for m in json.load(f)]
    markets = [m for m in parsed if m is not None]
    raw = ks.load_candles(cache_dir)
    if not raw:
        raise SystemExit(f"No cached candlesticks under {cache_dir}. Run "
                         "`python review/kalshi_market_study.py fetch` first.")
    candles = {t: ks.parse_candles(c) for t, c in raw.items()}
    cb = ks.load_or_fetch_coinbase(markets[0]["open_ts"] - 4 * 3600, markets[-1]["close_ts"],
                                   cache_dir=cache_dir, fetch=coinbase_fetch, progress=progress)
    grid = ks.MinuteGrid(cb)
    cutoff = ks.split_cutoff([m["close_ts"] for m in markets])
    return markets, candles, grid, cutoff


def print_coverage(frame):
    print("Coverage (what each rule could actually be tested on):")
    first = frame[frame.hold == sc.HOLDS_MIN[0]]
    for h in sc.ENTRY_HORIZONS_MIN:
        d = first[first.horizon == h]
        book = ks.valid_book(d.bid0.to_numpy(), d.ask0.to_numpy())
        print(f"  entry {h:2d} min before close: {len(d):6d} contracts, real book at entry {book.sum():6d}, "
              f"contract signal inputs {d.contract_move.notna().sum():6d}, BTC signal inputs {d.btc_move_bp.notna().sum():6d}")
    for h in sc.ENTRY_HORIZONS_MIN:
        parts = []
        for k in sc.HOLDS_MIN:
            d = frame[(frame.horizon == h) & (frame.hold == k)]
            d = d[ks.valid_book(d.bid0.to_numpy(), d.ask0.to_numpy())]
            parts.append(f"hold {k}m: {100 * (~ks.valid_book(d.bid1.to_numpy(), d.ask1.to_numpy())).mean():4.1f}%")
        print(f"  entry {h:2d}: share of entries with NO sellable exit book (held to settlement instead) -- "
              + ", ".join(parts))
    print()


def print_cost_floor(frame):
    cf = sc.cost_floor(frame)
    print("Cost floor and oracle ceiling, per entry horizon and hold, over every contract with a real book at both ends "
          "(descriptive, no rule, not a test):")
    print(f"  {'entry':>5} {'hold':>4} {'n':>6} {'sprd in':>8} {'sprd out':>8} {'mean |move|':>11}   "
          f"{'1-lot: cost':>11} {'moves>cost':>10} {'ORACLE':>8}   {'100-lot: cost':>13} {'moves>cost':>10} {'ORACLE':>8}")
    for r in cf.itertuples():
        print(f"  {r.horizon:5d} {r.hold:4d} {r.n:6d} {100 * r.spread_in:7.1f}c {100 * r.spread_out:7.1f}c "
              f"{100 * r.move:10.1f}c   {100 * r.cost1:10.1f}c {100 * r.share_moves_cost1:9.1f}% {_c(r.oracle1):>8}   "
              f"{100 * r.cost100:12.1f}c {100 * r.share_moves_cost100:9.1f}% {_c(r.oracle100):>8}")
    print("  cost = (spread in + spread out) / 2 + both fees at the mids: how far the mid must move in your favour just to "
          "break even.\n  ORACLE = profit per contract if the winning side were known in advance and the trade skipped when "
          "both lose. It is look-ahead: a ceiling on ANY entry rule at that hold, not a strategy.\n")


def print_family(results, split, z):
    print(f"{split.upper()}: profit per contract, mean [+/- {z:.2f} SE interval], MDE = smallest edge the cell could have detected")
    print(f"  {'rule':19} {'entry':>5} {'hold':>4} {'size':>4} {'trades':>6} {'n_eff':>6} {'YES%':>5} {'forced':>6}   "
          f"{'mean [interval]':>28} {'MDE':>6}")
    d = results[results.split == split]
    for r in d.itertuples():
        print(f"  {r.rule:19} {r.horizon:5d} {r.hold:4d} {r.contracts:4d} {r.n:6d} {r.n_eff:6.0f} "
              f"{100 * r.yes_share:4.0f}% {100 * r.forced_rate:5.1f}%   "
              f"{_c(r.mean):>7} [{_c(r.lo):>7},{_c(r.hi):>7}] {100 * r.mde:5.1f}c")
    print()


def report(markets, candles, grid, cutoff):
    frame = sc.with_split(sc.trade_frame(markets, candles, grid), cutoff)
    z = ks.bonferroni_z(sc.SCALP_TESTS)
    print(f"{len(markets)} settled contracts, {len(candles)} with cached quotes; "
          f"{pd.to_datetime(markets[0]['close_ts'], unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(markets[-1]['close_ts'], unit='s'):%Y-%m-%d} UTC.")
    n_test = len({m['ticker'] for m in markets if m['close_ts'] >= cutoff})
    print(f"Held out: the last {n_test} contracts, closing {pd.to_datetime(cutoff, unit='s'):%Y-%m-%d} onward.")
    print(f"Family: {sc.SCALP_TESTS} tests -> Bonferroni z = {z:.2f}.")
    print(FILL_NOTE + "\n")
    print_coverage(frame)
    print_cost_floor(frame)

    results = sc.family_results(frame, z)
    print_family(results, "train", z)
    out = sc.select_and_test(results, z)
    sel = out["selected"]
    if sel.empty:
        print("Selected on train (interval entirely above zero): none. Nothing goes to the held-out set as a claim.")
    else:
        print(f"Selected on train: {len(sel)} cell(s).")
        for r in sel.itertuples():
            print(f"  {r.rule} entry {r.horizon} hold {r.hold} size {r.contracts}: train mean {_c(r.mean)}")
        print(f"HELD-OUT, selected cells only (Bonferroni z = {out['z_test']:.2f} over the {len(sel)} selected):")
        for r in out["test"].itertuples():
            verdict = "CONFIRMED" if r.confirmed else "not confirmed"
            print(f"  {r.rule} entry {r.horizon} hold {r.hold} size {r.contracts}: {r.n} trades, mean {_c(r.mean)} "
                  f"[{_c(r.lo_t)}, {_c(r.hi_t)}]  {verdict}")
    print("\nHeld-out numbers for every cell (descriptive only -- nothing is claimed from cells not selected on train):")
    print_family(results, "test", z)
    return results, out


def main(argv=None, coinbase_fetch=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=ks.CACHE_DIR)
    args = ap.parse_args(argv)

    def progress(i, n):
        if i % 20 == 0 or i == n:
            print(f"  coinbase day {i}/{n}", flush=True)
    markets, candles, grid, cutoff = load_inputs(args.cache_dir, coinbase_fetch, progress)
    return report(markets, candles, grid, cutoff)


if __name__ == "__main__":
    main()
