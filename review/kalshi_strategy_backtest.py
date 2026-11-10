"""Run from the repo root.

    python review/kalshi_strategy_backtest.py [--cache-dir study_data]
    python review/kalshi_strategy_backtest.py --fetch [--kalshi-base-url URL]

Four further KXBTC15M strategy hypotheses (residual, streak, vol regime, basis), each as a rule
and its mirror: does any of them earn more than it costs? The design -- rules, thresholds,
entries, exits, costs, family, selection -- is fixed in kalshi_strategy_study.py and was committed
before any of it was run on an outcome.

It reads the cache the edge study built. `--fetch` builds that cache first, with the edge study's
own fetch (review/kalshi_market_study.py fetch); `--kalshi-base-url` points that fetch at another
host serving the same public API for this run only (cloud sandboxes cannot always reach the
default one). Nothing here changes a setting on disk, needs a key, or trades.
"""
import argparse
import importlib.util
import os
import sys

REVIEW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(REVIEW_DIR))

import numpy as np
import pandas as pd

import kalshi_strategy_study as st
import kalshi_study as ks

LIMITS = (
    "LIMITS, which apply to every number below:\n"
    "  - Fills are assumed at the quoted top of book with zero latency, on one-minute closing quotes. Candlesticks\n"
    "    carry no depth, so the size really available may be smaller: every figure is an upper bound on a real order.\n"
    "  - Costs are in: YES at the ask, NO at 1 - bid, Kalshi's taker fee rounded up to the cent per order on every leg.\n"
    "  - Only a cell selected on TRAIN and confirmed on the held-out set is a finding. Every other held-out number is\n"
    "    descriptive: it was not chosen in advance, and nothing is claimed from it.\n"
    "  - The model values inside `residual` use Coinbase minutes, not Kalshi's settlement index; `basis` uses a\n"
    "    one-minute OHLC mean as a stand-in for a 60-second average; `streak` assumes a result is public within two\n"
    "    minutes of the close. The streak, vol and basis rules were not tested at any other threshold.\n"
    "  - The designer had read the earlier studies' write-up, held-out tables included, before fixing this design.")


def _review_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REVIEW_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _c(x):
    return f"{100 * x:+.1f}c" if x == x else "   n/a"


def _pct(x):
    return f"{100 * x:5.1f}%" if x == x else "   n/a"


def _hold(k):
    return "settle" if k == st.SETTLE else f"sell+{k}m"


def print_coverage(frame):
    print("Coverage (what each rule could actually be tested on), per entry time:")
    for h in st.FRAME_HORIZONS:
        d = frame[frame.horizon == h]
        book = ks.valid_book(d.bid0.to_numpy(), d.ask0.to_numpy())
        b = d[book]
        parts = [f"{name} {int(b[col].notna().sum()):6d}" for name, col in
                 (("residual", "residual"), ("streak", "streak"), ("vol ratio", "vol_ratio"), ("basis", "basis_dev_bp"))]
        print(f"  {h:2d} min before close: {len(d):6d} contracts, real book at entry {int(book.sum()):6d}; "
              f"of those, feature present: " + ", ".join(parts))
    print()


def print_family(results, split, z):
    print(f"{split.upper()}: profit per contract net of costs, mean [+/- {z:.2f} SE interval]. MDE = smallest edge the "
          "cell could have detected.\n  win vs b/e = win rate against the break-even rate (settlement cells); "
          "'small' = n_eff under " f"{st.MIN_N_EFF}.")
    print(f"  {'rule':19} {'entry':>5} {'exit':>8} {'size':>4} {'trades':>6} {'n_eff':>6} {'YES%':>5} {'forced':>6} "
          f"{'win':>6} {'b/e':>6}   {'mean [interval]':>28} {'MDE':>6}")
    for r in results[results.split == split].itertuples():
        n_eff = f"{r.n_eff:6.0f}" if r.n_eff == r.n_eff else "   n/a"
        print(f"  {r.rule:19} {r.horizon:5d} {_hold(r.hold):>8} {r.contracts:4d} {r.n:6d} {n_eff} "
              f"{_pct(r.yes_share)[1:]:>5} {_pct(r.forced_rate)} {_pct(r.win_rate)} {_pct(r.breakeven)}   "
              f"{_c(r.mean):>7} [{_c(r.lo):>7},{_c(r.hi):>7}] {_c(r.mde)[1:]:>6}{'  small' if r.small else ''}")
    print()


def print_direction(results):
    info = st.direction_information(results)
    info = info[info.contracts == max(st.ORDER_SIZES)]
    print("Direction information per mirrored pair (half the difference of the two rules' means, "
          f"{max(st.ORDER_SIZES)}-lot): what the signal is worth per trade before costs. Descriptive, not a test.")
    for r in info.itertuples():
        print(f"  {r.split:5} {r.pair:36} entry {r.horizon:2d} {_hold(r.hold):>8}: {_c(r.information)}")
    print()


def print_summary(results, out):
    """One line per rule: its best cell on train, and that same cell held out."""
    confirmed = set()
    if len(out["test"]):
        confirmed = {tuple(r) for r in out["test"][out["test"]["confirmed"]][["rule", "horizon", "hold", "contracts"]].to_numpy()}
    selected = {tuple(r) for r in out["selected"][["rule", "horizon", "hold", "contracts"]].to_numpy()}
    train, test = results[results.split == "train"], results[results.split == "test"]
    print("SUMMARY: each rule's best cell on train (by mean), and the same cell on the held-out set.")
    print(f"  {'rule':19} {'cell':>20}   {'train mean [interval]':>30}   {'held-out mean [interval]':>30}   verdict")
    for rule in st.RULES:
        t = train[(train.rule == rule) & (train.n > 1)]
        if t.empty:
            print(f"  {rule:19} no trades")
            continue
        b = t.loc[t["mean"].idxmax()]
        key = (b.rule, b.horizon, b.hold, b.contracts)
        h = test[(test.rule == b.rule) & (test.horizon == b.horizon) & (test.hold == b.hold) & (test.contracts == b.contracts)].iloc[0]
        verdict = "CONFIRMED" if key in confirmed else "selected, not confirmed" if key in selected else "not selected (no claim)"
        cell = f"{int(b.horizon)}m {_hold(int(b.hold))} x{int(b.contracts)}"
        print(f"  {rule:19} {cell:>20}   {_c(b['mean']):>7} [{_c(b.lo):>7},{_c(b.hi):>7}]{'':6}   "
              f"{_c(h['mean']):>7} [{_c(h.lo):>7},{_c(h.hi):>7}]{'':6}   {verdict}")
    print()


def report(markets, candles, grid, cutoff):
    frame = st.with_split(st.trade_frame(markets, candles, grid), cutoff)
    z = ks.bonferroni_z(st.STRATEGY_TESTS)
    print(f"{len(markets)} settled contracts, {len(candles)} with cached quotes; "
          f"{pd.to_datetime(markets[0]['close_ts'], unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(markets[-1]['close_ts'], unit='s'):%Y-%m-%d} UTC.")
    n_test = len({m['ticker'] for m in markets if m['close_ts'] >= cutoff})
    print(f"Held out: the last {n_test} contracts, closing {pd.to_datetime(cutoff, unit='s'):%Y-%m-%d} onward.")
    print(f"Family: {st.STRATEGY_TESTS} tests -> Bonferroni z = {z:.2f}.")
    print(LIMITS + "\n")
    print_coverage(frame)

    results = st.family_results(frame, z)
    print_family(results, "train", z)
    out = st.select_and_test(results, z)
    sel = out["selected"]
    if sel.empty:
        print("Selected on train (interval entirely above zero): none. Nothing goes to the held-out set as a claim.\n"
              "NO RULE HERE HAS SHOWN AN EDGE.")
    else:
        print(f"Selected on train: {len(sel)} cell(s).")
        for r in sel.itertuples():
            print(f"  {r.rule} entry {r.horizon} {_hold(r.hold)} size {r.contracts}: train mean {_c(r.mean)}")
        print(f"HELD-OUT, selected cells only (Bonferroni z = {out['z_test']:.2f} over the {len(sel)} selected):")
        for r in out["test"].itertuples():
            verdict = "CONFIRMED" if r.confirmed else "not confirmed"
            print(f"  {r.rule} entry {r.horizon} {_hold(r.hold)} size {r.contracts}: {r.n} trades, mean {_c(r.mean)} "
                  f"[{_c(r.lo_t)}, {_c(r.hi_t)}]  {verdict}")
        if not out["test"]["confirmed"].any():
            print("NO RULE HERE HAS SHOWN AN EDGE: nothing selected on train held up on the held-out set.")
    print("\nHeld-out numbers for every cell (descriptive only -- nothing is claimed from cells not selected on train):")
    print_family(results, "test", z)
    print_direction(results)
    print_summary(results, out)
    return results, out


def fetch(base_url=None):
    """The edge study's own fetch, optionally against another host serving the same public API."""
    if base_url:
        ks.kalshi_api.BASE_URL = base_url.rstrip("/")        # this process only; nothing is written
    _review_script("kalshi_market_study").cmd_fetch()


def main(argv=None, coinbase_fetch=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=ks.CACHE_DIR)
    ap.add_argument("--fetch", action="store_true", help="build the study cache first (resumable)")
    ap.add_argument("--kalshi-base-url", default=None, help="another host serving Kalshi's public API, for --fetch")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(args.kalshi_base_url)

    def progress(i, n):
        if i % 20 == 0 or i == n:
            print(f"  coinbase day {i}/{n}", flush=True)
    markets, candles, grid, cutoff = _review_script("kalshi_scalp_backtest").load_inputs(args.cache_dir, coinbase_fetch, progress)
    return report(markets, candles, grid, cutoff)


if __name__ == "__main__":
    main()
