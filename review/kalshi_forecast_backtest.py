"""Run from the repo root.

    python review/kalshi_forecast_backtest.py [--cache-dir study_data]

The KXBTC15M forecasting study: is any fitted P(YES) better than Kalshi's mid, than
kalshi.settlement_probability as the dashboard computes it, or than the base rate? Scored by
Brier and log loss on held-out contracts. The design -- features, models, tuning, split, metrics,
family -- is fixed in kalshi_forecast_study.py and was committed before any model was fitted.

A forecasting study, not a trading rule: no fee, fill or profit appears here. It reads the cache
the edge study built (review/kalshi_market_study.py fetch, or review/kalshi_strategy_backtest.py
--fetch). Per-row probabilities are saved to <cache-dir>/forecast_predictions.csv. Nothing is
traded and no key is needed.
"""
import argparse
import json
import os
import sys

REVIEW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(REVIEW_DIR))

import numpy as np
import pandas as pd

import data_fetch
import kalshi_forecast_study as fc
import kalshi_study as ks

LIMITS = (
    "LIMITS, which apply to every number below:\n"
    "  - This is a FORECASTING study. No fee, spread, fill or profit is in any number here, and a better score is not\n"
    "    a trading edge: the edge study found that costs erase far larger differences than these.\n"
    "  - The market mid is a STRONG baseline (it already beat the dashboard's model). Matching it is not a failure.\n"
    "  - Calibration alone is trivial: 'always say the base rate' is perfectly calibrated and useless. Read the\n"
    "    calibration and reliability columns together with sharpness; Brier and log loss decide.\n"
    "  - A model is BETTER or WORSE than a reference only when its held-out skill interval excludes zero at the\n"
    "    family's Bonferroni z. Everything else, including every per-horizon number, is descriptive.\n"
    "  - The held-out set is the existing one (contracts closing from the 70% cutoff on). Earlier studies had already\n"
    "    scored the baselines on it and the designer had read those results, so it is not a pristine test.\n"
    "  - Only rows with a real two-sided book are scored. At 2 minutes the most decided contracts have none, so no\n"
    "    forecast here is tested on them.\n"
    "  - Prices are Coinbase one-minute candles, not Kalshi's settlement index, and the bot's read is replayed on\n"
    "    those candles with config.py's current settings.")


def load_inputs(cache_dir: str = ks.CACHE_DIR, coinbase_fetch=None, progress=None, now=None):
    """(markets, parsed candles, MinuteGrid, Coinbase frame of completed candles) from the study cache."""
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
    # Completed candles only: a day fetched today can end in a forming one.
    timed = cb.assign(time=pd.to_datetime(cb["ts"], unit="s"))
    done, _ = data_fetch.drop_forming_candle(timed, 60, now)
    cb = done.drop(columns="time").reset_index(drop=True)
    return markets, candles, ks.MinuteGrid(cb), cb


def _iv(lo, hi, scale=100.0, unit="%"):
    return f"[{scale * lo:+6.2f}{unit}, {scale * hi:+6.2f}{unit}]"


def print_coverage(d):
    print("Rows (one per contract per horizon) and how many are usable -- real book, plausible strike, every feature:")
    for split in ("fit", "validation", "test"):
        g = d[d.split == split]
        if g.empty:
            continue
        parts = [f"{h:2d} min {int(g[g.horizon == h].usable.sum()):6d}/{int((g.horizon == h).sum()):6d}" for h in ks.HORIZONS_MIN]
        days = pd.to_datetime(g.close_ts, unit="s")
        print(f"  {split:10} {days.min():%Y-%m-%d} .. {days.max():%Y-%m-%d}  {g.ticker.nunique():6d} contracts; usable: " + ", ".join(parts))
    print()


def print_tuning(fitted):
    t = fitted["tuning"]
    print("Tuning (train only: fitted on `fit`, compared on `validation` by pooled log loss; the held-out set is not read):")
    s = t[t.model == "structural"].sort_values("val_log_loss")
    print(f"  structural: {len(s)} candidates. Best five and the worst:")
    for r in list(s.head(5).itertuples()) + list(s.tail(1).itertuples()):
        print(f"    {r.setting:44} {r.val_log_loss:.5f}")
    kind, spot, df = fitted["structural_choice"]
    print(f"  chosen: vol {kind}, spot {spot}, tail {df or 'normal'}")
    for model in ("features", "combiner"):
        parts = [f"l2 {r.setting.split()[1]}: {r.val_log_loss:.5f}" for r in t[t.model == model].itertuples()]
        print(f"  {model}: " + ", ".join(parts) + f"  -> chosen l2 {fitted['l2'][model]:g}")
    print()


def print_scorecard(card):
    print("HELD-OUT scorecard, pooled over horizons, same rows for every forecast. Lower Brier / log loss is better.\n"
          "  sharpness: sd of the forecast and mean |forecast - 0.5| (0 = says the same thing every time).\n"
          "  calibration: intercept a and slope b of outcome on logit(forecast), +/- day-block SE; calibrated is a = 0, b = 1.")
    print(f"  {'forecast':13} {'kind':8} {'n':>6} {'Brier':>8} {'log loss':>9} {'sharp sd':>9} {'|p-.5|':>7}   {'a (SE)':>16} {'b (SE)':>16}")
    for r in card.itertuples():
        cal = (f"{r.cal_a:+.3f} ({r.cal_a_se:.3f})  {r.cal_b:6.3f} ({r.cal_b_se:.3f})" if r.cal_b == r.cal_b
               else "    n/a: a constant has no slope  ")
        print(f"  {r.forecast:13} {r.kind:8} {r.n:6d} {r.brier:8.5f} {r.log_loss:9.5f} {r.sharp_sd:9.4f} {r.sharp_abs:7.4f}   {cal}")
    print()


def print_skill(skills, z):
    print(f"HELD-OUT skill = 1 - loss(model) / loss(reference), with a +/- {z:.2f} SE day-block interval "
          f"({fc.FAMILY_TESTS} intervals in the family).\n  Positive: the model removes that share of the reference's loss. "
          "n_eff is the day-block effective sample size.")
    for kind, title in (("baseline", "vs each baseline"), ("component", "a fitted combination vs each of its components")):
        print(f"  -- {title} --")
        print(f"  {'model':13} {'reference':13} {'metric':9} {'n_eff':>7} {'skill':>8} {'interval':>22}   verdict")
        for r in skills[skills.kind == kind].itertuples():
            n_eff = f"{r.n_eff:7.0f}" if r.n_eff == r.n_eff else "    n/a"
            print(f"  {r.model:13} {r.reference:13} {r.metric:9} {n_eff} {100 * r.skill:+7.2f}% {_iv(r.lo, r.hi):>22}   {r.verdict}")
    print()


def print_reliability(pred, z):
    print(f"HELD-OUT reliability, per bucket of each forecast: mean forecast vs YES rate [+/- {z:.2f} SE day-block interval]. "
          f"'small' = n_eff under {fc.MIN_BUCKET_N_EFF}.\n  Descriptive. A forecast can be reliable here and still useless (see LIMITS).")
    for name in fc.BASELINES + fc.MODELS:
        t = fc.reliability_table(pred[name], pred["outcome"], pred["day"], z)
        print(f"  {name}")
        for r in t.itertuples():
            n_eff = f"{r.n_eff:6.0f}" if r.n_eff == r.n_eff else "   n/a"
            iv = f"[{r.lo:.3f}, {r.hi:.3f}]" if r.lo == r.lo else "[  n/a        ]"
            print(f"    {r.bucket:8} n {r.n:6d} n_eff {n_eff}  forecast {r.forecast:.3f}  YES rate {r.yes_rate:.3f} {iv}"
                  f"{'  small' if r.small else ''}")
    print()


def print_side_by_side(pred):
    names = list(fc.BASELINES + fc.MODELS)
    print("Raw probabilities side by side (held-out), never combined by hand. Mean of each forecast within each bucket of the market mid:")
    print(f"  {'mid bucket':10} {'n':>6} {'YES':>6} " + " ".join(f"{n[:12]:>12}" for n in names))
    b = ks.bucket_of(pred["market"])
    for i in range(len(ks.BUCKET_EDGES) - 1):
        g = pred[b == i]
        if g.empty:
            continue
        print(f"  {ks.BUCKET_EDGES[i]:.1f}-{ks.BUCKET_EDGES[i + 1]:.1f}    {len(g):6d} {g.outcome.mean():6.3f} "
              + " ".join(f"{g[n].mean():12.3f}" for n in names))
    print("  First rows as they are:")
    print(f"  {'ticker':28} {'min':>3} {'YES':>3} " + " ".join(f"{n[:12]:>12}" for n in names))
    for r in pred.head(8).itertuples():
        print(f"  {r.ticker:28} {r.horizon:3d} {int(r.outcome):3d} " + " ".join(f"{getattr(r, n):12.3f}" for n in names))
    print()


def print_by_horizon(pred):
    t = fc.by_horizon(pred)
    print("HELD-OUT per horizon (descriptive: no interval, not in the family, nothing is claimed from it):")
    print(f"  {'forecast':13} " + " ".join(f"{f'{h} min Brier':>13} {'log loss':>9}" for h in ks.HORIZONS_MIN))
    for name in fc.BASELINES + fc.MODELS:
        parts = []
        for h in ks.HORIZONS_MIN:
            r = t[(t.forecast == name) & (t.horizon == h)]
            parts.append(f"{r.brier.iloc[0]:13.5f} {r.log_loss.iloc[0]:9.5f}" if len(r) else f"{'n/a':>13} {'n/a':>9}")
        print(f"  {name:13} " + " ".join(parts))
    print()


def print_summary(card, skills, z):
    print(f"SUMMARY (held-out; family of {len(fc.MODELS)} models, {fc.FAMILY_TESTS} skill intervals, z = {z:.2f}):")
    print(f"  {'model':13} {'Brier':>8} {'log loss':>9} {'Brier skill vs market [interval]':>36} {'cal a':>7} {'cal b':>7}   vs market (Brier / log loss)")
    for m in fc.MODELS:
        c = card[card.forecast == m].iloc[0]
        s = skills[(skills.model == m) & (skills.reference == "market")].set_index("metric")
        b, ll = s.loc["brier"], s.loc["log_loss"]
        print(f"  {m:13} {c.brier:8.5f} {c.log_loss:9.5f} {100 * b.skill:+10.2f}% {_iv(b.lo, b.hi):>24} {c.cal_a:+7.3f} {c.cal_b:7.3f}   "
              f"{b.verdict} / {ll.verdict}")
    for name in fc.BASELINES:
        c = card[card.forecast == name].iloc[0]
        print(f"  {name + ' (baseline)':24} Brier {c.brier:.5f}  log loss {c.log_loss:.5f}")
    better = skills[(skills.reference == "market") & (skills.verdict == "BETTER")]
    if better.empty:
        print("NO MODEL HERE BEATS THE MARKET MID: no held-out skill interval against it lies above zero.")
    else:
        print("Better than the market mid, interval above zero: "
              + ", ".join(f"{r.model} ({r.metric})" for r in better.itertuples())
              + ".\n  A better forecast is not a trading edge: no cost is in these numbers.")
    print()


def report(markets, candles, grid, coinbase, cache_dir=None):
    val_cut, test_cut = fc.split_cutoffs(markets)
    z = ks.bonferroni_z(fc.FAMILY_TESTS)
    print(f"{len(markets)} settled contracts, {len(candles)} with cached quotes; "
          f"{pd.to_datetime(markets[0]['close_ts'], unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(markets[-1]['close_ts'], unit='s'):%Y-%m-%d} UTC.")
    print(f"Held-out: contracts closing from {pd.to_datetime(test_cut, unit='s'):%Y-%m-%d %H:%M} UTC (the existing held-out "
          f"set; nothing later is in the cache). Validation, inside train: from {pd.to_datetime(val_cut, unit='s'):%Y-%m-%d %H:%M}.")
    print(f"Family: {len(fc.MODELS)} models, {fc.FAMILY_TESTS} skill intervals -> Bonferroni z = {z:.2f}.")
    print(LIMITS + "\n")

    d = fc.build_dataset(markets, candles, grid, coinbase)
    print_coverage(d)
    fitted = fc.tune_and_fit(d)                      # train only
    print_tuning(fitted)
    print(f"Base rate (YES share of usable train rows): {fitted['base_rate']:.4f}\n")

    test = d[d.usable & (d.split == "test")]         # scored once, after every choice above is final
    pred = fc.predictions(fitted, test)
    card = fc.scorecard(pred, z)
    skills = fc.skill_table(pred, z)
    print_scorecard(card)
    print_skill(skills, z)
    print_by_horizon(pred)
    print_reliability(pred, z)
    print_side_by_side(pred)
    print_summary(card, skills, z)
    if cache_dir:
        path = os.path.join(cache_dir, "forecast_predictions.csv")
        pred.to_csv(path, index=False)
        print(f"Per-row held-out probabilities saved to {path}")
    return {"dataset": d, "fitted": fitted, "predictions": pred, "scorecard": card, "skills": skills}


def main(argv=None, coinbase_fetch=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=ks.CACHE_DIR)
    args = ap.parse_args(argv)

    def progress(i, n):
        if i % 50 == 0 or i == n:
            print(f"  coinbase day {i}/{n}", flush=True)
    markets, candles, grid, coinbase = load_inputs(args.cache_dir, coinbase_fetch, progress)
    return report(markets, candles, grid, coinbase, args.cache_dir)


if __name__ == "__main__":
    main()
