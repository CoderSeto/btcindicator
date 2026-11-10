"""Run from the repo root.

    python review/kalshi_flow_backtest.py --fetch [--kalshi-base-url URL]   # count new contracts, cache Binance.US trades (resumable)
    python review/kalshi_flow_backtest.py [--cache-dir study_data]          # fit on train, score the held-out set once

Does Binance.US trade flow improve a calibrated P(YES) forecast for KXBTC15M? Each flow model is
compared with the same model without its flow columns, and with the market mid. The design --
features, windows, models, tuning, split, metrics, family, verdict rule -- is fixed in
kalshi_flow_study.py and was committed before any trade was fetched.

A forecasting study, not a trading rule: no fee, fill or profit appears here. It reads the cache the
edge study built and adds study_data/binance_us_trades/. Per-row probabilities are saved to
<cache-dir>/flow_predictions.csv. Public endpoints only; nothing is traded and no key is needed.
"""
import argparse
import importlib.util
import os
import sys
import time

REVIEW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(REVIEW_DIR))

import numpy as np
import pandas as pd
import requests

import kalshi_flow_study as fl
import kalshi_forecast_study as fc
import kalshi_study as ks

LIMITS = (
    "LIMITS, which apply to every number below:\n"
    "  - This is a FORECASTING study. No fee, spread, fill or profit is in any number here; a better score is not a\n"
    "    trading edge.\n"
    "  - ONE venue: Binance.US BTCUSDT only. The live bot's agg_delta pools Coinbase, Binance.US and Bybit; Bybit was\n"
    "    geo-blocked where this ran and Coinbase has no time-ranged trade history. Binance.US is a thin venue, so this\n"
    "    is an approximation of the live flow signal, not a reconstruction of it.\n"
    "  - Binance.US quotes USDT, not USD. Flow is used only as dimensionless ratios, which the USDT/USD basis does not\n"
    "    touch; the venue's prices are not used at all.\n"
    "  - No order-book data: none exists historically. Only executed trades are in the flow features.\n"
    "  - Kalshi quotes are one-minute closes, so flow is read at whole minutes although trades are timed to the ms.\n"
    "  - Rows whose flow window has a data hole or too few trades are EXCLUDED and counted below, never imputed, so\n"
    "    every score here is on a subset of the forecasting study's rows and is not comparable with its numbers.\n"
    "  - The market mid is a strong baseline, and calibration alone is trivial (the base rate is calibrated and\n"
    "    useless): Brier and log loss decide.\n"
    "  - FLOW HELPS a model only if both its skill intervals against the same model without flow lie above zero at the\n"
    "    family's Bonferroni z AND it is not worse than the market. Everything else is descriptive.")


def _review_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REVIEW_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iv(lo, hi):
    return f"[{100 * lo:+6.2f}%, {100 * hi:+6.2f}%]"


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "btc-indicator-study/1.0"
    return s


def trade_span_ms(markets) -> tuple:
    """The span of trades the study needs: an hour before the first open to the last close."""
    return (markets[0]["open_ts"] - 3600) * 1000, markets[-1]["close_ts"] * 1000


def fetch(cache_dir=ks.CACHE_DIR, kalshi_base_url=None, workers=2):
    """Count contracts newer than the cache (extending it only if there are enough), then cache the trades."""
    forecast = _review_script("kalshi_forecast_backtest")
    if kalshi_base_url:
        ks.kalshi_api.BASE_URL = kalshi_base_url.rstrip("/")     # this process only; nothing is written
    markets, _, _, _ = forecast.load_inputs(cache_dir)
    meta = fl.read_meta(cache_dir)
    if "new_settled_contracts" not in meta:
        last_close = markets[-1]["close_ts"]
        n_new = fl.count_new_settled(_session(), last_close)
        meta = {"cache_last_close": last_close, "new_settled_contracts": n_new, "checked_at": int(time.time()),
                "pristine_after": None}
        if n_new >= fl.PRISTINE_MIN_CONTRACTS:
            raw = ks.load_or_fetch_markets(_session(), cache_dir, refresh=True)
            parsed = [m for m in (ks.parse_settled_market(r) for r in raw) if m is not None]
            ks.fetch_missing_candles(_session(), parsed, cache_dir, historical_before=ks.historical_cutoff(_session()),
                                     workers=4, session_factory=_session)
            meta["pristine_after"] = last_close
            markets, _, _, _ = forecast.load_inputs(cache_dir)
        fl.write_meta(cache_dir, meta)
    print(f"Contracts settled after the cache's last close: {meta['new_settled_contracts']} "
          f"(a pristine held-out set needs {fl.PRISTINE_MIN_CONTRACTS}).", flush=True)

    start_ms, end_ms = trade_span_ms(markets)
    local = {}

    def get(params):
        import threading
        key = threading.get_ident()
        if key not in local:
            local[key] = _session()
        return fl.binance_get(local[key], params)

    def progress(i, n, day):
        if i % 10 == 0 or i == n:
            print(f"  trades for {i}/{n} days (last: {day})", flush=True)
    got = fl.fetch_missing_days(get, cache_dir, start_ms, end_ms, progress=progress, workers=workers)
    print(f"fetched {got} day(s) of Binance.US {fl.SYMBOL} aggTrades", flush=True)


def print_trades(trades, start_ms, end_ms):
    n = len(trades["ts"])
    days = len(fl.days_between(start_ms, end_ms))
    buy = float(trades["buy"].mean()) if n else float("nan")
    print(f"Binance.US {fl.SYMBOL} aggTrades cached: {n:,} over {days} UTC days ({n / max(days, 1):,.0f} a day); "
          f"taker-buy share {100 * buy:.1f}%.")
    print(f"  gaps: {len(trades['id_gaps'])} break(s) in the trade-id sequence ({trades['missing_ids']:,} ids missing), "
          f"{len(trades['holes']) - len(trades['id_gaps'])} uncovered span(s), {trades['duplicate_ids']} duplicate id(s). "
          "Every window touching one is excluded.\n")


def print_coverage(d):
    print(f"Rows (one per contract per horizon). usable = the forecasting study's rule; hole = the {fl.LONG_MIN}-minute flow "
          f"window touches a data hole; thin = under {fl.MIN_TRADES_LONG} trades in it. Scored: usable, no hole, not thin.")
    print(f"  {'split':10} {'horizon':>7} {'rows':>7} {'usable':>7} {'hole':>6} {'thin':>6} {'scored':>7} {'short window empty':>19}")
    for split in ("fit", "validation", "test"):
        for h in ks.HORIZONS_MIN:
            g = d[(d.split == split) & (d.horizon == h)]
            if g.empty:
                continue
            u = g[g.usable]
            used = g[g.use]
            empty = f"{100 * used.short_empty.mean():5.1f}% of scored" if len(used) else "n/a"
            print(f"  {split:10} {h:7d} {len(g):7d} {len(u):7d} {int(u.hole.sum()):6d} {int(u.thin.sum()):6d} {len(used):7d} {empty:>19}")
    print()


def print_tuning(fitted):
    print("Tuning (train only: fitted on `fit`, compared on `validation` by pooled log loss; the held-out set is not read):")
    t = fitted["tuning"]
    for m in fl.MODELS:
        parts = [f"l2 {r.l2:g}: {r.val_log_loss:.5f}" for r in t[t.model == m].itertuples()]
        print(f"  {m:19} " + ", ".join(parts) + f"  -> l2 {fitted['l2'][m]:g}")
    print(f"  high_vol threshold (median 120-minute realized vol of `fit` rows): {fitted['vol_threshold']:.4f}")
    print(f"Base rate (YES share of the train rows used): {fitted['base_rate']:.4f}\n")


def print_scorecard(card):
    print("HELD-OUT scorecard, pooled over horizons, same rows for every forecast. Lower Brier / log loss is better.\n"
          "  sharpness: sd of the forecast and mean |forecast - 0.5|. calibration: intercept a and slope b of outcome on\n"
          "  logit(forecast), +/- day-block SE; calibrated is a = 0, b = 1.")
    print(f"  {'forecast':19} {'kind':9} {'n':>6} {'Brier':>8} {'log loss':>9} {'sharp sd':>9} {'|p-.5|':>7}   {'a (SE)':>16} {'b (SE)':>16}")
    for r in card.itertuples():
        cal = (f"{r.cal_a:+.3f} ({r.cal_a_se:.3f})  {r.cal_b:6.3f} ({r.cal_b_se:.3f})" if r.cal_b == r.cal_b
               else "    n/a: a constant has no slope  ")
        print(f"  {r.forecast:19} {r.kind:9} {r.n:6d} {r.brier:8.5f} {r.log_loss:9.5f} {r.sharp_sd:9.4f} {r.sharp_abs:7.4f}   {cal}")
    print()


def print_skill(skills, z):
    print(f"HELD-OUT skill = 1 - loss(model) / loss(reference), +/- {z:.2f} SE day-block interval ({fl.FAMILY_TESTS} intervals "
          "in the family). Positive: the model removes that share of the reference's loss. n_eff is the day-block "
          "effective sample size.")
    titles = (("counterpart", "THE EFFECT OF FLOW: each flow model vs the same model without its flow columns"),
              ("baseline", "vs each baseline"), ("component", "the model with every input vs each fitted component"))
    for kind, title in titles:
        print(f"  -- {title} --")
        print(f"  {'model':19} {'reference':19} {'metric':9} {'n_eff':>7} {'skill':>8} {'interval':>20}   verdict")
        for r in skills[skills.kind == kind].itertuples():
            n_eff = f"{r.n_eff:7.0f}" if r.n_eff == r.n_eff else "    n/a"
            small = "  (n_eff under 1,000)" if not r.n_eff >= 1000 else ""
            print(f"  {r.model:19} {r.reference:19} {r.metric:9} {n_eff} {100 * r.skill:+7.2f}% {_iv(r.lo, r.hi):>20}   {r.verdict}{small}")
    print()


def print_verdicts(verdicts):
    print("DOES FLOW HELP? (pre-registered rule: both metrics better than the flow-free counterpart AND not worse than the market)")
    for r in verdicts.itertuples():
        print(f"  {r.model:19} vs {r.counterpart:19}: {r.verdict}")
    if not verdicts.helps.any():
        print("BINANCE.US TRADE FLOW DOES NOT IMPROVE THE FORECAST: no flow model passes the rule.")
    print()


def print_by_horizon(pred):
    print("HELD-OUT per horizon (descriptive: no interval, not in the family, nothing is claimed from it):")
    print(f"  {'forecast':19} " + " ".join(f"{f'{h} min Brier':>13} {'log loss':>9} {'n':>6}" for h in ks.HORIZONS_MIN))
    for name in fl.BASELINES + fl.MODELS:
        parts = []
        for h in ks.HORIZONS_MIN:
            g = pred[pred.horizon == h]
            parts.append(f"{ks.brier(g[name], g.outcome).mean():13.5f} {ks.log_loss(g[name], g.outcome).mean():9.5f} {len(g):6d}"
                         if len(g) else f"{'n/a':>13} {'n/a':>9} {0:6d}")
        print(f"  {name:19} " + " ".join(parts))
    print()


def print_reliability(pred, z):
    print(f"HELD-OUT reliability: mean forecast vs YES rate [+/- {z:.2f} SE day-block interval] per bucket. "
          f"'small' = n_eff under {fc.MIN_BUCKET_N_EFF}. Descriptive.")
    for name in fl.BASELINES + fl.MODELS:
        print(f"  {name}")
        for r in fc.reliability_table(pred[name], pred["outcome"], pred["day"], z).itertuples():
            n_eff = f"{r.n_eff:6.0f}" if r.n_eff == r.n_eff else "   n/a"
            iv = f"[{r.lo:.3f}, {r.hi:.3f}]" if r.lo == r.lo else "[  n/a        ]"
            print(f"    {r.bucket:8} n {r.n:6d} n_eff {n_eff}  forecast {r.forecast:.3f}  YES rate {r.yes_rate:.3f} {iv}"
                  f"{'  small' if r.small else ''}")
    print()


def print_coefficients(fitted):
    c = fl.flow_coefficients(fitted)
    print("Fitted flow coefficients (standardized inputs, logit units per sd; train fit; descriptive, no interval):")
    for m in fl.FLOW_MODELS:
        g = c[c.model == m]
        print(f"  {m}")
        for inp in dict.fromkeys(g.input):
            parts = [f"{h:2d} min {g[(g.input == inp) & (g.horizon == h)].beta_std.iloc[0]:+.3f}" for h in ks.HORIZONS_MIN]
            print(f"    {inp:20} " + "   ".join(parts))
    print()


def print_side_by_side(pred):
    names = list(fl.BASELINES + fl.MODELS)
    print("Raw probabilities side by side (held-out), never combined by hand. Mean of each forecast per bucket of the market mid:")
    print(f"  {'mid bucket':10} {'n':>6} {'YES':>6} " + " ".join(f"{n[:11]:>11}" for n in names))
    b = ks.bucket_of(pred["market"])
    for i in range(len(ks.BUCKET_EDGES) - 1):
        g = pred[b == i]
        if len(g):
            print(f"  {ks.BUCKET_EDGES[i]:.1f}-{ks.BUCKET_EDGES[i + 1]:.1f}    {len(g):6d} {g.outcome.mean():6.3f} "
                  + " ".join(f"{g[n].mean():11.3f}" for n in names))
    print()


def print_summary(card, skills, verdicts, z, pristine):
    print(f"SUMMARY (held-out{'' if pristine else ', NOT pristine'}; {len(fl.MODELS)} fitted models, {fl.FAMILY_TESTS} skill "
          f"intervals, z = {z:.2f}). Skill columns are Brier skill.")
    print(f"  {'model':19} {'Brier':>8} {'log loss':>9} {'vs market':>9} {'interval':>20} {'vs flow-free':>12} {'interval':>20} {'cal a':>7} {'cal b':>6}")
    for m in fl.MODELS:
        c = card[card.forecast == m].iloc[0]
        mk = skills[(skills.model == m) & (skills.reference == "market") & (skills.metric == "brier")].iloc[0]
        cp = skills[(skills.model == m) & (skills.kind == "counterpart") & (skills.metric == "brier")]
        cp_txt = f"{100 * cp.iloc[0].skill:+11.2f}% {_iv(cp.iloc[0].lo, cp.iloc[0].hi):>20}" if len(cp) else f"{'(flow-free)':>12} {'':>20}"
        print(f"  {m:19} {c.brier:8.5f} {c.log_loss:9.5f} {100 * mk.skill:+8.2f}% {_iv(mk.lo, mk.hi):>20} {cp_txt} {c.cal_a:+7.3f} {c.cal_b:6.3f}")
    for name in fl.BASELINES:
        c = card[card.forecast == name].iloc[0]
        print(f"  {name + ' (baseline)':24} Brier {c.brier:.5f}  log loss {c.log_loss:.5f}")
    helped = verdicts[verdicts.helps]
    if helped.empty:
        print("BINANCE.US TRADE FLOW DOES NOT IMPROVE THE FORECAST: no flow model passes the rule.")
    else:
        print("FLOW HELPS: " + ", ".join(helped.model) + ". A better forecast is not a trading edge: no cost is in these numbers.")
    better = skills[(skills.reference == "market") & (skills.verdict == "BETTER")]
    if better.empty:
        print("NO MODEL HERE BEATS THE MARKET MID: no held-out skill interval against it lies above zero.")
    print()


def report(markets, candles, grid, coinbase, cache_dir):
    meta = fl.read_meta(cache_dir)
    pristine_after = meta.get("pristine_after")
    z = ks.bonferroni_z(fl.FAMILY_TESTS)
    print(f"{len(markets)} settled contracts, {len(candles)} with cached quotes; "
          f"{pd.to_datetime(markets[0]['close_ts'], unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(markets[-1]['close_ts'], unit='s'):%Y-%m-%d} UTC.")
    if "new_settled_contracts" not in meta:
        print("HELD-OUT: the existing held-out set, which earlier families have already been scored on -- NOT pristine. "
              "Whether newer contracts exist was not checked (run with --fetch).")
    elif pristine_after is None:
        print(f"HELD-OUT: the existing held-out set, which earlier families have already been scored on -- NOT pristine. "
              f"Only {meta['new_settled_contracts']} contract(s) had settled after the cache's last close when checked; a "
              f"pristine set needs {fl.PRISTINE_MIN_CONTRACTS}.")
    else:
        print(f"HELD-OUT: PRISTINE -- the {meta['new_settled_contracts']} contracts that settled after "
              f"{pd.to_datetime(pristine_after, unit='s'):%Y-%m-%d %H:%M} UTC, never scored before. Every earlier contract is train.")
    print(f"Family: {len(fl.MODELS)} fitted models ({len(fl.FLOW_MODELS)} with flow, each with its flow-free counterpart), "
          f"{fl.FAMILY_TESTS} skill intervals -> Bonferroni z = {z:.2f}.")
    print(LIMITS + "\n")

    start_ms, end_ms = trade_span_ms(markets)
    trades = fl.load_trades(cache_dir, start_ms, end_ms)
    print_trades(trades, start_ms, end_ms)
    d = fl.assign_splits(fl.add_flow(fc.build_dataset(markets, candles, grid, coinbase), trades, grid), pristine_after)
    test_rows = d[d.split == "test"]
    print(f"Held-out contracts close {pd.to_datetime(test_rows.close_ts.min(), unit='s'):%Y-%m-%d %H:%M} .. "
          f"{pd.to_datetime(test_rows.close_ts.max(), unit='s'):%Y-%m-%d %H:%M} UTC ({test_rows.ticker.nunique()} contracts).")
    print_coverage(d)
    if not (d.use & (d.split == "fit")).any() or not (d.use & (d.split == "test")).any():
        raise SystemExit("No scorable rows: the Binance.US trade cache does not cover the contracts. Run with --fetch first.")

    fitted = fl.tune_and_fit(d)                      # train only
    print_tuning(fitted)
    test = d[d.use & (d.split == "test")]            # scored once, after every choice above is final
    pred = fl.predictions(fitted, test)
    card = fl.scorecard(pred)
    skills = fl.skill_table(pred, z)
    verdicts = fl.flow_verdicts(skills)
    print_scorecard(card)
    print_skill(skills, z)
    print_verdicts(verdicts)
    print_by_horizon(pred)
    print_reliability(pred, z)
    print_coefficients(fitted)
    print_side_by_side(pred)
    print_summary(card, skills, verdicts, z, pristine_after is not None)
    path = os.path.join(cache_dir, "flow_predictions.csv")
    pred.to_csv(path, index=False)
    print(f"Per-row held-out probabilities saved to {path}")
    return {"dataset": d, "fitted": fitted, "predictions": pred, "scorecard": card, "skills": skills, "verdicts": verdicts,
            "trades": trades}


def main(argv=None, coinbase_fetch=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=ks.CACHE_DIR)
    ap.add_argument("--fetch", action="store_true", help="count new contracts and cache Binance.US trades, then stop")
    ap.add_argument("--kalshi-base-url", default=None, help="another host serving Kalshi's public API, for --fetch")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    if args.fetch:
        return fetch(args.cache_dir, args.kalshi_base_url, args.workers)

    def progress(i, n):
        if i % 50 == 0 or i == n:
            print(f"  coinbase day {i}/{n}", flush=True)
    markets, candles, grid, coinbase = _review_script("kalshi_forecast_backtest").load_inputs(args.cache_dir, coinbase_fetch, progress)
    return report(markets, candles, grid, coinbase, args.cache_dir)


if __name__ == "__main__":
    main()
