"""Run from the repo root.

    python review/kalshi_market_study.py fetch     # download + cache (resumable; ~10-20 min the first time)
    python review/kalshi_market_study.py step2     # is the Kalshi market itself mispriced?
    python review/kalshi_market_study.py step3     # does kalshi.settlement_probability beat it?

Steps 2 and 3 of the KXBTC15M edge study (REVIEW_FINDINGS.md, "Is there an edge?"). The design
-- split, horizons, buckets, costs, selection rule -- is fixed in kalshi_study.py and was
committed before any of this was run. Raw data is cached under study_data/ (gitignored).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests

import kalshi_study as ks


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "btc-indicator-study/1.0"
    return s


def load_markets(session=None, refresh=False):
    raw = ks.load_or_fetch_markets(session or _session(), refresh=refresh)
    parsed = [ks.parse_settled_market(m) for m in raw]
    return raw, [m for m in parsed if m is not None]


def cmd_fetch(refresh=False):
    s = _session()
    raw, markets = load_markets(s, refresh)
    print(f"{len(raw)} settled markets listed, {len(markets)} settled yes/no; "
          f"{pd.to_datetime(markets[0]['close_ts'], unit='s')} .. {pd.to_datetime(markets[-1]['close_ts'], unit='s')} UTC")

    def progress(i, n):
        if i % 1000 == 0 or i == n:
            print(f"  candles for {i}/{n} markets", flush=True)
    got = ks.fetch_missing_candles(s, markets, progress=progress, historical_before=ks.historical_cutoff(s),
                                   workers=4, session_factory=_session)
    print(f"fetched candles for {got} markets; cache holds {len(ks.load_candles())}")


def build_observations():
    _, markets = load_markets()
    raw = ks.load_candles()
    candles = {t: ks.parse_candles(c) for t, c in raw.items()}
    obs = ks.observations(markets, candles)
    cutoff = ks.split_cutoff([m["close_ts"] for m in markets])
    obs["split"] = np.where(obs["close_ts"] < cutoff, "train", "test")
    return markets, obs, cutoff


def _pct(x):
    return f"{100 * x:+.1f}c"


FILL_NOTE = ("Fills are assumed at the quoted top of book (YES at the ask, NO at 1 - bid). Kalshi's "
             "candlesticks carry no depth, so the size really available there may be small -- "
             "especially for 100-contract orders.")


def family_z():
    return ks.bonferroni_z(ks.TOTAL_TESTS)


def print_bucket_table(t):
    print(f"  {'bucket':9} {'n':>5} {'n_eff':>6} {'mid':>6} {'sprd':>5} {'YES rate [CI]':>20}   "
          f"{'EV buy YES [CI]':>24} {'MDE':>6}   {'EV buy NO [CI]':>24} {'MDE':>6}")
    for r in t.itertuples():
        print(f"  {r.bucket:9} {r.n:5d} {r.n_eff:6.0f} {r.mid:6.3f} {100 * r.spread:4.1f}c "
              f"{r.yes_rate:6.3f} [{r.yes_lo:5.3f},{r.yes_hi:5.3f}]   "
              f"{_pct(r.ev_yes):>7} [{_pct(r.ev_yes_lo):>7},{_pct(r.ev_yes_hi):>7}] {100 * r.ev_yes_mde:5.1f}c   "
              f"{_pct(r.ev_no):>7} [{_pct(r.ev_no_lo):>7},{_pct(r.ev_no_hi):>7}] {100 * r.ev_no_mde:5.1f}c")


def print_exclusions(obs):
    print("Excluded contract-rows by month (no candle at that exact minute, or not a two-sided book), "
          "and the exclusion rate among YES- vs NO-settled contracts:")
    t = ks.exclusions(obs)
    wide = t.pivot(index="month", columns="horizon")
    cols = []
    for h in ks.HORIZONS_MIN:
        cols += [("contracts", h), ("no_quote", h), ("bad_book", h), ("excl_rate_yes", h), ("excl_rate_no", h)]
    out = wide[cols]
    out.columns = [f"{c}@{h}" for c, h in cols]
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))
    z = family_z()
    print(f"\nOutcome dependence of exclusion (YES rate - NO rate, day-block SE; |gap/SE| > {z:.2f} "
          f"would be significant at the family level):")
    for split in ("train", "test", "all"):
        for h in ks.HORIZONS_MIN:
            d = obs[obs.horizon == h] if split == "all" else obs[(obs.horizon == h) & (obs.split == split)]
            g = ks.exclusion_outcome_gap(d)
            print(f"  {split:5} h={h:2d}: excluded {g['rate_yes']:.4f} of YES, {g['rate_no']:.4f} of NO, "
                  f"gap {g['gap']:+.4f} +/- {g['se']:.4f} (z {g['gap'] / g['se']:+.2f})")
    print()


def cmd_step2():
    markets, obs, cutoff = build_observations()
    z = family_z()
    print(f"{len(markets)} settled contracts, {pd.to_datetime(markets[0]['close_ts'], unit='s')} .. "
          f"{pd.to_datetime(markets[-1]['close_ts'], unit='s')} UTC")
    print(f"held out from {pd.to_datetime(cutoff, unit='s')} UTC "
          f"({(obs.drop_duplicates('ticker').close_ts >= cutoff).sum()} contracts)")
    print(f"Tests across Steps 2 and 3: {ks.STEP2_TESTS} (step 2: horizon x bucket x side x order size) + "
          f"{ks.STEP3_TRADING_TESTS} (step 3 trading) + {ks.STEP3_SCORING_TESTS} (step 3 scoring) = "
          f"{ks.TOTAL_TESTS}; Bonferroni z = {z:.2f} on every interval below")
    print(f"Order sizes reported: {ks.ORDER_SIZES} contracts; the fee rounds up to the cent per order.")
    print(f"Buckets: fixed 10-cent bands of the mid, {ks.BUCKET_EDGES}, identical on train and test. "
          f"MDE = z x SE, the smallest edge per contract the bucket could detect.")
    print(FILL_NOTE + "\n")

    print_exclusions(obs)

    v5 = obs[(obs.horizon == 5) & obs.valid].copy()
    v5["month"] = v5["day"].str[:7]
    sp = v5.groupby("month").apply(lambda g: pd.Series({
        "n": len(g), "median_spread_c": 100 * (g.ask - g.bid).median(),
        "share_spread_le_2c": ((g.ask - g.bid) <= 0.0201).mean()}), include_groups=False)
    print("Book quality by month (5 min before close, valid books):")
    print(sp.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    for split in ("train", "test"):
        for h in ks.HORIZONS_MIN:
            v = obs[(obs.split == split) & (obs.horizon == h) & obs.valid]
            cal = ks.calibration_bootstrap(v["mid"], v["outcome"], v["day"])
            br = ks.block_bootstrap(ks.brier(v["mid"], v["outcome"]), v["day"])
            print(f"== {split.upper()}  {h} min before close: n={len(v)}, days={v.day.nunique()}; "
                  f"calibration logit slope b = {cal['b']:.3f} +/- {cal['b_se']:.3f} (1 = calibrated, "
                  f">1 = favorite-longshot bias), intercept a = {cal['a']:+.3f} +/- {cal['a_se']:.3f}; "
                  f"Brier(mid) = {br['mean']:.4f}")
    print()

    tables = {}
    for c in ks.ORDER_SIZES:
        for split in ("train", "test"):
            for h in ks.HORIZONS_MIN:
                d = obs[(obs.split == split) & (obs.horizon == h)]
                tables[(c, split, h)] = ks.bucket_table(d, z, c)
                print(f"== {c}-contract orders, {split.upper()}, {h} min before close "
                      f"(EV in cents per contract, net of spread and fee)")
                print_bucket_table(tables[(c, split, h)])
                print()

    # Pre-registered selection: train interval entirely above zero, per order size.
    selected = [(c, h, r.bucket, side)
                for c in ks.ORDER_SIZES for h in ks.HORIZONS_MIN
                for r in tables[(c, "train", h)].itertuples()
                for side in ("yes", "no") if getattr(r, f"ev_{side}_lo") > 0]
    print(f"Selected on train (family interval above zero): {selected or 'none'}")
    if selected:
        z_test = ks.bonferroni_z(len(selected))
        for c, h, bucket, side in selected:
            r = tables[(c, "test", h)].set_index("bucket")
            if bucket not in r.index:
                print(f"  TEST {c}x h={h} {bucket} {side.upper()}: no test rows")
                continue
            row = r.loc[bucket]
            se = (row[f"ev_{side}_hi"] - row[f"ev_{side}"]) / z
            print(f"  TEST {c}x h={h} {bucket} buy {side.upper()}: n={row['n']} mean {_pct(row[f'ev_{side}'])} "
                  f"[{_pct(row[f'ev_{side}'] - z_test * se)}, {_pct(row[f'ev_{side}'] + z_test * se)}] "
                  f"(z={z_test:.2f})")


def _dist(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (f"n={len(x)} mean {x.mean():+.2f} median {np.median(x):+.2f} sd {x.std():.2f} "
            f"p5 {np.percentile(x, 5):+.2f} p95 {np.percentile(x, 95):+.2f} mean|.| {np.abs(x).mean():.2f}")


def cmd_step3():
    markets, obs, cutoff = build_observations()
    cache = os.path.join(ks.CACHE_DIR, "model_obs.csv")
    if os.path.exists(cache):
        mo = pd.read_csv(cache)
    else:
        def progress(i, n):
            if i % 20 == 0 or i == n:
                print(f"  coinbase day {i}/{n}", flush=True)
        cb = ks.load_or_fetch_coinbase(markets[0]["open_ts"] - 4 * 3600, markets[-1]["close_ts"],
                                       progress=progress)
        grid = ks.MinuteGrid(cb)
        print(f"coinbase: {len(cb)} candles, {len(grid.cols['close']) - len(cb)} missing minutes in the span")
        mo = ks.model_observations(obs, grid)
        mo.to_csv(cache, index=False)
    mo["split"] = np.where(mo["close_ts"] < cutoff, "train", "test")
    z = family_z()
    print(f"Bonferroni z = {z:.2f} over all {ks.TOTAL_TESTS} tests in Steps 2 and 3.")
    print(FILL_NOTE + "\n")

    # --- which valid rows the model could not price, and whether that depends on the outcome ---
    for h in ks.HORIZONS_MIN:
        d = mo[(mo.horizon == h) & mo.valid].copy()
        d["valid"] = d["p_model_adj"].notna()           # "excluded" = model could not price it
        g = ks.exclusion_outcome_gap(d)
        print(f"h={h:2d}: {len(d)} valid quotes, model unpriced {(~d.valid).sum()} "
              f"(YES {g['rate_yes']:.4f}, NO {g['rate_no']:.4f}, gap z {g['gap'] / g['se']:+.2f})")
    print()

    # --- basis: Coinbase vs the settlement index ---
    per = mo.drop_duplicates("ticker")
    print("Coinbase-vs-index basis, bp (Kalshi's 60s BRTI average / Coinbase OHLC-mean of that minute - 1):")
    for split in ("train", "test"):
        d = per[per.split == split]
        print(f"  {split:5} at open : {_dist(d.basis_open_bp)}")
        print(f"  {split:5} at close: {_dist(d.basis_close_bp)}")
    proxy_close = per.expiration_value / (1 + per.basis_close_bp / 1e4)
    proxy_open = per.strike / (1 + per.basis_open_bp / 1e4)
    ok = per.basis_close_bp.notna() & per.basis_open_bp.notna()
    raw_call = (proxy_close >= per.strike)[ok] == per.outcome[ok].astype(bool)
    rel_call = (proxy_close >= proxy_open)[ok] == per.outcome[ok].astype(bool)
    print(f"  knowing Coinbase's final minute exactly, vs the strike: right {raw_call.mean():.4f} of {ok.sum()}; "
          f"Coinbase close-minute vs Coinbase open-minute: right {rel_call.mean():.4f}")
    print(f"  basis drift open->close within a contract: {_dist(per.basis_close_bp - per.basis_open_bp)}\n")

    # --- scoring against the market mid ---
    print(f"Scoring vs market mid on the same rows (valid book and model priced); paired difference "
          f"model - market, negative = model better; z = {z:.2f}")
    for split in ("train", "test"):
        for h in ks.HORIZONS_MIN:
            d = mo[(mo.split == split) & (mo.horizon == h) & mo.valid & mo.p_model.notna() & mo.p_model_adj.notna()]
            line = f"  {split:5} h={h:2d} n={len(d):5d}  Brier mkt {ks.brier(d.mid, d.outcome).mean():.4f}"
            for col in ("p_model", "p_model_adj"):
                line += f" {col[2:]} {ks.brier(d[col], d.outcome).mean():.4f}"
            print(line)
            for metric, fn in (("Brier", ks.brier), ("logloss", ks.log_loss)):
                parts = []
                for col in ("p_model", "p_model_adj"):
                    s = ks.block_bootstrap(fn(d[col], d.outcome) - fn(d.mid, d.outcome), d.day)
                    lo, hi = ks.interval(s, z)
                    parts.append(f"{col[2:]:9} {s['mean']:+.4f} [{lo:+.4f},{hi:+.4f}]")
                print(f"      {metric:7} diff: " + "   ".join(parts))
            cal = ks.calibration_fit(d.p_model_adj, d.outcome)
            print(f"      model_adj calibration a={cal[0]:+.3f} b={cal[1]:.3f};  mean |model_adj - mid| "
                  f"{(d.p_model_adj - d.mid).abs().mean():.3f}")
    print()

    # --- trading toward the model ---
    variants = [(c, src, m, h) for c in ks.ORDER_SIZES for src in ks.SPOT_SOURCES
                for m in ks.EDGE_MARGINS for h in ks.HORIZONS_MIN]
    col = {"coinbase": "p_model", "basis_adjusted": "p_model_adj"}
    results = {}
    for split in ("train", "test"):
        print(f"Trading toward the model, {split.upper()} (profit per contract, z = {z:.2f}):")
        for c, src, m, h in variants:
            d = mo[(mo.split == split) & (mo.horizon == h) & mo.valid & mo[col[src]].notna()]
            side, pnl = ks.model_trades(d[col[src]], d.bid, d.ask, d.outcome, m, c)
            traded = side != 0
            s = ks.block_bootstrap(pnl[traded], d.day.to_numpy()[traded])
            lo, hi = ks.interval(s, z) if s["n"] > 1 else (np.nan, np.nan)
            results[(split, c, src, m, h)] = s
            print(f"  {c:3d}x {src:14} margin {m:.2f} h={h:2d}: {s['n']:5d} trades of {len(d):5d} "
                  f"(YES {np.sum(side == 1):5d})  mean {_pct(s['mean'])} [{_pct(lo)}, {_pct(hi)}]  "
                  f"MDE {100 * z * s['se']:.1f}c")
        print()
    selected = [v for v in variants
                if results[("train",) + v]["n"] > 1
                and results[("train",) + v]["mean"] - z * results[("train",) + v]["se"] > 0]
    print(f"Selected on train (family interval above zero): {selected or 'none'}")
    if selected:
        z_t = ks.bonferroni_z(len(selected))
        for v in selected:
            s = results[("test",) + v]
            print(f"  TEST {v}: n={s['n']} mean {_pct(s['mean'])} "
                  f"[{_pct(s['mean'] - z_t * s['se'])}, {_pct(s['mean'] + z_t * s['se'])}] (z={z_t:.2f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step2"
    if cmd == "fetch":
        cmd_fetch(refresh="--refresh" in sys.argv)
    else:
        globals()[f"cmd_{cmd}"]()
