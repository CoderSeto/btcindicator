"""Run from the repo root: python review/backtest_overlap.py [runs=60]   (~3s per run)

How much does backtest.py's stated n overstate the evidence?
Runs the REAL run_backtest() on many independent driftless random walks (no signal has any edge
by construction, true hit rate = 50%) and compares the observed spread of each bucket's hit rate
across runs with the binomial spread sqrt(0.25/n) that taking n at face value implies.
design effect = (observed var) / (binomial var) ~= how many stated samples equal one independent one.
"""
import json, os, sys, tempfile
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import backtest, config
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
tmp = tempfile.mkdtemp()
cfg = SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
cfg.BACKTEST_RESULTS_PATH = os.path.join(tmp, "bt.json")
import weighting
rates, ns, neffs, offs = {}, {}, {}, {}
for r in range(runs):
    rng = np.random.default_rng(1000 + r)
    n = cfg.BACKTEST_HISTORY_CANDLES
    close = 75_000 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    openp = np.r_[close[0], close[:-1]]
    vol = rng.lognormal(0, 0.6, n)
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="min"),
                       "low": np.minimum(openp, close), "high": np.maximum(openp, close),
                       "open": openp, "close": close, "volume": vol})
    backtest.fetch_history = lambda *a, **k: df.copy()
    res = backtest.run_backtest(cfg)
    for group in ("indicator_buckets", "strength_buckets", "divergence_buckets"):
        for k, b in res[group].items():
            if b["total"] >= 30:
                rates.setdefault(k, []).append(b["hits"] / b["total"])
                ns.setdefault(k, []).append(b["total"])
                neffs.setdefault(k, []).append(b.get("n_eff", b["total"]))
                if group == "indicator_buckets":
                    offs.setdefault(k, []).append(weighting._weight_from_bucket(b, cfg) != 1.0)
print(f"{runs} independent {cfg.BACKTEST_HISTORY_CANDLES}-candle random walks, "
      f"lookahead={cfg.BACKTEST_LOOKAHEAD_CANDLES}. No signal has any edge by construction.\n")
print("eff. n (empirical) = mean n / design effect, from the spread of the hit rate across runs.")
print("n_eff (estimated)  = the mean of what backtest.effective_sample_size() reported per run.")
print("weights != 1       = share of runs whose bucket got a weight off 1.0 (false positives).\n")
print(f"{'bucket':20s} {'mean n':>7} {'obs sd':>7} {'binom sd':>8} {'design eff':>10} "
      f"{'eff n emp':>9} {'n_eff est':>9} {'weights != 1':>12}")
des = []
for k in sorted(rates, key=lambda k: -np.mean(ns[k])):
    if len(rates[k]) < runs * 0.8:
        continue
    mn = np.mean(ns[k]); obs = np.std(rates[k], ddof=1); bin_sd = np.sqrt(0.25 / mn)
    de = (obs / bin_sd) ** 2
    des.append(de)
    off = f"{np.mean(offs[k]) * 100:10.0f}%" if k in offs else f"{'--':>11}"
    print(f"{k:20s} {mn:7.0f} {obs:7.3f} {bin_sd:8.3f} {de:10.1f} {mn / de:9.0f} {np.mean(neffs[k]):9.0f} {off}")
print(f"\nmedian design effect: {np.median(des):.1f}")
