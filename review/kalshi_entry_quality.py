"""
Entry-quality survey for KXBTC15M (Part 1a of the smart-decisions work).

Pre-cutoff rows only (edge_hunt.load_exploration). For each market state: how
much an entry at the touch costs (half-spread + per-order fee, in cents), and
how well the market mid is calibrated there (outcome - mid, day-block CI;
Brier of the mid). No direction is claimed. Prints <= 30 lines.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edge_hunt  # noqa: E402
import kalshi_study as ks  # noqa: E402


def cost_c(df, contracts=1):
    half = (df.ask - df.bid) / 2
    fee_yes = ks.fee_per_contract(df.ask, contracts)
    fee_no = ks.fee_per_contract(1 - df.bid, contracts)
    return 100 * (half + (fee_yes + fee_no) / 2)


def row(name, d):
    if len(d) == 0:
        return f"{name:<22} n=0"
    gap = ks.block_bootstrap((d.outcome - d.mid).to_numpy(), d.day.to_numpy())
    lo, hi = ks.interval(gap, 1.96)
    br = float(ks.brier(d.mid, d.outcome).mean())
    return (f"{name:<22} n={gap['n']:>6} n_eff={gap['n_eff']:>7.0f} cost={cost_c(d).mean():5.2f}c "
            f"spread={100 * (d.ask - d.bid).mean():5.2f}c  out-mid={100 * gap['mean']:+5.2f}c "
            f"[{100 * lo:+5.2f},{100 * hi:+5.2f}] brier={br:.4f}")


def main():
    obs = edge_hunt.load_exploration()
    v = obs[obs.valid & obs.mid.notna()].copy()
    v["spread"] = v.ask - v.bid
    v["stale"] = (v["last"] - v.mid).abs()
    print(f"valid rows {len(v)} of {len(obs)}; brier(mid) all={float(ks.brier(v.mid, v.outcome).mean()):.4f} "
          f"brier(0.5)={float(ks.brier(np.full(len(v), 0.5), v.outcome).mean()):.4f}")
    print(row("all valid", v))
    for lo_, hi_ in ((0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.21)):
        print(row(f"spread {lo_*100:.0f}-{hi_*100:.0f}c", v[(v.spread > lo_ - 1e-9) & (v.spread <= hi_)]))
    for h in (10, 5, 2):
        print(row(f"horizon {h}m", v[v.horizon == h]))
    for lo_, hi_ in ((0, 0.1), (0.1, 0.25), (0.25, 0.75), (0.75, 0.9), (0.9, 1.0)):
        print(row(f"mid {lo_*100:.0f}-{hi_*100:.0f}c", v[(v.mid > lo_) & (v.mid <= hi_)]))
    s = v[v["last"].notna()]
    print(row("last-mid <= 5c", s[s.stale <= 0.05]))
    print(row("last-mid > 5c", s[s.stale > 0.05]))
    print(row("no last", v[v["last"].isna()]))
    b = v[v.basis_open_bp.notna()]
    print(row("|basis| <= 10bp", b[b.basis_open_bp.abs() <= 10]))
    print(row("|basis| > 10bp", b[b.basis_open_bp.abs() > 10]))
    q = v.vol.quantile([0.25, 0.75])
    print(row("vol low quartile", v[v.vol <= q.iloc[0]]))
    print(row("vol high quartile", v[v.vol >= q.iloc[1]]))
    # cheaper to wait? paired spread change within a market, 10 -> 5 -> 2 minutes
    w = v.pivot_table(index="close_ts", columns="horizon", values="spread")
    for a, b_ in ((10, 5), (5, 2)):
        p = w[[a, b_]].dropna()
        d = 100 * (p[b_] - p[a])
        print(f"wait {a}->{b_}m: n={len(p)} spread change mean={d.mean():+.2f}c  narrower in {(d < 0).mean():.0%}, "
              f"wider in {(d > 0).mean():.0%}")
    print(f"invalid share by horizon: {obs.groupby('horizon').valid.apply(lambda x: f'{1 - x.mean():.1%}').to_dict()}")


if __name__ == "__main__":
    main()
