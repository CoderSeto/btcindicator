"""
Decision-quality survey for KXBTC15M (DECISION_RULES.md). Pre-cutoff rows only.

Sections (python review/kalshi_decision_rules.py <section>), each <= 30 lines:
  timing  -- per state, the cost of entering now vs at the next cached snapshot
             (paired within a market), and how many books are still tradable then.
  exit    -- for a position opened at a snapshot, hold-to-settle vs selling back at
             the next snapshot's real bid (fee paid again), by state.
  regime  -- time of day, vol regime, basis, streaks: cost, mid calibration, Brier.
  features-- does anything in the cache explain outcome - mid? OLS with day-block CIs.
  peek    -- three fixed sides (favourite, underdog, model) in every state above,
             each logged to EDGE_TRIALS.csv; a candidate needs CI lower > 0 at
             Bonferroni z over every row of that file.
Fills: YES at ask, NO at 1 - bid, kalshi_study.order_fee, one entry per market.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import csv_utils  # noqa: E402
import edge_hunt  # noqa: E402
import kalshi_study as ks  # noqa: E402

_spec = importlib.util.spec_from_file_location("kalshi_edge_hunt", os.path.join(ROOT, "review", "kalshi_edge_hunt.py"))
eh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eh)

BANDS = ((0, .1), (.1, .25), (.25, .75), (.75, .9), (.9, 1.0))
SPREADS = ((0, .02), (.02, .05), (.05, .21))


def ci(values, days, z=1.96):
    s = ks.block_bootstrap(np.asarray(values, float), np.asarray(days))
    lo, hi = ks.interval(s, z) if s["n"] > 1 else (np.nan, np.nan)
    return s, lo, hi


def fmt(name, s, lo, hi, extra=""):
    return f"{name:<26} n={s['n']:>6} n_eff={s['n_eff']:>7.0f} mean={100 * s['mean']:+6.2f}c [{100 * lo:+6.2f},{100 * hi:+6.2f}] {extra}"


def load():
    d = eh.features(edge_hunt.load_exploration())
    d["cost"] = (d.ask - d.bid) / 2 + (ks.fee_per_contract(d.ask, 1) + ks.fee_per_contract(1 - d.bid, 1)) / 2
    d["band"] = pd.cut(d.mid, [b[0] for b in BANDS] + [1.0], labels=[f"{int(a*100)}-{int(b*100)}c" for a, b in BANDS],
                       include_lowest=True).astype(str)
    d["sband"] = pd.cut(d.spread, [0, .02, .05, .21], labels=["<2c", "2-5c", "5c+"], include_lowest=True).astype(str)
    d["hour"] = pd.to_datetime(d.close_ts, unit="s", utc=True).dt.hour
    d["hour4"] = (d.hour // 4 * 4).astype(str) + "h"
    q = d.vol.quantile([.25, .75])
    d["volq"] = np.where(d.vol <= q.iloc[0], "low", np.where(d.vol >= q.iloc[1], "high", "mid"))
    return d


def wide(d):
    """One row per market with bid/ask/valid/cost at each horizon."""
    v = d[["close_ts", "horizon", "bid", "ask", "valid", "cost", "mid", "outcome", "day", "band", "sband", "volq"]]
    w = v.pivot_table(index="close_ts", columns="horizon", values=["bid", "ask", "valid", "cost", "mid"], aggfunc="first")
    w.columns = [f"{a}{b}" for a, b in w.columns]
    first = v.sort_values("horizon", ascending=False).drop_duplicates("close_ts").set_index("close_ts")
    return w.join(first[["outcome", "day", "band", "sband", "volq"]])


def timing():
    d = load()
    w = wide(d)
    print("entering now vs at the next snapshot (paired, valid-now markets): cost change, mid move, share tradable later")
    for h, nx in ((10, 5), (5, 2)):
        for key, groups in (("sband", ["<2c", "2-5c", "5c+"]), ("band", [f"{int(a*100)}-{int(b*100)}c" for a, b in BANDS]),
                            ("volq", ["low", "mid", "high"])):
            sub = w[w[f"valid{h}"] == True]  # noqa: E712
            lab = d[d.horizon == h].drop_duplicates("close_ts").set_index("close_ts")[key]
            sub = sub.assign(k=lab.reindex(sub.index))
            for g in groups:
                m = sub[sub.k == g]
                if len(m) < 100:
                    continue
                later = m[f"valid{nx}"] == True  # noqa: E712
                dc = 100 * (m.loc[later, f"cost{nx}"] - m.loc[later, f"cost{h}"])
                s, lo, hi = ci(dc.to_numpy() / 100, m.loc[later, "day"].to_numpy())
                mv = 100 * (m.loc[later, f"mid{nx}"] - m.loc[later, f"mid{h}"]).abs().mean()
                print(f"{h}->{nx}m {key}={g:<8} n={len(m):>6} tradable_later={later.mean():5.1%} "
                      f"cost_change={100 * s['mean']:+5.2f}c [{100 * lo:+5.2f},{100 * hi:+5.2f}] |mid move|={mv:4.1f}c "
                      f"cost_now={100 * m[f'cost{h}'].mean():4.2f}c")


def exit_rules():
    d = load()
    w = wide(d)
    print("sell-back at the next snapshot's real bid (fee again) minus hold-to-settle, per contract; both sides entered")
    for h, nx in ((10, 5), (5, 2), (10, 2)):
        ok = (w[f"valid{h}"] == True) & (w[f"valid{nx}"] == True)  # noqa: E712
        m = w[ok].copy()
        # YES bought at ask_h: hold pays outcome; sale pays bid_nx - fee. NO bought at 1 - bid_h: sale pays 1 - ask_nx - fee.
        m["dy"] = (m[f"bid{nx}"] - ks.fee_per_contract(m[f"bid{nx}"], 1)) - m.outcome
        m["dn"] = ((1 - m[f"ask{nx}"]) - ks.fee_per_contract(1 - m[f"ask{nx}"], 1)) - (1 - m.outcome)
        m["moved_for_yes"] = np.where(m[f"mid{nx}"] - m[f"mid{h}"] >= 0.05, "up5+", np.where(m[f"mid{nx}"] - m[f"mid{h}"] <= -0.05, "down5+", "flat"))
        rows = [("all, YES held", m.dy, m), ("all, NO held", m.dn, m)]
        for b in m.band.unique():
            rows.append((f"band {b}, YES", m[m.band == b].dy, m[m.band == b]))
        for mv in ("up5+", "down5+", "flat"):
            rows.append((f"mid {mv} since entry, YES", m[m.moved_for_yes == mv].dy, m[m.moved_for_yes == mv]))
            rows.append((f"mid {mv} since entry, NO", m[m.moved_for_yes == mv].dn, m[m.moved_for_yes == mv]))
        for name, vals, sub in rows:
            if len(sub) < 100:
                continue
            s, lo, hi = ci(vals.to_numpy(), sub.day.to_numpy())
            print(f"{h}->{nx}m " + fmt(name, s, lo, hi))
        if h == 10 and nx == 5:
            print("---")


def regime():
    d = load()
    v = d[d.valid]
    print("state: cost per entry, outcome - mid (day-block 95% CI), Brier(mid), share of books untradable")
    for key in ("hour4", "volq", "sband"):
        for g in sorted(v[key].unique()):
            sub = v[v[key] == g]
            s, lo, hi = ci((sub.outcome - sub.mid).to_numpy(), sub.day.to_numpy())
            inval = 1 - d[d[key] == g].valid.mean() if key != "sband" else np.nan
            print(fmt(f"{key}={g}", s, lo, hi, f"cost={100 * sub.cost.mean():4.2f}c brier={float(ks.brier(sub.mid, sub.outcome).mean()):.4f}"
                      + (f" untradable={inval:.1%}" if np.isfinite(inval) else "")))
    for name, mask in (("|basis|<2bp", v.basis_open_bp.abs() < 2), ("|basis| 2-10bp", (v.basis_open_bp.abs() >= 2) & (v.basis_open_bp.abs() <= 10)),
                       ("|basis|>10bp", v.basis_open_bp.abs() > 10), ("streak3 none", v.streak3 == 0), ("streak3 up", v.streak3 == 1),
                       ("streak3 down", v.streak3 == -1)):
        sub = v[mask]
        s, lo, hi = ci((sub.outcome - sub.mid).to_numpy(), sub.day.to_numpy())
        print(fmt(name, s, lo, hi, f"cost={100 * sub.cost.mean():4.2f}c brier={float(ks.brier(sub.mid, sub.outcome).mean()):.4f}"))
    print(f"streak3 continuation: outcome==streak side share = "
          f"{(v[v.streak3 != 0].outcome == (v[v.streak3 != 0].streak3 == 1)).mean():.3f} vs mid-implied "
          f"{np.where(v[v.streak3 != 0].streak3 == 1, v[v.streak3 != 0].mid, 1 - v[v.streak3 != 0].mid).mean():.3f}")


def features():
    d = load()
    cols = ["model_gap", "adj_gap", "dist", "spread", "vol", "ret5_v", "range15_v", "basis_open_bp", "hour_sin", "hour_cos",
            "weekend", "streak3", "prev_out", "h10", "h2"]
    v = d[d.valid].dropna(subset=cols).copy()
    y = (v.outcome - v.mid).to_numpy()
    X = v[cols].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    Xc = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    days = v.day.to_numpy()
    labels, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(ks.BOOTSTRAP_SEED)
    boots = []
    for _ in range(300):
        pick = rng.integers(0, len(labels), len(labels))
        idx = np.concatenate([np.flatnonzero(inv == k) for k in pick])
        boots.append(np.linalg.lstsq(Xc[idx], y[idx], rcond=None)[0])
    se = np.std(boots, axis=0, ddof=1)
    z = ks.bonferroni_z(len(cols))
    r2 = 1 - ((y - Xc @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"outcome - mid ~ z-scored features, n={len(v)} days={len(labels)} R2={r2:.4f}; coef in cents per 1 SD, "
          f"day-block SE, Bonferroni z={z:.2f} over {len(cols)}")
    for c, b, s in zip(cols, beta[1:], se[1:]):
        print(f"{c:<14} {100 * b:+6.3f}c  se {100 * s:5.3f}  z {b / s:+5.2f}  {'EXPLAINS' if abs(b / s) > z else 'no'}")
    print("composite/strength: not in the cache and signal_log.csv is absent on this machine -- untestable here")


def peek():
    d = load()
    v = d[d.valid & d.p_model.notna()].copy()
    v["fav"] = np.where(v.mid >= 0.5, 1, -1)
    v["model"] = np.where(v.p_model - v.ask - ks.fee_per_contract(v.ask, 1) > 0, 1,
                          np.where((1 - v.p_model) - (1 - v.bid) - ks.fee_per_contract(1 - v.bid, 1) > 0, -1, 0))
    def pnl(side):
        return np.where(side == 1, ks.ev_buy_yes(v.outcome, v.ask, 1), np.where(side == -1, ks.ev_buy_no(v.outcome, v.bid, 1), np.nan))
    v["p_fav"], v["p_dog"], v["p_model_side"] = pnl(v.fav), pnl(-v.fav), pnl(v.model)
    states = [("band", b) for b in v.band.unique()] + [("hour4", h) for h in v.hour4.unique()] + \
             [("volq", q) for q in ("low", "mid", "high")] + [("sband", s) for s in ("<2c", "2-5c", "5c+")]
    if len(sys.argv) > 2 and sys.argv[2] == "streak":            # second pass: the streak states the regime section flagged
        v["streak"] = v.streak3.map({0: "none", 1: "up", -1: "down"})
        states = [("streak", s) for s in ("none", "up", "down")]
    stamp = int(time.time())
    out = []
    for key, g in states:
        for h in (10, 5, 2):
            sub = v[(v[key] == g) & (v.horizon == h)]
            for side in ("p_fav", "p_dog", "p_model_side"):
                vals = sub[side].to_numpy()
                keep = np.isfinite(vals)
                s = ks.block_bootstrap(vals[keep], sub.day.to_numpy()[keep])
                tid = f"peek_{key}={g}_h{h}_{side[2:]}"
                csv_utils.append_row(eh.TRIALS_PATH, eh.TRIALS_HEADER,
                                     [tid, "peek", f"state={key}={g};horizon={h};side={side[2:]}", s["n"], round(s["n_eff"], 1),
                                      round(100 * s["mean"], 3) if s["n"] else "", round(100 * s["se"], 3) if s["n"] > 1 else "",
                                      "", "", "", "", stamp])
                out.append((tid, s))
    z = ks.bonferroni_z(len(pd.read_csv(eh.TRIALS_PATH)))
    best = sorted([(s["mean"] - z * s["se"], tid, s) for tid, s in out if s["n"] > 1 and np.isfinite(s["se"])], reverse=True)
    print(f"peek: {len(out)} state x side cells logged; Bonferroni z={z:.2f} over all {len(pd.read_csv(eh.TRIALS_PATH))} trials")
    for lo, tid, s in best[:8]:
        print(f"{tid:<40} n={s['n']:>5} n_eff={s['n_eff']:>6.0f} mean={100 * s['mean']:+6.2f}c lo={100 * lo:+6.2f}c "
              f"{'CANDIDATE' if lo > 0 and s['n_eff'] >= 500 else '-'}")
    print("peek: " + ("candidate " + best[0][1] if best and best[0][0] > 0 and best[0][2]["n_eff"] >= 500 else "nothing"))


if __name__ == "__main__":
    {"timing": timing, "exit": exit_rules, "regime": regime, "features": features, "peek": peek}[sys.argv[1]]()
