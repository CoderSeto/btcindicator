"""
Edge hunt on KXBTC15M, pre-cutoff rows only (EDGE_RESULTS.md, Part 2).

Every variant is a walk-forward (kalshi_decision.fold_days) over the exploration
days that edge_hunt.load_exploration() returns -- the sealed holdout cannot be
loaded here. Fills: YES at ask, NO at 1 - bid, kalshi_study.order_fee, one
entry per market (rule_pnl's earliest-horizon rule). EVERY variant evaluated
is appended to EDGE_TRIALS.csv; the row count sets the Bonferroni z.

Families (one-line mechanism each):
  recal   -- the market mid, shifted by what a ridge logistic fit on earlier
             days says about residual information in the model, distance, spread,
             horizon, time of day and recent BTC momentum.
  pocket  -- (price band x horizon) cells where earlier days' outcomes ran away
             from the mid by more than the entry cost; trade that direction.
  basis   -- the model rule only where the Coinbase/CF basis is large, since
             p_model_adj already prices the basis the market may not.
  streak  -- after k same outcomes, buy the continuation side if it clears cost
             (under-reaction) -- the one "anything else" family.

Advance bar (fixed before running): walk-forward mean net P&L > 0, day-block CI
lower bound > 0 at z = bonferroni_z(len(EDGE_TRIALS.csv)), n_eff >= 500, still
positive with every fill 1c worse, and beats 199 side-flip permutations.
Prints <= 30 lines; the full table is EDGE_TRIALS.csv.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv_utils  # noqa: E402
import edge_hunt  # noqa: E402
import kalshi_decision as kd  # noqa: E402
import kalshi_study as ks  # noqa: E402

TRIALS_PATH = "EDGE_TRIALS.csv"
TRIALS_HEADER = ["id", "family", "params", "n", "n_eff", "mean_c", "se_c", "brier_model", "brier_mid",
                 "mean_worse1c", "perm_p", "stamp"]
MIN_NEFF = 500
PERMUTATIONS = 1999      # AMENDMENT 2: 199 made p < 0.05/trials unreachable
RIDGE = 1.0
SLIP = 0.01
SEED = 20261108


# --- features ----------------------------------------------------------------

def _momentum(obs: pd.DataFrame, cache_dir: str = "study_data/coinbase_1m") -> pd.DataFrame:
    """5-minute log return and 15-minute high-low range (in vol units) before each quote."""
    out = pd.Series(np.nan, index=obs.index), pd.Series(np.nan, index=obs.index)
    for day, g in obs.groupby("day"):
        p = os.path.join(cache_dir, f"{day}.csv")
        prev = os.path.join(cache_dir, f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).date()}.csv")
        frames = [pd.read_csv(f) for f in (prev, p) if os.path.exists(f)]
        if not frames:
            continue
        c = pd.concat(frames).drop_duplicates("ts").sort_values("ts").set_index("ts")
        ts = c.index.to_numpy()
        close = c["close"].to_numpy()
        hi, lo = c["high"].to_numpy(), c["low"].to_numpy()
        q = g["quote_ts"].to_numpy()
        # Completed candles only: a candle is stamped with its bucket START, so the one
        # containing q is still forming at q (AMENDMENT 1 in EDGE_RESULTS.md: the first
        # run used it, a 60s look-ahead).
        i_now = np.searchsorted(ts, q - 60, side="right") - 1
        i_5 = np.searchsorted(ts, q - 360, side="right") - 1
        i_15 = np.searchsorted(ts, q - 960, side="right") - 1
        ok = (i_now >= 0) & (i_5 >= 0) & (i_15 >= 0) & (ts[np.clip(i_now, 0, None)] >= q - 180)
        r5 = np.where(ok, np.log(close[np.clip(i_now, 0, None)] / close[np.clip(i_5, 0, None)]), np.nan)
        rng = np.full(len(q), np.nan)
        for k in np.flatnonzero(ok):
            a, b = i_15[k], i_now[k] + 1
            rng[k] = np.log(hi[a:b].max() / lo[a:b].min())
        out[0].loc[g.index] = r5
        out[1].loc[g.index] = rng
    return pd.DataFrame({"ret5": out[0], "range15": out[1]})


def features(obs: pd.DataFrame) -> pd.DataFrame:
    d = obs.copy()
    d["spread"] = d.ask - d.bid
    d["logit_mid"] = ks._logit(d.mid.clip(0.01, 0.99))
    d["model_gap"] = d.p_model - d.mid
    d["adj_gap"] = d.p_model_adj - d.mid
    t_years = d.horizon / (365.0 * 24 * 60)
    d["dist"] = np.log(d.spot / d.strike) / (d.vol * np.sqrt(t_years))
    hour = pd.to_datetime(d.close_ts, unit="s", utc=True)
    d["hour_sin"] = np.sin(2 * np.pi * hour.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour.dt.hour / 24)
    d["weekend"] = (hour.dt.day_of_week >= 5).astype(float)
    for h in kd.HORIZONS:
        d[f"h{h}"] = (d.horizon == h).astype(float)
    mom = _momentum(d)
    d["ret5_v"] = mom.ret5 / (d.vol * np.sqrt(5 / (365.0 * 24 * 60)))
    d["range15_v"] = mom.range15 / (d.vol * np.sqrt(15 / (365.0 * 24 * 60)))
    # outcome streak: the last 3 settled markets' outcomes before this one opened
    m = d.drop_duplicates("close_ts").sort_values("close_ts")[["close_ts", "outcome"]]
    prev = m.outcome.shift(1), m.outcome.shift(2), m.outcome.shift(3)
    streak = pd.Series(np.where((prev[0] == prev[1]) & (prev[1] == prev[2]), np.where(prev[0] == 1, 1, -1), 0),
                       index=m.close_ts)
    d["streak3"] = d.close_ts.map(streak).fillna(0.0)
    d["prev_out"] = d.close_ts.map(pd.Series(prev[0].to_numpy(), index=m.close_ts)).fillna(0.5)
    return d


FEATURE_SETS = {
    "A": ["logit_mid", "model_gap"],
    "B": ["logit_mid", "model_gap", "dist", "spread", "h10", "h5"],
    "C": ["logit_mid", "model_gap", "dist", "spread", "h10", "h5", "hour_sin", "hour_cos", "weekend",
          "ret5_v", "range15_v", "prev_out"],
}


def ridge_logistic(X, y, lam=RIDGE, iters=25):
    """L2-penalised logistic regression by Newton steps (intercept unpenalised)."""
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


# --- trading a probability ---------------------------------------------------

def trade_prob(df: pd.DataFrame, p_hat, margin: float, slip: float = 0.0):
    """One entry per market at the earliest horizon where p_hat clears the touch + fee + margin."""
    r = kd.Rule(margin=margin, shrink=1.0, horizons=tuple(kd.HORIZONS), band=None)
    d = df.assign(p_model=np.asarray(p_hat, dtype=float), ask=df.ask + slip, bid=df.bid - slip)
    side, pnl = kd.rule_pnl(r, d, 1)
    return side, pnl


# --- the walk-forward ----------------------------------------------------------

def walk(d: pd.DataFrame, fit_predict, margin: float, slip: float = 0.0):
    """fit_predict(train, test) -> p_hat for test rows. Returns pnl (NaN = no trade), day, p_hat, side."""
    pnl = np.full(len(d), np.nan)
    phat = np.full(len(d), np.nan)
    sides = np.zeros(len(d))
    for train_days, test_days in kd.fold_days(d.day):
        tr = d[d.day.isin(train_days)]
        te_mask = d.day.isin(test_days).to_numpy()
        te = d[te_mask]
        p = fit_predict(tr, te)
        side, pp = trade_prob(te, p, margin, slip)
        pnl[te_mask], phat[te_mask], sides[te_mask] = pp, p, side
    return pnl, phat, sides


def perm_p(pnl, sides, df, rng):
    """Share of side-flip permutations whose mean P&L >= the real one (fills recomputed)."""
    traded = np.isfinite(pnl)
    real = np.nanmean(pnl)
    o, a, b = df.outcome.to_numpy()[traded], df.ask.to_numpy()[traded], df.bid.to_numpy()[traded]
    s = sides[traded]
    hits = 0
    for _ in range(PERMUTATIONS):
        flip = rng.integers(0, 2, size=s.shape) * 2 - 1
        ss = s * flip
        v = np.where(ss == 1, ks.ev_buy_yes(o, a, 1), ks.ev_buy_no(o, b, 1))
        hits += v.mean() >= real
    return (hits + 1) / (PERMUTATIONS + 1)


def log_trial(tid, family, params, d, pnl, phat, sides, pnl_worse, rng):
    traded = np.isfinite(pnl)
    s = ks.block_bootstrap(pnl[traded], d.day.to_numpy()[traded])
    scored = np.isfinite(phat) & d.valid.to_numpy()
    bm = float(ks.brier(phat[scored], d.outcome.to_numpy()[scored]).mean()) if scored.any() else np.nan
    bq = float(ks.brier(d.mid.to_numpy()[scored], d.outcome.to_numpy()[scored]).mean()) if scored.any() else np.nan
    pp = perm_p(pnl, sides, d, rng) if traded.sum() else np.nan
    row = [tid, family, params, s["n"], round(s["n_eff"], 1), round(100 * s["mean"], 3) if s["n"] else "",
           round(100 * s["se"], 3) if s["n"] > 1 else "", round(bm, 5) if scored.any() else "", round(bq, 5) if scored.any() else "",
           round(100 * np.nanmean(pnl_worse), 3) if np.isfinite(pnl_worse).any() else "", round(pp, 3) if traded.sum() else "",
           int(time.time())]
    csv_utils.append_row(TRIALS_PATH, TRIALS_HEADER, row)
    return s, bm, bq, pp


def main():
    rng = np.random.default_rng(SEED)
    obs = edge_hunt.load_exploration()
    d = features(obs)
    d = d[d.valid & d.p_model.notna()].reset_index(drop=True)
    edge_hunt.assert_exploration(d)
    print(f"exploration rows {len(d)}, days {d.day.nunique()}, folds {len(kd.fold_days(d.day))}")
    results = []

    def run(tid, family, params, fit_predict, margin):
        pnl, phat, sides = walk(d, fit_predict, margin)
        pnl_w, _, _ = walk(d, fit_predict, margin, slip=SLIP)
        s, bm, bq, pp = log_trial(tid, family, params, d, pnl, phat, sides, pnl_w, rng)
        results.append((tid, family, params, s, bm, bq, float(np.nanmean(pnl_w)) if np.isfinite(pnl_w).any() else np.nan, pp))

    # recal: ridge logistic on feature set, margins 0 / 2c
    for fs, margin in itertools.product(FEATURE_SETS, (0.0, 0.02)):
        cols = FEATURE_SETS[fs]

        def fp(tr, te, cols=cols):
            tr = tr.dropna(subset=cols)
            w = ridge_logistic(tr[cols].to_numpy(float), tr.outcome.to_numpy(float))
            X = te[cols].to_numpy(float)
            p = predict(w, np.nan_to_num(X))
            return np.where(np.isfinite(X).all(axis=1), p, np.nan)
        run(f"recal_{fs}_m{int(margin * 100)}", "recal", f"features={fs};ridge={RIDGE};margin={margin}", fp, margin)

    # pocket: train-cell outcome - mid gap applied as a shift to the mid
    for edges, margin in itertools.product(((0, .1, .25, .75, .9, 1), tuple(np.linspace(0, 1, 11))), (0.0, 0.02)):
        def fp(tr, te, edges=edges):
            cell = lambda f: list(zip(pd.cut(f.mid, edges, include_lowest=True).astype(str), f.horizon))  # noqa: E731
            g = pd.DataFrame({"c": cell(tr), "gap": tr.outcome - tr.mid}).groupby("c").gap.agg(["mean", "count"])
            shift = pd.Series(cell(te)).map(g["mean"].where(g["count"] >= 100, 0.0)).fillna(0.0).to_numpy()
            return te.mid.to_numpy() + shift
        run(f"pocket_{len(edges) - 1}b_m{int(margin * 100)}", "pocket", f"bands={len(edges) - 1};min_cell=100;margin={margin}",
            fp, margin)

    # basis: adjusted model where |basis_open_bp| exceeds a floor, else the mid (no trade)
    for floor, margin in itertools.product((2.0, 5.0), (0.0, 0.02)):
        def fp(tr, te, floor=floor):
            big = te.basis_open_bp.abs() >= floor
            return np.where(big & te.p_model_adj.notna(), te.p_model_adj, te.mid)
        run(f"basis_{int(floor)}bp_m{int(margin * 100)}", "basis", f"|basis|>={floor}bp;p=p_model_adj;margin={margin}", fp, margin)

    # streak: after three same outcomes, nudge toward the continuation by the train-measured gap
    for margin in (0.0, 0.02):
        def fp(tr, te):
            t = tr[tr.streak3 != 0]
            gap = float(((t.outcome - t.mid) * t.streak3).mean()) if len(t) else 0.0
            return te.mid.to_numpy() + gap * te.streak3.to_numpy()
        run(f"streak3_m{int(margin * 100)}", "streak", f"k=3;shift=train_gap;margin={margin}", fp, margin)

    trials = pd.read_csv(TRIALS_PATH)
    z = ks.bonferroni_z(len(trials))
    print(f"trials logged: {len(trials)}; Bonferroni z = {z:.2f}; advance needs mean - z*se > 0, n_eff >= {MIN_NEFF}, "
          f"worse-fill mean > 0, perm p < 0.05/{len(trials)}")
    print(f"{'id':<16}{'n':>6}{'n_eff':>8}{'mean_c':>8}{'lo_c':>8}{'brier':>8}{'vs_mid':>8}{'w1c':>7}{'perm':>6} adv")
    for tid, fam, params, s, bm, bq, w1, pp in results:
        lo = s["mean"] - z * s["se"] if s["n"] > 1 else np.nan
        adv = s["n"] > 1 and lo > 0 and s["n_eff"] >= MIN_NEFF and w1 > 0 and pp < 0.05 / len(trials)
        print(f"{tid:<16}{s['n']:>6}{s['n_eff']:>8.0f}{100 * s['mean']:>+8.2f}{100 * lo:>+8.2f}{bm:>8.4f}{bq:>8.4f}"
              f"{100 * w1:>+7.2f}{pp:>6.3f} {'ADVANCE' if adv else '-'}")


if __name__ == "__main__":
    main()
