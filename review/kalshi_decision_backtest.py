"""Run from the repo root, after `python review/kalshi_market_study.py step3` (it writes study_data/model_obs.csv).

    python review/kalshi_decision_backtest.py

Walk-forward backtest of kalshi_decision's rule grid on the cached KXBTC15M study data: each fold's rule is
chosen on earlier days only and traded on the next block. Prints the out-of-sample record and writes
decision_backtest.json, which the dashboard reads for its BUY YES / BUY NO / NO TRADE line.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
import kalshi_decision as kd
import kalshi_study as ks

NEEDED = {"close_ts", "day", "horizon", "bid", "ask", "p_model", "outcome"}


def main():
    path = os.path.join(ks.CACHE_DIR, "model_obs.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run `python review/kalshi_market_study.py fetch` then `step3` first.")
    obs = pd.read_csv(path)
    missing = NEEDED - set(obs.columns)
    if missing:
        sys.exit(f"{path} lacks columns {sorted(missing)}")
    obs = obs[obs["valid"] & obs["p_model"].notna()] if "valid" in obs else obs
    print(f"{obs.close_ts.nunique()} markets, {obs.day.nunique()} days, {len(kd.GRID)} rules searched.")
    res = kd.walk_forward(obs)
    for f in res["folds"]:
        r = f["rule"]
        print(f"  {f['test_days'][0]}..{f['test_days'][1]}: "
              + (f"{kd.Rule(r['margin'], r['shrink'], tuple(r['horizons']), r['band']).label()}  "
                 f"n={f['n']} mean={f['mean'] * 100:+.2f}c" if r else "no rule cleared the train bar -> NO TRADE"))
    print(kd.oos_line(res))
    print("Rule the dashboard will run: "
          + (kd.Rule(res['rule']['margin'], res['rule']['shrink'], tuple(res['rule']['horizons']),
                     res['rule']['band']).label() if res["rule"] else "none (NO TRADE always)"))
    ref = ks.model_trades  # the pre-registered rule, for scale: trade toward the raw model at margin 5c, 10 min
    d = obs[obs.horizon == 10]
    side, pnl = ref(d.p_model, d.bid, d.ask, d.outcome, 0.05, 1)
    pnl = pnl[~pd.isna(pnl)]
    print(f"For scale, the study's raw-model rule (5c margin, 10 min, all data): n={len(pnl)} mean={pnl.mean() * 100:+.2f}c")
    kd.save_result(res, config.DECISION_BACKTEST_PATH, asof=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    print(f"wrote {config.DECISION_BACKTEST_PATH}")


if __name__ == "__main__":
    main()
