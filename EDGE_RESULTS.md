# KXBTC15M edge hunt -- results

Companion to EDGE_HYPOTHESES.md (forward) and REVIEW_FINDINGS.md. Tables, not prose.
Fills: YES at ask, NO at 1 - bid, `kalshi_study.order_fee`, one entry per market. Never the mid.

## Sealed holdout (committed before any modelling)

| item | value |
|---|---|
| data | `study_data/model_obs.csv`, 92,064 rows, 329 days, 2025-12-15 .. 2026-11-08 |
| sha256 prefix | 8e8d50e7dd5f12c0 |
| exploration days | 246 (2025-12-15 .. 2026-08-17), 68,736 rows |
| holdout days | 83 (2026-08-18 .. 2026-11-08), 23,328 rows, SEALED |
| enforcement | `edge_hunt.load_exploration()` raises `HoldoutViolation` on any row >= 2026-08-18 |
| sealed on | 2026-11-08, commit "Seal the edge-hunt holdout" |

Advance bar (pre-cutoff, walk-forward via `kalshi_decision.fold_days`): mean net P&L > 0, day-block CI
lower bound > 0 at Bonferroni z over ALL rows of EDGE_TRIALS.csv, n_eff >= 500, survives +1c worse
fills and a side-flip permutation null. Holdout: ONE run per advancing candidate, single-test z.

## Part 1 -- entry quality by market state (pre-cutoff rows only; `review/kalshi_entry_quality.py`)

cost = half-spread + per-order fee (1 contract), cents. out-mid = outcome minus market mid, cents, day-block 95% CI.

| state | n | n_eff | cost | out-mid [CI] | Brier(mid) | reading |
|---|---|---|---|---|---|---|
| all valid books | 63574 | 31926 | 2.18 | -0.09 [-0.48,+0.30] | 0.128 | mid is calibrated; 0.5 scores 0.250 |
| spread 0-2c | 50969 | 27505 | 1.85 | -0.01 [-0.41,+0.40] | 0.119 | GOOD |
| spread 2-5c | 13016 | 10083 | 2.93 | -0.56 [-1.34,+0.23] | 0.163 | GOOD, wait hint |
| spread 5-10c | 1722 | 1216 | 4.92 | -1.15 [-3.38,+1.08] | 0.158 | EXPENSIVE |
| spread 10-20c | 787 | 772 | 8.31 | +1.17 [-1.58,+3.93] | 0.152 | AVOID |
| last print > 5c off mid | 1168 | 916 | 5.41 | -0.51 [-3.17,+2.15] | 0.169 | EXPENSIVE (stale/placeholder) |
| mid 0-10c | 12085 | 10416 | 1.39 | -0.76 [-1.08,-0.44] | 0.028 | favourite under-priced by < cost |
| mid 90-100c | 12064 | 10055 | 1.39 | +0.57 [+0.24,+0.90] | 0.029 | favourite under-priced by < cost |
| horizon 10 / 5 / 2 min | 22686 / 22279 / 18609 | | 2.60 / 2.10 / 1.78 | all CIs cover 0 | .188/.117/.069 | 18.8% of 2-min books untradable |
| basis > 10bp, vol quartiles | 54 / 15866 | | ~2.2-2.7 | CIs cover 0 | | no usable state |

Waiting (paired, same market): spread 2-5c at 10 min was narrower at 5 min in 74% of 5508 markets (median -1c, 97% still
tradable); 5c+ in 76% of 967 (median -3c, 89%); 5->2 min: 74%/71% narrower but only 78%/68% still tradable. Books under 2c
did not change. Shipped as `kalshi_decision.entry_quality` / `mid_calibration` / `wait_hint` (frozen tables, provenance in
the module), shown beside the action. No direction is claimed anywhere; the extreme-band under-pricing is below the entry cost.

## Part 2 -- edge hunt, pre-cutoff walk-forward (`review/kalshi_edge_hunt.py`; every variant in `EDGE_TRIALS.csv`)

6 expanding folds over 246 days (`kalshi_decision.fold_days`), 63,464 valid rows. Bar: mean - z*SE > 0 at
z = bonferroni_z(32 logged rows) = 3.16, n_eff >= 500, mean > 0 with 1c worse fills, perm p < 0.05/32.

| amendment | what | why |
|---|---|---|
| 1 (2026-11-08) | momentum features use completed Coinbase candles only (`q - 60`) | run 1 read the forming candle: a 60 s look-ahead. Its 16 rows stay in EDGE_TRIALS.csv and in the family count |
| 2 (2026-11-08) | side-flip permutations 199 -> 1999 | with 199 the smallest p (0.005) could never clear 0.05/trials; the threshold is unchanged |

Run 2 (after amendment 1), cents per contract net of fees; lo = mean - 3.16 SE; w1c = mean with 1c worse fills:

| variant | n | n_eff | mean | lo | Brier model / mid | w1c | perm p | advance |
|---|---|---|---|---|---|---|---|---|
| recal A (logit mid, model gap), m=0 / 2c | 7232 / 919 | 8385 / 957 | -0.27 / +0.48 | -1.53 / -4.05 | .1227 / .1226 | -1.02 / -0.41 | .001 / .026 | no |
| recal B (+dist, spread, horizon), m=0 / 2c | 7395 / 1112 | 8983 / 1198 | -0.15 / +0.16 | -1.37 / -3.82 | .1227 / .1226 | -1.59 / -0.79 | .001 / .026 | no |
| recal C (+hour, weekend, 5m ret, 15m range, prev outcome), m=0 / 2c | 9148 / 2479 | 10927 / 2850 | -0.25 / +0.39 | -1.47 / -2.22 | .1227 / .1226 | -1.23 / +0.17 | .001 / .001 | no |
| pocket 5 bands x horizon, m=0 / 2c | 4231 / 0 | 4675 / 0 | -0.61 / -- | -1.68 / -- | .1226 | -2.30 / -- | .034 / -- | no |
| pocket 10 bands x horizon, m=0 / 2c | 5470 / 138 | 5117 / 123 | -0.84 / -2.35 | -2.10 / -14.81 | .1227 | -3.86 / -- | .051 / .457 | no |
| basis >= 2bp, p_model_adj, m=0 / 2c | 2173 / 1790 | 2295 / 1679 | -2.47 / -2.57 | -5.18 / -5.73 | .1236 / .1226 | -3.34 / -3.08 | .645 / .697 | no |
| basis >= 5bp, p_model_adj, m=0 / 2c | 89 / 81 | 85 / 77 | -1.35 / -1.09 | -15.24 / -15.43 | .1227 | -2.06 / -2.57 | .442 / .411 | no |
| streak k=3 continuation, m=0 / 2c | 2859 / 0 | 3367 / 0 | -0.61 / -- | -2.42 / -- | .1226 | -1.45 / -- | .054 / -- | no |

Run 1 (leaked) for the record: recal C read +4.27c / +6.19c with lo +3.31 / +5.21 and Brier .1177 -- the whole gain was
the forming candle. Nothing else moved between runs beyond the permutation count.

| verdict | |
|---|---|
| candidates advancing | 0 of 16 variants (32 logged trials). No recalibration beat the mid's Brier (.1226) by more than .0001 |
| holdout (2026-08-18 ..) | UNTOUCHED: zero evaluations run; `edge_hunt.load_holdout()` was never called |
| forward data (edge_*.csv) | none on this machine; nothing to confirm against |
| result | **no edge found**. Decision line stays NO TRADE by default; nothing wired into live, paper or the saved rule |
