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
