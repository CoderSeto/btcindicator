# KXBTC15M decision rules -- measured, not assumed

Pre-cutoff study rows only (`edge_hunt.load_exploration`, 246 days to 2026-08-17). Scripts: `review/kalshi_decision_rules.py
<timing|exit|regime|features|peek>`. Fills: YES at ask, NO at 1 - bid, `kalshi_study.order_fee`, one entry per market.
cost = half-spread + fee per contract. CIs are day-block bootstrap 95%. Frozen tables live in `kalshi_decision.py`.

## 1. Entry timing -> ENTER NOW / WAIT / SKIP (`entry_timing`)

Paired within a market: cost at this snapshot vs the next. The mid moved ~17-29c between snapshots in every state.

| state (now -> next) | n | still tradable | cost change [CI] | cost now | rule |
|---|---|---|---|---|---|
| 10->5, spread <2c | 17341 | 98.7% | -0.38 [-0.40,-0.35] | 2.28 | ENTER NOW (saving < 1c) |
| 10->5, spread 2-5c | 4572 | 97.0% | -0.75 [-0.81,-0.69] | 3.19 | ENTER NOW |
| 10->5, spread 5c+ | 773 | 87.5% | -1.91 [-2.17,-1.66] | 6.32 | **WAIT** (saving >= 1c, >= 85% open) |
| 10->5, mid <=10c / >=90c | 835 / 862 | 90% / 89% | +0.11 / +0.09, CI > 0 | 1.5 | ENTER NOW (never cheaper) |
| 5->2, spread <2c / 2-5c / 5c+ | 17887 / 3749 / 643 | 84% / 80% / 66% | -0.25 / -0.64 / -1.72 | 1.8 / 2.9 / 6.3 | ENTER NOW (too many books close) |
| 5->2, mid <=10c / >=90c | 4820 / 4853 | 70% / 69% | +0.03 / +0.04 | 1.4 | ENTER NOW |
| 2 min | | | no later snapshot | | ENTER NOW |
| any, spread >= 10c or one-sided | 787 | | cost 8.3c, mid uninformative | | SKIP |

Vol regime did not change the timing picture (cost change -0.5c in all three quartiles at 10->5).

## 2. Exit -> HOLD TO SETTLE (`decide_exit`, evidence in `EXIT_EVIDENCE`)

Sell-back at the next snapshot's REAL bid, fee paid again, minus hold-to-settle. Both sides entered in every valid market.

| held, from -> to | n | n_eff | sell-back minus hold [CI] |
|---|---|---|---|
| YES, 10 -> 2 / 5 -> 2 | 18567 / 18527 | 18166 / 18183 | -1.56 [-1.95,-1.18] / -1.56 [-1.94,-1.18] |
| NO, 10 -> 2 / 5 -> 2 | 18567 / 18527 | 17623 / 17574 | -1.99 [-2.38,-1.60] / -1.99 [-2.37,-1.60] |
| YES by band 0-10 / 10-25 / 25-75 / 75-90 / 90-100c (10->2) | 485..12361 | | -1.81 / -1.05 / -1.60 / -2.06 / -0.77 [-2.26,+0.72] |
| mid moved 5c+ against the position, YES / NO (10->2) | 8832 / 8698 | | -0.95 [-1.50,-0.40] / -1.44 [-2.00,-0.89] |
| mid moved 5c+ for the position, YES / NO (10->2) | 8698 / 8832 | | -2.05 [-2.62,-1.47] / -2.52 [-3.10,-1.94] |

Selling back lost in every state; no state's CI is above zero. The round trip costs ~2c (entry) + ~1.6-2.0c (exit).
`decide_exit` keeps its 3c margin (EXIT only when the sale beats the hold value by 3c) and now prints this evidence.

## 3. Regime filters (`regime`)

| state | n | out - mid [CI] | Brier(mid) | cost | untradable | flag |
|---|---|---|---|---|---|---|
| hour (UTC) 0 / 4 / 8 / 12 / 16 / 20 | ~10.5k each | all CIs cover 0 (4h: -1.03 [-2.12,+0.05]) | .123-.133 | 2.16-2.21 | 7-8% | none |
| vol low / mid / high | 16373 / 31909 / 15292 | 0.00 / -0.31 / +0.27, CIs cover 0 | .126/.128/.131 | 2.25/2.13/2.23 | 4.6% / 7.3% / **10.9%** | high vol: more books close |
| basis <2 / 2-10 / >10 bp | 53297 / 10140 / 54 | CIs cover 0 | .128 | 2.1-2.7 | | none (>10bp too rare) |
| streak3 none / up / down | 49314 / 7157 / 7103 | -0.29 / -0.79 / **+2.00 [+0.83,+3.16]** | .128/.132/.128 | 2.2 | | see peek: not tradable after cost |

## 4. Does anything explain outcome - mid? (`features`; OLS on z-scored features, n=63458, R2=0.0014, Bonferroni z=2.94)

| feature | cents per SD | z | explains? |
|---|---|---|---|
| 5-min BTC return (completed candles) | +0.81 | +8.0 | **yes**, but tiny: recal C in EDGE_RESULTS.md traded it walk-forward at +0.39c [lo -2.22] |
| model gap, adjusted gap, distance, spread, vol, 15-min range, basis, hour, weekend, streak, prev outcome, horizon | |z| <= 2.24 | no |
| composite / strength | not in the cache; `signal_log.csv` absent on this machine | | untestable here |

## 5. Dashboard line (`kalshi_decision.decision_line`): action @ price (+fee) · ENTER NOW/WAIT/SKIP · entry GOOD/EXPENSIVE/AVOID (cost) · held: HOLD/EXIT.
Caption: rule reason, timing reason, exit reason + evidence, entry cost both sides, break-evens, mid calibration (n, n_eff).

## Edge peek: nothing
180 state x horizon x side cells (favourite / underdog / model side at the touch; bands, hours, vol, spread, streak) logged to
`EDGE_TRIALS.csv` (365 rows incl. one accidental duplicate pass, kept). Bonferroni z = 3.81. Best lower bound -1.35c
(band 90-100c, 5 min, favourite); streak-down favourite not in the top 8. No candidate; nothing wired anywhere.
