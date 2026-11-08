# Forward edge hunt in KXBTC15M: pre-registration

**Status: pre-registered, no confirmatory data yet.** This file was written and committed before any
row of forward data existed. Nothing below may change after data has been looked at, except by a
labelled amendment that is counted in the family (as in `kalshi_study.py`).

**Confirmatory data = rows in `edge_record.csv` with `ts_utc >= 2026-11-08T23:15:00Z`**
(`CONFIRMATORY_FROM` in `kalshi_forward_study.py`). The evaluator refuses earlier rows.

Expected outcome: none of the five passes. A clean negative is the deliverable. Two of them
(H1, H2) are re-tests at second resolution of hypotheses already negative at minute resolution on
30,688 settled markets (REVIEW_FINDINGS.md: `btc_momentum` in the scalping backtest; Step 3 of the
edge study). The other three test information that has no historical source, so they could not be
tested before. That is the only reason this study exists.

## What is already settled, and not retested

- **Price-bucket / longshot bias (candidate c).** Tested on 30,688 markets at 10, 5 and 2 minutes:
  nothing in any of 120 cells, with minimum detectable edges of 0.8-2.9 c. A forward sample of a few
  thousand markets cannot improve on that. Excluded.
- **Scalping (sell back within minutes).** The round trip costs 4-5 c and the four fixed rules lose on
  train and held out. Every trade here is HELD TO SETTLEMENT; no exit rule is tested. Excluded.
- **Time of day / vol regime (candidate e)** is not a hypothesis (no mechanism says WHICH way a
  regime mispricing would go). It is a mandatory stratification of every result (below).
- **Passive (maker) quoting (candidate f).** Excluded from confirmation. The recorder logs Kalshi's
  prints (`edge_trades.csv`) but not the queue: a resting order's fill depends on how much size was
  ahead of it at that price, which no public endpoint reports. Any "my bid was filled if a print went
  through my price" model counts fills that would have gone to the queue ahead, so it is an upper
  bound on fills and biased toward finding edge. If one is ever computed it is EXPLORATORY ONLY and
  says so in its output. There is no maker fee on this series, which is exactly why a flattering fill
  model would be dangerous here.

## Power, and why 1 c is untestable

Per-contract P&L of a hold-to-settlement trade is roughly Bernoulli: SD = 0.45 near a 50-80 c
price. Costs per contract on a 1-lot: half a 1 c spread plus a 2 c fee at 50 c, about 2.5 c. Family
of 5 tests, two-sided alpha 0.05, Bonferroni z = 2.576; 80% power adds z = 0.842, so the
n needed for a CI excluding zero at true edge e with 80% power is n = ((2.576 + 0.842) x 0.45 / e)^2,
times 1/0.8 for day-block n_eff ~ 0.8 n. Conditional rules fire in maybe a quarter of the 96 daily
markets, 24 trades a day:

| true net edge | n_eff needed (80% power) | raw trades (n_eff = 0.8 n) | at 24 trades/day |
|---|---|---|---|
| 1 c | 23,650 | 29,560 | ~41 months |
| 2 c | 5,910 | 7,390 | ~10 months |
| 3 c | 2,630 | 3,290 | ~4.6 months |

**A 1 c edge cannot be tested with a forward sample. The smallest edge this study can confirm is
about 3 c per contract**, more than the whole cost of the trade. Collection is months, not weeks.

## Stopping and futility rules (fixed)

Each hypothesis is evaluated ONCE, at the earlier of:

- `n_eff >= 2,630` for that hypothesis (80% power at 3 c), or
- **2027-05-08** (six months), whatever has accumulated.

Verdicts at that one evaluation, in this order:
1. **Inconclusive**: the Bonferroni interval's half-width `z x SE > 2 c`. Not enough data to say,
   whatever the sign of the mean. Reported as such, not as "no edge".
2. **Passes confirmation**: interval entirely above zero. Then, and only then, the rule is
   paper-traded by `paper_trading.py` under its own `source` (`auto:forward:H<k>`), kept apart by
   `summarize_by_source`, before any claim of edge.
3. **No edge found (rejected)**: interval entirely below zero.
4. **No edge found (not detected)**: interval includes zero with half-width <= 2 c.

An evaluation before the stop (to check the pipeline) is a dry run: its numbers are not a result and
must be labelled "interim, not a verdict" in whatever reports them.

## Common rules (every hypothesis)

- **One entry per market per hypothesis**: the FIRST tick inside the hypothesis's window at which its
  rule fires. Later firings in the same market are ignored.
- **Fill:** 1 contract, YES at `k_yes_ask`, NO at `1 - k_yes_bid`, never the mid; requires
  `kalshi_study.valid_book` (0 < bid < ask < 1, spread <= 20 c) and at least 1 contract at the touch
  on the side bought (`k_yes_ask_sz >= 1` for YES, `k_yes_bid_sz >= 1` for NO). Fee:
  `kalshi_study.order_fee(price, 1)` (Kalshi's 7% x P x (1-P), rounded UP to the cent per order, the
  costlier reading). P&L per contract = settles-in-favour (0/1) - price - fee.
- **Hold to settlement.** Outcome is Kalshi's own `result` via `kalshi_tracking.resolve_outcomes`.
- **Sources up:** the tick must have `k_up = 1` (the book was polled THIS tick), `cb_up = 1` with
  `cb_source = ticker` (never a candle fallback), and the hypothesis's own feed flag(s) = 1.
- **Not modelled, stated here and in the evaluator's output:** queue position, market impact, the
  size behind the touch, and any latency beyond the recorder's own round trip (`k_rtt_ms`). H1
  additionally prices its fill from the NEXT tick (below). All of these flatter the result.
- **Statistics:** mean net P&L per contract, day-block (UTC day) bootstrap SE from
  `kalshi_study.block_bootstrap`, `n_eff` from the same, interval mean +/- 2.576 SE (Bonferroni over
  the family of 5). The scoring comparison under H2 is descriptive and not counted.

## The hypotheses (family of 5, z = 2.576)

**H1. Stale Kalshi quote after a fast Coinbase move.**
*Mechanism:* Kalshi's makers re-quote on a lag; immediately after a move the standing quote still
prices the old spot, so the side the move favours is underpriced.
*Window:* `secs_to_close` in [60, 300].
*Fires when:* Coinbase moved by at least **5 bp** between the tick 10 s earlier and this tick (both
ticker-sourced, no gap row in between), AND Kalshi's YES bid and ask are both UNCHANGED across those
same 10 s (quote staleness is the market's, not the recorder's: `k_up = 1` on both ticks), AND the
logged `p_settle` on this tick clears the entry price plus fee by at least **1 c** on the side of the
move (YES after an up move, NO after a down move).
*Fill:* at the quote of the NEXT recorded tick (>= 1 s later), to charge the recorder's own latency.
If the next tick's book fails `valid_book` or the next tick is more than 3 s away, no trade.
*Feeds:* `k_up`, `cb_up`.

**H2. The final-minutes quote prices the point-in-time rule, not the 60-second average.**
*Mechanism:* the contract settles on the average of 60 one-second prints, whose variance is lower
than the terminal print's, so the true probability is MORE extreme than N(d2) (gotcha 13). A maker
quoting point-in-time is systematically too close to 50 c in the last minutes.
*Window:* `secs_to_close` in [60, 180] only. Inside 60 s the model assumes the prints already taken
equal spot, which is an approximation; above 180 s the two models differ by too little to clear cost.
*Fires when:* `vol_ann` present, and the logged `p_settle` clears the entry price plus fee by at least
**2 c** on either side (YES if `p_settle - ask - fee >= 0.02`, NO if
`(1 - p_settle) - (1 - bid) - fee >= 0.02`; if both, no trade).
*Feeds:* `k_up`, `cb_up`, `vol_ann` non-blank.
*Descriptive, not counted:* on the same rows, Brier of the market mid vs `p_settle` vs `p_point`, as
paired day-block differences. Step 3 already found the mid beats `p_settle` at 2 minutes on minute
data; this says whether second-level inputs change that, and whether the mid sits nearer `p_point`.

**H3. Persistent aggregated depth imbalance.**
*Mechanism:* resting-liquidity imbalance near the mid predicts short-horizon drift on the spot
venues; Kalshi's makers price off the last trade, so that drift is not yet in the quote.
*Window:* `secs_to_close` in [60, 300].
*Fires when:* `book_up = 1`, `book_venues_up >= 2`, `|book_imb| >= 0.18` (`BOOK_IMBALANCE_THRESHOLD`)
with `book_persist_s >= 30`, AND the Kalshi mid is in **[0.40, 0.60]**. Side: `book_persist_dir`
bullish -> YES, bearish -> NO.
*Why the mid gate:* the information must beat the MARKET, not 50%. Away from 50 c the quote already
carries a strong view from spot's distance to the strike, and a depth lean has to overturn it; near
50 c the market says it does not know, so if depth knows anything it shows here first.
*Feeds:* `k_up`, `cb_up`, `book_up`.

**H4. Aggressive trade-flow imbalance.**
*Mechanism:* aggregated taker flow across three venues over the last 60 s leads the next minutes'
price (order-flow momentum), and the Kalshi quote lags it.
*Window:* [60, 300]. *Fires when:* `agg_up = 1`, `agg_venues_up >= 2`, `agg_delta >= +0.30` -> YES or
`<= -0.30` -> NO (twice `AGG_DELTA_THRESHOLD`: a strong reading, fixed here), mid in [0.40, 0.60].
*Feeds:* `k_up`, `cb_up`, `agg_up`.

**H5. Liquidation cascade momentum.**
*Mechanism:* forced liquidations are price-insensitive flow that continues for minutes (stops and
further margin calls), pushing spot the same way; the quote lags it.
*Window:* [60, 300]. *Fires when:* `liq_up = 1` and, over the trailing 30 s, `liq_long_notional >=
1,000,000` USD -> NO (longs liquidated = forced selling) or `liq_short_notional >= 1,000,000` -> YES;
if both sides cross, no trade. Mid in [0.40, 0.60].
*Feeds:* `k_up`, `cb_up`, `liq_up`.

**H3-H5 overlap.** All three read the same few minutes of the same spot market and will often fire
together on the same side in the same market. They are not three independent chances: a market where
all three win is one observation of the spot move, not three. Bonferroni over 5 is conservative for
correlated tests, so the family-wise error is still controlled, but a joint pass is reported as one
finding. The evaluator prints the pairwise overlap (share of markets traded by both, same side) so
nobody reads three passes as three confirmations.

## Mandatory sanity checks (each hypothesis, same output)

- **Permutation null:** outcomes shuffled across the hypothesis's trades 2,000 times (seed fixed), the
  share of shuffles with mean P&L >= the observed one.
- **Stratification:** mean P&L by UTC day; by UTC 6-hour bin (00-06, 06-12, 12-18, 18-24); by
  tercile of `vol_ann` at entry, with tercile edges fixed from ALL confirmatory ticks (not from trades).
- **Worse fills:** the same trades with every entry 1 c worse (ask + 1 c, bid - 1 c).
- **Concentration:** share of total P&L from the five most profitable markets, and the mean P&L with
  those five removed.
- **Model drift:** count of rows whose `model_version` differs from the evaluator's, and the max gap
  between logged `p_settle` and `p_settle` recomputed from the row's own inputs.

## Data rules

- Only `row_kind = tick` rows are observations. A `gap` row breaks any lookback (H1's 10 s).
- Blank cells are "feed down" and disqualify the tick for a hypothesis that needs that field.
  Zero is a reading and does not.
- A market is identified by `ticker`; `strike` comes from Kalshi's record (a missing or mis-scaled
  `floor_strike` means `kalshi_api.parse_market` returned unavailable and nothing was recorded for
  that market; the evaluator reports how many market tickers in `edge_trades.csv` never appear in the
  record for this reason).
- Markets whose outcome `resolve_outcomes` cannot return are reported as pending, never dropped
  silently or counted as a loss.

## Exploratory pass on the existing cache

Not performed: the cache (`study_data/`) does not exist where this was written and Kalshi's API was
unreachable from there. At minute resolution the cache can only speak to H1 (as `btc_momentum`, which
lost on train and held out) and H2 (Step 3, where the market beat the model at 2 minutes, both
metrics, held out). Both are already negative; anything run on that cache later is EXPLORATORY ONLY
and may not set a parameter above.

## Amendments

None. (Format: date, what changed, why, and the new family size and z.)
