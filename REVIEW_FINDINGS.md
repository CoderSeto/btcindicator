# Review findings, 2026-11-07

This review follows `CODE_REVIEW.md`. It started from `978c6c0`, a fast-forward of
`claude/dreamy-pascal-8u88jt`, which is where `CODE_REVIEW.md` lives.

**Tests:** 391 passed before the review and 420 after it. The suite also passes under
`pytest-randomly` with seeds 1, 2 and 3, in reverse file order, and on two consecutive runs. I
found no order dependence.

**Follow-up pass (same day):** the review's three recommended next steps are done, in four
commits. **D1** and **D6** (shared config, a log row per rerun), **D2** and **D3** (overstated n,
edge against 50%), and **D4** (the Kalshi model ignored the settlement average, and calibration
scored the closest-to-close row) are all addressed. **D7**, the concurrent-writer race on the
CSV logs, is fixed too, and re-measured as worse than first reported. 466 tests pass. The rows
below say what changed; the verdict is updated to match.

## Verdict

**Use it as a descriptive monitor. Don't trade on its numbers.** The data plumbing is careful:
fixed CSV schemas, per-venue books, the Binance sequencing procedure, and caveats printed in the
output. The worst problems were in the numbers people act on:

- **Kalshi block.** On any pair other than BTC-USD it priced the BTC contract against that pair's
  candles. It also logged hand-edited strikes under the live ticker, which corrupted the
  calibration data. And its settlement-averaging caveat pointed the wrong way. All three are fixed.
- **Order book.** It could cast a `book_flow` vote when a venue merely dropped out. It credited
  "held" persistence and wall age across outages. Wall tracking merged every wall on a side into
  one, so a brand-new wall could pass the anti-spoof age filter. All fixed.

The Kalshi P(YES) now prices the 60-second settlement average and matches a Monte Carlo of the
real rule to within 0.001 from 14 minutes down to 1 (it was 13 points off at 1 minute). It is
still a driftless-GBM model fed Coinbase prices, and inside the final minute it assumes the
already-taken prints equal the current price. The page flags that minute.

**The backtest no longer overstates its evidence.** Every hit rate is shown next to its base rate
and its overlap-corrected effective sample size, and small-sample flags use the effective size. A
weight only leaves 1.0 when the edge over the base rate clears 2 standard errors; on random walks
that happened in 0–4% of runs, against ±0.08–0.16 of pure noise before. Expect most weights to read
1.00, which is the honest answer at these sample sizes.

## Findings

Severity follows CODE_REVIEW.md §6. *Confirmed bug* means a failing test reproduced it before the
fix. Every fixed item is its own commit, named in the Evidence column.

| ID | Severity | Type | Location | Summary | Evidence |
|----|----------|------|----------|---------|----------|
| B1 | **Critical** | Confirmed bug, fixed | `app.py:267` | On ETH-USD, SOL-USD and BTC-EUR, auto-track still fetched KXBTC15M. It priced the ~$75k BTC strike against that pair's candles: ~0% on ETH, an FX-shifted value on BTC-EUR. The result showed as "Kalshi P(YES)" and was logged under the BTC ticker. | `test_app.py::test_kalshi_auto_track_only_follows_the_btc_contract_on_btc_usd`; commit `8fc4817` |
| B2 | **High** | Confirmed bug, fixed | `app.py:296` | A strike edited by hand stays in place on purpose, but each row was still logged under the live market's ticker. The calibration report then scored a P(YES) for one strike against the settlement of another. | `test_app.py::test_manually_edited_strike_is_never_logged_under_the_live_ticker` (before the fix the log held `[75743.66, 75000.0, 75000.0]`); `97a592d` |
| B3 | **High** | Confirmed bug, fixed | `kalshi.py:24`, `app.py:673`, README, CLI | The settlement-averaging caveat had the direction backwards. Averaging the last 60s **lowers** terminal variance, so the true probability is *more* extreme than the model. Example: 1 min left, spot 5bp above strike, model 79% vs averaged settlement ~92%. Anyone following the caveat would adjust the wrong way. (The follow-up then modelled the average itself; see D4.) | `test_kalshi.py::test_settlement_averaging_makes_the_true_probability_more_extreme` (independent Monte Carlo); `python review/settlement_averaging.py`; `b8a04e1` |
| B4 | **High** | Confirmed bug, fixed | `orderbook_feeds.py:544` (`_flow`) | Flow divided summed notional over whichever venues were usable at each end of the window. When a bid-heavy venue left the aggregate (a disconnect, or the spread gate during volatility), the result was **−0.69, a bearish flow vote that nobody placed** (threshold 0.20). | `test_orderbook_feeds.py::test_flow_ignores_a_venue_leaving_the_aggregate`; `f64490b` |
| B5 | **High** | Confirmed bug, fixed | `orderbook_feeds.py:478` (`_track_walls`) | The wall key was `round(price / (price*bps/1e4))`, which is a constant. Every wall on a venue's side merged into one entry, so only one was ever listed. A new wall 80bp away inherited the old wall's age and showed as "held 31s", which defeats the spoof filter that wall age exists for. | `test_walls_at_different_prices_are_tracked_separately`, `test_a_new_wall_at_another_price_does_not_inherit_an_old_walls_age`; `9545b27` |
| B6 | Medium | Confirmed bug, fixed | `kalshi.py:57` | Realized vol treated every candle as one minute. With `GRANULARITY` set to anything but 60, vol and P(YES) were wrong; 5-min candles gave √5 ≈ 2.24× the vol. Correct at the default of 60. | `test_kalshi.py::test_realized_vol_annualizes_by_the_candle_size_not_by_minutes`, `test_app.py::test_fair_value_uses_the_configured_candle_size`; `0190517` |
| B7 | Medium | Confirmed bug, fixed | `orderbook_feeds.py:505,470` | Persistence and wall age bridged gaps in the sampled history (an outage, or the laptop asleep). A lean seen before and after a 5-min gap counted as "held 301s" and passed the 20s vote gate on its first sample back. | `test_persistence_does_not_bridge_a_gap_in_the_history`, `test_wall_age_does_not_bridge_a_sampling_gap`; `648d8cc` |
| B8 | Medium | Confirmed bug, fixed | `orderbook_feeds.py:755-880` | Binance.US resync. (a) One REST failure during sync was terminal: nothing re-synced on `not_seeded`, so the venue stayed dark until the socket reconnected. (b) Events arriving while the snapshot was installed could be applied out of order. That caused a spurious gap, a swallowed retry, and "connected" shown over an empty book. | `test_binance_recovers_after_a_failed_snapshot_fetch`, `test_binance_events_arriving_while_the_snapshot_is_applied_are_replayed_in_order` (these drive the real callbacks); `f45e4bf` |
| B9 | Low | Confirmed bug, fixed | `exchange_feeds.py:370` | The reconnect backoff never reset, so after a few routine drops every reconnect waited the full 30s. `time.sleep` also delayed `stop()` by up to 30s. | `tests/test_exchange_feeds.py` (3 tests); `98ef90b` |
| B10 | Low | Confirmed bug, fixed | `signals.py:40-95` | Ties voted "bearish", and a zero-width Bollinger band read as "at the upper band". A dead-flat market therefore cast 4 bearish votes and fired a bearish alert at the default threshold of 3. | `test_signals_price.py::test_a_perfectly_flat_market_casts_no_directional_votes`; `3ead1d2` |
| B11 | Low | Confirmed bug, fixed | `outlook.py:40` | The outlook said "MACD just crossed" for what is a state vote: the crossover branch in `signals.py` returns the same answer as the state check. | `test_outlook.py::test_macd_phrase_describes_a_state_not_a_fresh_cross`; `da2c571` |
| B12 | Low | Confirmed bug, fixed | `app.py` Notes tab | The weights note said agg_delta "always" shows 1.00, right beside a table showing 1.40 once `backtest_agg_delta.py` has run. | `test_app.py::test_weights_note_does_not_claim_agg_delta_is_always_unit_weight`; `9e6dcf3` |
| B13 | Low | Confirmed bug, fixed | `kalshi_api.py:79` | A `close_time` with no timezone raised TypeError, which no caller catches, so the page would show an exception. Known item 6, now verified. | `test_kalshi_api.py::test_a_close_time_without_a_timezone_is_read_as_utc_not_a_crash`; `7706d4e` |
| B14 | Nit | Confirmed bug, fixed | `kalshi.py:125` | An expired exact tie returned 0.5. The contract is `greater_or_equal`, so a tie settles YES. This changed a previously pinned assertion. | `test_kalshi.py::test_crossing_expired_is_decided_by_price`; `1e4ecfa` |
| B15 | Nit | Confirmed bug, fixed | `kalshi_tracking.py:159` | `3*0.2 == 0.6000000000000001`, so p=0.6 landed in the 40–60% bucket. | `test_a_prediction_exactly_on_a_bucket_edge_lands_in_the_upper_bucket`; `55af275` |
| B16 | Nit | Confirmed bug, fixed | `kalshi_tracking.py:221` | A blank ticker cell became the string `"nan"`, was counted as a contract, and was queried as `GET /markets/nan` on every run. | `test_report_does_not_query_kalshi_for_a_blank_ticker`; `0bc5412` |
| D1 | Medium | Design concern, **addressed** | `app.py:39` | `app.py` wrote the sidebar values into the shared `config` module. A new browser session opened with the previous session's pair and alert threshold, and two concurrent sessions could render or log one pair's data under the other's label. *Now:* each run works on a private copy of config.py's settings. The two tests that pinned the old behaviour assert the opposite. | `test_app.py::test_one_session_does_not_change_another_sessions_defaults` (the old demo printed "session B opened with pair = ETH-USD \| alert threshold = 7"); `457ec5b` |
| D2 | Medium | Design concern (known item 7), **addressed** | `backtest.py` (`effective_sample_size`) | Overlapping 5-candle outcomes, plus state votes that persist for many candles, meant the stated `n` overstated the evidence. Over 80 independent 4000-candle random walks with **no edge**, the median design effect was **2.8**, and **3.4–3.7 for SMA/EMA/MACD**: n≈1980 behaved like ≈570. *Now:* every bucket carries `n_eff`, a covariance-corrected effective size, exact under the no-edge null. Small-sample flags use it and the outlook quotes it. On the same 80 walks it tracks the empirical effective n: SMA bullish 568 vs 572, RSI bullish 169 vs 167, volume bullish 126 vs 122. | `test_backtest.py` (`n_eff` tests); `python review/backtest_overlap.py 80` (the `n_eff est` column); `a13bd89` |
| D3 | Medium | Design concern, **addressed** | `weighting.py` (`weight_details`) | Edge was measured against 0.5, not against the base rate for that direction in the backtested window. A no-skill "always bullish" call in a window drifting +5% would get ≈1.04. Sampling noise was larger still: with no edge at all, weights spread ±0.08 (2 sd) for the cross signals and ±0.16 for RSI. *Now:* edge = hit rate − that direction's base rate (agg_delta uses its own window's). A weight stays 1.0 unless the edge is ≥ `INDICATOR_WEIGHT_MIN_Z` (2) standard errors on `n_eff`. On the random walks, weights left 1.0 in 0–4% of runs. The weights table explains each weight, and the outlook says weighting changes nothing when no vote has a measured edge. | `test_weighting.py` (base-rate and noise-gate tests); `review/backtest_overlap.py` (the `weights != 1` column); `a13bd89` |
| D4 | Medium | Design concern, **addressed** | `kalshi.py` (`settlement_probability`), `kalshi_tracking.py` | (a) The model ignored the 60s averaging: under 1pt off at 14 min, 1–2pt at 5 min, 3–4pt at 2 min and up to 13pt at 1 min. *Now:* P(YES) prices the average, using the exact variance of the remaining prints. It matches a Monte Carlo of the rule to within 0.001 at 14, 5, 2 and 1 min. The point-in-time N(d2) is logged alongside as `p_yes_point`. In the final minute, prints already taken are assumed equal to spot; the output and the page say so. (b) Calibration scored the row *closest to close*, which is nearly decided. *Now:* it scores the row nearest 5 ± 1 min before close, and scores `p_yes_point` on the same rows for comparison. | `test_kalshi.py::test_settlement_probability_matches_monte_carlo_of_the_real_rule`, `test_kalshi_tracking.py::test_calibration_at_a_fixed_horizon_uses_the_row_nearest_it`; `python review/settlement_averaging.py`; `914f05e` |
| D5 | Medium | Hypothesis | `data_fetch.py:26`, `signals.py:216` | The live path evaluates the newest Coinbase candle, which I believe is still forming (partial volume, partial close). The backtest scores completed candles. The volume-spike vote is most affected, since a partial candle rarely reaches 2× average volume. The hit rates may therefore describe a slightly different signal from the live one. | Not verifiable here (Coinbase is blocked by the proxy). Check by comparing the last candle across two calls 10s apart. |
| D6 | Low | Design concern, **addressed** | `main.py` (`_is_repeat`) | The alert and signal logs appended on **every** Streamlit rerun: each 5s refresh, each widget click, each open tab. That duplicated alert rows while a condition held and inflated the n `analyze_correlation.py` reports. *Now:* a repeat of the log's last alert within `LOG_MIN_INTERVAL_SECONDS` (20s) is skipped, and the signal log takes at most one row per interval. A *different* alert is always written. The check reads the file's own last row, so it holds across tabs and the console bot. | `test_log_schema.py` (duplicate-suppression tests), `test_app.py::test_reruns_from_clicks_do_not_duplicate_log_rows`; `bb6dfd2` |
| D7 | **Medium** (was Low) | Confirmed bug, **fixed** | `csv_utils.py` (`append_row`, `file_lock`) | Concurrent writers (dashboard plus console, or several tabs) raced on every log, and it was worse than first reported. With 8 simultaneous writers, a new file got 7–8 header lines, after which `pd.read_csv` turns numeric columns into strings. A writer could read a half-written header and rotate a good file aside. On a layout change, the writers that lost the rename crashed, and 120–241 of 320 rows were lost. *Now:* every log appends through `append_row()`, which holds a cross-process lock file (stdlib `O_EXCL`, Windows-safe; stale after 10s) across the header check, rotation, duplicate check and write. Locked, every trial gives 1 header, 320 rows and at most 1 `.bak`. | `tests/test_csv_utils.py` (separate-process stress tests fail 8/8 runs with the lock disabled); `python review/csv_writer_race.py` |
| D8 | Low | Design concern | `backtest_agg_delta.py:145,186` | `candle_times` are candle **open** times, but the outcome is measured from `close[i]`, 60s later. The replayed agg_delta is therefore one candle staler than the live reading. This is not look-ahead; it biases the agg_delta hit rate (likely downward). | Code reading |
| D9 | Low | Design concern | `exchange_feeds.py:301,330` | `TradeAggregator` keeps trades in arrival order but prunes by exchange timestamp, stopping at the first unexpired trade. `snapshot()` prunes against the local clock. If venue latency differs, stale trades can linger. If the local clock skews by more than 60s, agg_delta empties or freezes. | Code reading |
| D10 | Low | Design concern | `app.py:271` | Kalshi is polled on every rerun (12/min per tab at a 5s refresh), and an HTTP 429 has been seen before. A `st.cache_data(ttl=…)` around `current_btc15m_market` would cap this. | Code reading |
| D11 | Low | Design concern | `app.py:145` | Feed handles are never evicted. Each pair visited keeps 5 trade sockets, 3 depth sockets (including Coinbase's full L2 book) and a sampler running for the life of the process. This is bounded by the 4-entry `PAIRS` list. | Code reading (the app's own comment acknowledges it) |
| D12 | Low | Design concern | `orderbook_feeds.py:219` | `LocalBook.last_update_ts` is set but never read. A frozen venue book stays in the aggregate until the WebSocket ping timeout (~30s) closes the socket. | Code reading |
| D13 | Low | Design concern | `divergence.py:45` | The bearish pattern is checked first, so a newer bullish divergence is masked whenever an older bearish one exists in the window. The same divergence is also re-reported, and re-counted in the backtest, for every candle until a new swing forms (part of D2). | Code reading |
| D14 | Nit | Design concern | `orderbook_feeds.py:478` | Even after B5, a one-tick reprice that crosses a 2bp log-bucket boundary resets a wall's age. Matching against the nearest live entry within ±2bp would remove this. | Code reading |
| D15 | Nit | Design concern | `signals.py:177` | `imbalance or 0.0` turns "never computed" into a neutral vote. No current caller reaches it, but it contradicts "absent, not neutral". | Code reading |
| D16 | Low | Design concern (known item 5) | `requirements.txt`, `app.py` (9× `use_container_width`) | All dependencies are `>=` with no lock file. Streamlit 1.67 warns that `use_container_width` "will be removed after 2025-12-31". Migrating to `width="stretch"` needs a higher `streamlit>=` floor (not verified which version). | Deprecation warnings in every `test_app.py` run |
| T1 | Medium | Test gap | `exchange_feeds.py` | The trade/liquidation parsers and `TradeAggregator` have **no tests**, including the Binance/Bybit `S` inversion in gotcha #3. (CLAUDE.md claimed "only their parsers and aggregators are covered"; I've corrected that.) | `grep -l parse_bybit_liquidation tests/*.py` → nothing |
| T2 | Low | Test gap | `orderbook_feeds.py`, `exchange_feeds.py` | The Coinbase and Bybit runner callbacks are untested. `_BinanceHarness` in `tests/test_orderbook_feeds.py` shows how to cover them without a socket. | — |

**Ruled out (checked, not a problem):**

- **No float-residue sibling in `TradeAggregator`.** It recomputes its sums on each snapshot rather than updating them incrementally.
- **Idle feeds don't hit `websocket.setdefaulttimeout(15)`.** websocket-client's `run_forever` reads only when its selector reports the socket readable, so quiet liquidation streams are not dropped.
- **Datetime conversions are safe.** Every `unit="s"` and `astype("int64")` site is either display-only or the already-fixed `astype("datetime64[ns]")`.
- **No look-ahead or off-by-one in `backtest.py`.** Signals at *i* read rows ≤ *i*. Swing confirmation needs `order` later points, which all sit inside a window ending at *i*. The loop ends at `n − lookahead − 1`.
- **`markets[0]` is fine.** The live API returned exactly one open KXBTC15M market.

## Critical and High: test, fix, result

All of these failed before their fix for the reason shown, and pass after. Every intermediate
commit was checked in a clean worktree with the full suite.

- **B1.** *Test:* switch to ETH-USD with logging on. Before the fix, Kalshi was called again
  (`assert 2 == 1`) and a row was logged. *Fix:* auto-track and logging only run when
  `product_id == "BTC-USD"`. Elsewhere the block is a plain calculator and a caption says so.
  *Result:* passes.
- **B2.** *Test:* edit the strike while auto-tracking, then refresh. *Before:* the log held
  `[75743.66, 75000.0, 75000.0]`. *Fix:* only log when `kalshi_result["strike"] == kalshi_live["strike"]`.
  *After:* `[75743.66]`.
- **B3.** *Test:* a second-by-second Monte Carlo of the rule quoted in the fixture's
  `rules_primary` (YES if the 60s average ≥ strike), at T = 1, 2 and 5 min with spot ±5bp from the
  strike. The true probability is further from 50% than the model by more than 5 SE in all six
  cases. A second test checks the Notes text. *Fix:* corrected the caveat in four places
  (docstring, README, CLI footer, dashboard) and gave its size. The model is unchanged; see D4.
- **B4.** *Test:* a real `BookAggregator` with two seeded venues on a fake clock. The bid-heavy one
  drops after 61s. *Before:* flow −0.693. *Fix:* each sample records per-venue notional, and flow
  is computed over venues present at both ends (unavailable if none). *After:* 0.0.
- **B5.** *Test:* two walls at different prices in one sample; then a wall held 30s that vanishes
  as a new one appears 80bp away. *Before:* 1 wall listed, and the new one aged 31s. *Fix:* key
  walls by `round(log(price) * 1e4 / MATCH_BPS)`. *After:* 2 walls, and the new one aged 0.

## Known items (CODE_REVIEW.md §4)

| # | Verdict |
|---|---------|
| 1 | **Still unresolved.** The proxy refuses both `external-api.kalshi.com` and Coinbase with 403, and local DNS resolves only allowlisted hosts, so this sandbox can't tell whether the host exists. `api.elections.kalshi.com` answered live on 2026-11-07 with 1 open KXBTC15M market in the fixture schema. **Recommendation:** switch `BASE_URL` to `api.elections.kalshi.com` unless a live check outside this sandbox shows `external-api` works. Run `python kalshi_api.py` once from your machine. |
| 2 | **Still true.** `SMA_FAST = 10` raises `KeyError: 'sma_9'`. Low severity, since it's a config edit that fails loudly. Fix by reading `f"sma_{cfg.SMA_FAST}"` etc. in `signals.py`. |
| 3 | **Still true. Defensible choice, but undocumented in the output.** Cutler's RSI (simple mean) is a legitimate variant, but it doesn't match the Wilder RSI on most charting sites, so the 30/70 thresholds behave differently. All-gains reporting **50** is a **latent bug**: a one-way run is the most overbought case and reads as neutral. It needs ≥14 consecutive up-closes on 1-min candles, so it's rare. Recommend `rsi = 100` when `avg_loss == 0 and avg_gain > 0`. |
| 4 | **Still true. Defensible**, but it understates spikes: the current candle inflates its own baseline by ~1/20 of its excess. Combined with D5, the live volume vote fires less often than the backtest assumes. |
| 5 | **Still true** (9 uses). See D16. |
| 6 | **Confirmed and fixed** (B13). |
| 7 | **Quantified** (D2): the median design effect is 2.8, and about 3.5 for the state signals. |

## What I could not verify

- **Live endpoints.** Coinbase (REST and WebSocket), Binance.US, Bybit and `external-api.kalshi.com`
  all return 403 from the proxy. Only `api.elections.kalshi.com` was reachable. Every exchange
  claim here comes from code reading or a mocked test, not observed live behaviour.
- **D5.** Whether Coinbase's newest candle is still forming.
- **Binance procedure.** Whether Binance.US's live diff-depth stream matches the documented procedure
  the B8 tests encode, for example whether `U`/`u` ever overlap across events.
- **Threads.** Real-thread timing in the runners. The B8 and B9 tests run the callbacks
  deterministically, inline.
- **Windows.** Append atomicity on Windows (relevant to D7) and the `py` launcher setup.
- **Visual layout.** Not checked; that needs a browser.

## Re-running the evidence

```bash
python -m pytest -q                                              # 466 passed
python review/settlement_averaging.py                            # B3 / D4: point vs averaged vs Monte Carlo
python review/backtest_overlap.py 80                             # D2 / D3: n_eff and noise gate (~4 min)
python review/csv_writer_race.py                                 # D7: unlocked vs append_row, 8 writers
```

D1 and D6 are now ordinary tests in `tests/test_app.py` and `tests/test_log_schema.py`.
