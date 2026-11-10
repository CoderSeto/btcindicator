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
CSV logs, is fixed too, and re-measured as worse than first reported.

**Third follow-up (run against the live APIs from the user's Windows machine):** the suite now passes on
Windows. Two portability bugs were fixed there: a test read `app.py` as cp1252, and `file_lock` crashed on a
"delete pending" lock. Known item 1 is closed: `external-api.kalshi.com` works. **B18** is new: Coinbase
caches `/candles` for ~5 minutes, so the header and Kalshi spot could be 5 minutes stale. Spot now comes
from `/ticker`, and live candle requests send a time range so they skip that cache.

**Edge study (2026-11-08):** is there a tradeable edge in Kalshi's KXBTC15M contracts? **No.**
Step 2: none of 120 pre-registered cells of the market's own prices survives costs on 30,688
contracts. Step 3: the dashboard's fair-value model never beats Kalshi's mid. It is significantly worse at
every horizon on train and at 5 and 2 minutes on held-out data, and none of 36 ways of trading
toward it is selected. Step 4 was therefore not
triggered. Step 3 also found that Kalshi's own records hold two strikes scaled by 10⁻⁴, which the
live dashboard would price as P(YES) = 100%. The dashboard now refuses such a strike and says why. Details in the
section "Is there an edge?" at the end.

**Scalping backtest (2026-11-08):** can a position be bought and sold back 1-3 minutes later for a
profit? **Not with the four entry rules tested.** All 96 train and held-out cells lose money, with
intervals entirely below zero, and none of the 48 pre-registered cells is selected. A trader who
knew each direction in advance would profit (+5 to +14¢ a contract), so the contract moves enough;
simple momentum and reversion rules just don't predict which way. Details in the section "Scalping
backtest" at the end.

**Second follow-up:** five more items are done: T1 (feed-parser tests), known item 2
(configured column names), D5 (forming candle), D10 (Kalshi poll) and D16 (deprecated API).
Testing the parsers turned up **B17**, a real bug: Coinbase trades were counted backwards in the
live agg_delta. T2 (runner-callback tests) is closed too. 508 tests pass. The rows below say what changed; the verdict is updated to match.

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
- **Trade flow.** The live `agg_delta` counted every Coinbase trade backwards (B17), because
  Coinbase's `side` is the maker's. Any live agg_delta reading or logged agg_delta vote from
  before this fix is suspect. Fixed, and all three venues are now tested to agree on what a taker
  buy looks like.

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
| B17 | **High** | Confirmed bug, fixed | `exchange_feeds.py` (`parse_coinbase_match`) | Coinbase's `match.side` is the **maker's** side, but it was read as the taker's. Every Coinbase trade, on the anchor venue for BTC-USD, was counted backwards in the live 3-venue `agg_delta`, and Coinbase whale trades were labelled backwards. Coinbase's own official client applies a match to the order tree for `side` and asserts that `maker_order_id` rests there. The agg_delta backtest (Binance.US only) was unaffected, so live and backtested agg_delta were measuring opposite things on Coinbase volume. | `test_exchange_feeds.py::test_coinbase_match_side_is_the_makers_so_the_taker_traded_the_other_way`, `::test_every_venue_reports_an_aggressive_buy_as_buy`; `4e70996` |
| B18 | **High** | Confirmed bug, fixed | `dashboard_data.py`, `app.py`, `main.py`, `kalshi.py` CLI | The header price, the console status line and the Kalshi model's spot were the newest candle's close, and Coinbase's `/candles` response is cached for about 5 minutes. Measured from the user's machine on 2026-11-07 (60 polls, 10s apart, against `/ticker`): the newest candle went 23:17 → 23:22 → 23:27, its close stayed frozen in between, and the price read off it was up to 314s old and **$48 (~6bp) off the ticker**. Near the close a few bp moves P(YES) by tens of points. Every row in an existing `kalshi_predictions.csv` priced a spot up to 5 minutes stale, so calibration built on those rows is suspect. *Now:* spot comes from `/ticker` (`data_fetch.latest_price`). If the ticker fails it falls back to the candle, and the page, console and CLI all say so. Volatility still comes from the candles. *Then the cause:* the bare `/candles?granularity=60` URL comes back `cache-control: public, max-age=300` and `cf-cache-status: HIT` (age 113s, newest candle 2 minutes behind). Over 18 polls in 3 minutes, the bare URL was a `HIT` stuck on 23:42 (age 113 → 288s). The same request with `start`/`end` was a `MISS` all 18 times, and 17 of the 18 included the current, still-forming minute. The ticker's last trade was 1.6–3.3s old throughout (also a CDN `HIT`, but cached for seconds). `get_candles` now always sends a range ending at the moment of the request, so the **indicators** stop lagging by up to 5 minutes too. | `test_app.py::test_header_and_kalshi_price_come_from_the_ticker_not_the_cached_candles`, `test_dashboard_data.py` (`latest_price`, `load_state`), `tests/test_live_price_paths.py` (console and CLI); 16 tests, all failing before the fix; `test_dashboard_data.py::test_get_candles_as_the_app_calls_it_never_sends_the_bare_cached_url` and 3 more for the range |
| D1 | Medium | Design concern, **addressed** | `app.py:39` | `app.py` wrote the sidebar values into the shared `config` module. A new browser session opened with the previous session's pair and alert threshold, and two concurrent sessions could render or log one pair's data under the other's label. *Now:* each run works on a private copy of config.py's settings. The two tests that pinned the old behaviour assert the opposite. | `test_app.py::test_one_session_does_not_change_another_sessions_defaults` (the old demo printed "session B opened with pair = ETH-USD \| alert threshold = 7"); `457ec5b` |
| D2 | Medium | Design concern (known item 7), **addressed** | `backtest.py` (`effective_sample_size`) | Overlapping 5-candle outcomes, plus state votes that persist for many candles, meant the stated `n` overstated the evidence. Over 80 independent 4000-candle random walks with **no edge**, the median design effect was **2.8**, and **3.4–3.7 for SMA/EMA/MACD**: n≈1980 behaved like ≈570. *Now:* every bucket carries `n_eff`, a covariance-corrected effective size, exact under the no-edge null. Small-sample flags use it and the outlook quotes it. On the same 80 walks it tracks the empirical effective n: SMA bullish 568 vs 572, RSI bullish 169 vs 167, volume bullish 126 vs 122. | `test_backtest.py` (`n_eff` tests); `python review/backtest_overlap.py 80` (the `n_eff est` column); `a13bd89` |
| D3 | Medium | Design concern, **addressed** | `weighting.py` (`weight_details`) | Edge was measured against 0.5, not against the base rate for that direction in the backtested window. A no-skill "always bullish" call in a window drifting +5% would get ≈1.04. Sampling noise was larger still: with no edge at all, weights spread ±0.08 (2 sd) for the cross signals and ±0.16 for RSI. *Now:* edge = hit rate − that direction's base rate (agg_delta uses its own window's). A weight stays 1.0 unless the edge is ≥ `INDICATOR_WEIGHT_MIN_Z` (2) standard errors on `n_eff`. On the random walks, weights left 1.0 in 0–4% of runs. The weights table explains each weight, and the outlook says weighting changes nothing when no vote has a measured edge. | `test_weighting.py` (base-rate and noise-gate tests); `review/backtest_overlap.py` (the `weights != 1` column); `a13bd89` |
| D4 | Medium | Design concern, **addressed** | `kalshi.py` (`settlement_probability`), `kalshi_tracking.py` | (a) The model ignored the 60s averaging: under 1pt off at 14 min, 1–2pt at 5 min, 3–4pt at 2 min and up to 13pt at 1 min. *Now:* P(YES) prices the average, using the exact variance of the remaining prints. It matches a Monte Carlo of the rule to within 0.001 at 14, 5, 2 and 1 min. The point-in-time N(d2) is logged alongside as `p_yes_point`. In the final minute, prints already taken are assumed equal to spot; the output and the page say so. (b) Calibration scored the row *closest to close*, which is nearly decided. *Now:* it scores the row nearest 5 ± 1 min before close, and scores `p_yes_point` on the same rows for comparison. | `test_kalshi.py::test_settlement_probability_matches_monte_carlo_of_the_real_rule`, `test_kalshi_tracking.py::test_calibration_at_a_fixed_horizon_uses_the_row_nearest_it`; `python review/settlement_averaging.py`; `914f05e` |
| D5 | Medium | Confirmed, **addressed** | `data_fetch.py` (`drop_forming_candle`) | The live path scored signals on the newest Coinbase candle, which can still be forming (partial close, partial volume). The backtest scored completed candles, so the hit rates described a different signal. *Since measured:* Coinbase does return the forming bucket. At 23:22:24 the newest candle was 23:22, 24s old, and that partial snapshot was then served unchanged for 5 minutes (see B18). *Now:* a candle stamped *t* is treated as forming while *t* + granularity > now, so the fix holds either way. Signals, the console loop and both backtests use completed candles only; the header price, console status line and Kalshi model keep the latest print. | `test_dashboard_data.py::test_signals_ignore_the_forming_candle_but_the_live_price_does_not`, `test_backtest.py::test_a_forming_last_candle_is_not_used_as_an_outcome`; `85fd22e` |
| D6 | Low | Design concern, **addressed** | `main.py` (`_is_repeat`) | The alert and signal logs appended on **every** Streamlit rerun: each 5s refresh, each widget click, each open tab. That duplicated alert rows while a condition held and inflated the n `analyze_correlation.py` reports. *Now:* a repeat of the log's last alert within `LOG_MIN_INTERVAL_SECONDS` (20s) is skipped, and the signal log takes at most one row per interval. A *different* alert is always written. The check reads the file's own last row, so it holds across tabs and the console bot. | `test_log_schema.py` (duplicate-suppression tests), `test_app.py::test_reruns_from_clicks_do_not_duplicate_log_rows`; `bb6dfd2` |
| D7 | **Medium** (was Low) | Confirmed bug, **fixed** | `csv_utils.py` (`append_row`, `file_lock`) | Concurrent writers (dashboard plus console, or several tabs) raced on every log, and it was worse than first reported. With 8 simultaneous writers, a new file got 7–8 header lines, after which `pd.read_csv` turns numeric columns into strings. A writer could read a half-written header and rotate a good file aside. On a layout change, the writers that lost the rename crashed, and 120–241 of 320 rows were lost. *Now:* every log appends through `append_row()`, which holds a cross-process lock file (stdlib `O_EXCL`, Windows-safe; stale after 10s) across the header check, rotation, duplicate check and write. Locked, every trial gives 1 header, 320 rows and at most 1 `.bak`. | `tests/test_csv_utils.py` (separate-process stress tests fail 8/8 runs with the lock disabled); `python review/csv_writer_race.py` |
| D8 | Low | Design concern | `backtest_agg_delta.py:145,186` | `candle_times` are candle **open** times, but the outcome is measured from `close[i]`, 60s later. The replayed agg_delta is therefore one candle staler than the live reading. This is not look-ahead; it biases the agg_delta hit rate (likely downward). | Code reading |
| D9 | Low | Design concern | `exchange_feeds.py:301,330` | `TradeAggregator` keeps trades in arrival order but prunes by exchange timestamp, stopping at the first unexpired trade. `snapshot()` prunes against the local clock. If venue latency differs, stale trades can linger. If the local clock skews by more than 60s, agg_delta empties or freezes. | Code reading |
| D10 | Low | Design concern, **addressed** | `kalshi_api.py` (`cached_btc15m_market`) | Kalshi was polled on every rerun of every tab (12/min per tab at a 5s refresh), and an HTTP 429 has been seen before. *Now:* at most one fetch per `KALSHI_POLL_SECONDS` (10) across the process. The countdown is recomputed from the clock on each call; a closed window is refetched at once; failures are cached too, so a 429 is not hammered. | `test_kalshi_api.py` (poll-cache tests), `test_app.py::test_reruns_share_one_kalshi_poll_within_the_poll_interval`; `cd22fd8` |
| D11 | Low | Design concern | `app.py:145` | Feed handles are never evicted. Each pair visited keeps 5 trade sockets, 3 depth sockets (including Coinbase's full L2 book) and a sampler running for the life of the process. This is bounded by the 4-entry `PAIRS` list. | Code reading (the app's own comment acknowledges it) |
| D12 | Low | Design concern, **fixed** | `orderbook_feeds.py` (`sample`) | `LocalBook.last_update_ts` was set but never read. A frozen venue book stayed in the aggregate until the WebSocket ping timeout (~30s) closed the socket. *Now:* a ready book unchanged for `BOOK_STALE_SECONDS` (15s) is left out of the sample, rejoins on its next update, and is listed in the snapshot's `frozen_venues`; the page shows it red as "frozen, not counted". | `test_orderbook_feeds.py` (4 tests: excluded, boundary, rejoins, all frozen), `test_app.py::test_a_frozen_depth_venue_shows_as_frozen_not_connected` |
| D13 | Low | Design concern, **masking fixed** | `divergence.py` (`detect_divergence`) | The bearish pattern was checked first, so a newer bullish divergence was masked whenever an older bearish one existed in the window. *Now:* both are looked for and the one completed most recently is reported. Re-run `python backtest.py` to refresh the divergence buckets. *Still true:* the same divergence is re-reported, and re-counted in the backtest, for every candle until a new swing forms. The backtest's `n_eff` (D2) already discounts that repetition. | `test_divergence.py::test_a_newer_bullish_divergence_is_not_masked_by_an_older_bearish_one` |
| D14 | Nit | Design concern | `orderbook_feeds.py:478` | Even after B5, a one-tick reprice that crosses a 2bp log-bucket boundary resets a wall's age. Matching against the nearest live entry within ±2bp would remove this. | Code reading |
| D15 | Nit | Design concern, **fixed** | `signals.py` (`_orderbook_signal`) | `imbalance or 0.0` turned "never computed" into a neutral vote. No current caller reached it, but it contradicted "absent, not neutral". *Now:* with no live book and no REST reading the order-book vote is left out, and the loggers write a blank cell; a real 0.0 reading still votes neutral. | `test_signals_book.py::test_no_book_reading_at_all_leaves_the_vote_out_rather_than_neutral` |
| D16 | Low | Design concern (known item 5), **addressed** | `requirements.txt`, `app.py` | The 9 `use_container_width` uses are now `width="stretch"`. Inspecting the published wheels, all three affected APIs accept it from Streamlit **1.51.0**, so the floor is now `streamlit>=1.51`; the dashboard tests pass on 1.51.0 exactly, and a test fails on any Streamlit deprecation warning. *Now also:* `requirements-lock.txt` pins all 41 packages (direct and transitive) at the versions the full suite passed with in a clean environment built from `requirements.txt`. `requirements.txt` keeps the minimums. `tests/test_requirements_lock.py` fails if a requirement isn't pinned or a pin is below its minimum. The lock was resolved on Linux/Python 3.13, so confirm it on Windows with `py -m pip install -r requirements-lock.txt` and `py -m pytest`. | `test_app.py::test_the_page_uses_no_deprecated_streamlit_apis`; `500aa01` |
| T1 | Medium | Test gap, **closed** — and it hid a bug (B17) | `exchange_feeds.py` | The trade/liquidation parsers and `TradeAggregator` had no tests, including the Binance/Bybit `S` inversion in gotcha #3. They're now tested against each exchange's own published payloads. The docs sites were unreachable, so Binance and Bybit payloads come verbatim from their docs' GitHub sources, and Binance futures from its official SDK; each is pinned by commit. | `tests/test_exchange_feeds.py`; `4e70996` |
| T2 | Low | Test gap, **closed** | `orderbook_feeds.py`, `exchange_feeds.py` | The Coinbase and Bybit runner callbacks were untested, and so were the Binance.US trade and Binance liquidation runners. *Now:* `tests/test_feed_runners.py` drives every runner's real callbacks with a fake socket. It checks what each subscribes to and on which URL (including gotcha #4's `level2_batch`), and what each message type, error and close does: trades counted on the taker side, the liquidation symbol filter, books built from the snapshot and patched by updates, discarded on close, and overwritten on Bybit's `u`=1 restart. The Bybit delta is the docs example verbatim. Five deliberate code mutations are each caught by a test; no new bug turned up. | `tests/test_feed_runners.py` |

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
| 1 | **Resolved: `BASE_URL` is correct, no change.** The sandbox's proxy refused `external-api.kalshi.com`, so it couldn't be checked here. On 2026-11-07 the user ran `py kalshi_api.py` from their own machine against `external-api.kalshi.com`. It returned the open market `KXBTC15M-26NOV071830-30`: strike $75,662.60, close 23:30 UTC (18:30 ET, matching the ticker), 7.9 min left, YES $0.32/$0.33, last $0.33. `api.elections.kalshi.com`, where the fixtures were captured, serves the same schema and stays a working fallback. |
| 2 | **Fixed** (`afb98a5`). `signals.py` now reads `sma_{SMA_FAST}` etc., and the dead crossover branches are gone. Covered by `test_signals_price.py::test_the_cross_votes_follow_the_configured_periods`. |
| 3 | **Still true. Defensible choice, but undocumented in the output.** Cutler's RSI (simple mean) is a legitimate variant, but it doesn't match the Wilder RSI on most charting sites, so the 30/70 thresholds behave differently. All-gains reporting 50 was a latent bug: a one-way run is the most overbought case and read as neutral. **Fixed:** gains with no losses now read 100. A flat window and the warm-up candles stay at 50. `test_indicators.py::test_rsi_all_gains_is_100_not_neutral`, `::test_a_one_way_run_votes_overbought` (a 14+ candle up-run now casts RSI's overbought vote). The Cutler-vs-Wilder point stands. |
| 4 | **Still true. Defensible**, but it understates spikes: the current candle inflates its own baseline by ~1/20 of its excess. Combined with D5, the live volume vote fires less often than the backtest assumes. |
| 5 | **Fixed** (`500aa01`). See D16. |
| 6 | **Confirmed and fixed** (B13). |
| 7 | **Quantified** (D2): the median design effect is 2.8, and about 3.5 for the state signals. |

## What I could not verify

- **Live endpoints.** Coinbase (REST and WebSocket), Binance.US, Bybit and `external-api.kalshi.com`
  all return 403 from the proxy. Only `api.elections.kalshi.com` was reachable. (`external-api` has
  since been confirmed live from the user's machine; see known item 1.) Every exchange
  claim here comes from code reading or a mocked test, not observed live behaviour.
- **D5.** ~~Whether Coinbase's newest candle is still forming.~~ It is; see D5 and B18.
- **Binance procedure.** Whether Binance.US's live diff-depth stream matches the documented procedure
  the B8 tests encode, for example whether `U`/`u` ever overlap across events.
- **Threads.** Real-thread timing in the runners. The B8 and B9 tests run the callbacks
  deterministically, inline.
- **Windows.** Append atomicity on Windows (relevant to D7) and the `py` launcher setup. *Since checked:* the
  first Windows run of the suite crashed D7's stress-test writers with `PermissionError` on
  `log.csv.lock`. Windows refuses to re-create a file that is "delete pending" (deleted while
  another process still had it open), and `file_lock` only expected `FileExistsError`. The same
  crash would have hit the dashboard and console bot logging together. It's now treated as a busy
  lock on Windows; a lock whose age can't be read no longer spins forever; and a refused release is
  retried. Tests in `tests/test_csv_utils.py` simulate the refusal on any OS.
- **Visual layout.** Not checked; that needs a browser.

## Re-running the evidence

```bash
python -m pytest -q                                              # 616 passed, 2 skipped
python review/settlement_averaging.py                            # B3 / D4: point vs averaged vs Monte Carlo
python review/backtest_overlap.py 80                             # D2 / D3: n_eff and noise gate (~4 min)
python review/csv_writer_race.py                                 # D7: unlocked vs append_row, 8 writers
```

D1 and D6 are now ordinary tests in `tests/test_app.py` and `tests/test_log_schema.py`.

## Is there an edge? The KXBTC15M study (2026-11-08)

**Question:** is there a tradeable edge in Kalshi's 15-minute BTC contracts, after real costs?
**Step 2 answer: not in the market's own prices.** Buying YES at the ask or NO at 1 − bid, in any
price bucket, at 10, 5 or 2 minutes before close, loses money or breaks even on the training data.
Not one of the 120 cells passes the selection rule, so nothing goes to the held-out set as a
claim. Step 3 (does `kalshi.settlement_probability` beat the market?) follows below.

Code: `kalshi_study.py` (tested: `tests/test_kalshi_study.py`, `tests/test_kalshi_study_model.py`).
Analysis: `python review/kalshi_market_study.py fetch|step2|step3|sources`. Raw data is cached in
`study_data/` (gitignored).

### Design, fixed before any outcome was seen

The design is pinned in `kalshi_study.py`'s docstring. It was committed in `f8a4da5` before any data
had been fetched, and amended twice before any outcome had been looked at (only raw quotes had been
seen, enough to notice that December 2025 books were thin). The amendments are labelled in the
docstring.

- **Data.** Kalshi lists 31,046 settled KXBTC15M markets, opening from 2025-12-10. All settled yes
  or no, but 358 have no `floor_strike` in Kalshi's record: every market from 2025-12-10 to 12-15,
  plus a scattered few up to 2026-08-14. Those are dropped, so the **30,688 contracts studied close
  2025-12-15 → 2026-11-08 UTC**. (An earlier message gave 2025-12-10; that was the first listed
  open, not the first studied close.) 84 contracts have no candles at any endpoint.
- **Two candle sources.** Markets settled before `/historical/cutoff` (2026-09-08) come from the
  per-market `/historical/` endpoint: all of train, and held-out contracts up to the cutoff (3,250
  at 2 minutes). Later ones come from the batch endpoint. **The two agree.** For 50 markets both
  serve, closing 2026-09-01 → 09-07, bid and ask are identical at all 150 contract-horizons
  (`python review/kalshi_market_study.py sources`). Train-vs-test differences below are not a
  data-source artifact.
- **Split.** Chronological. Contracts closing from 2026-08-02 21:00 UTC (the last 30%, 9,207
  contracts, 99 days) are held out. Training is the first 21,481 contracts over 231 days.
- **One observation per contract per horizon** (10, 5 and 2 minutes before close). Each is the quote
  standing at that minute: the closing bid/ask of the 1-minute candle whose period *ends* then.
  Tests pin that it is never the next candle.
- **Real book:** `0 < bid < ask < 1` and `ask − bid ≤ 20¢`. The spread cap was the second amendment.
  Without it, a placeholder such as bid 0.001 / ask 0.55 counted as a book with a "mid" of 0.28.
- **Costs.** YES is bought at the ask, NO at 1 − bid, plus Kalshi's taker fee
  `round up(0.07 × C × P × (1−P))`. That is the formula in Kalshi's own schedule
  (kalshi.com/docs/kalshi-fee-schedule.pdf, "Last updated and effective: July 7, 2026"); there is no
  settlement fee. KXBTC15M isn't on the non-standard list, and `GET /series/KXBTC15M` reports
  `fee_type: quadratic, fee_multiplier: 1`. The schedule's text says the round-up is "to a
  centicent", but its printed table rounds each order to the cent ($0.02 for one contract at 50¢).
  This study uses the cent, the costlier reading, **per order**. Orders of **1 contract** (2¢ at
  50¢) and **100 contracts** (1.75¢ each) are both reported in full; neither size was chosen by its
  results. A test reproduces every row of Kalshi's printed table.
- **Fills** are assumed at the quoted top of book. Candlesticks carry no depth, so the size really
  available there may be small, especially for 100 contracts. That makes every EV here an upper
  bound on what a real order would get.
- **Statistics.** Intervals are a block bootstrap by UTC day (2,000 resamples), shown as
  mean ± z·SE. **z = 3.62 is one Bonferroni correction over every test in Steps 2 and 3 together:
  168.** That is 120 Step 2 cells (3 horizons × 10 buckets × 2 sides × 2 order sizes), 36 Step 3
  trading variants and 12 Step 3 scoring comparisons. Each bucket also shows its **minimum detectable
  edge (MDE = z × SE)**, the smallest edge per contract it could have detected. `n_eff` is the
  sample size adjusted for same-day correlation.
- **Buckets** are fixed 10¢ bands of the mid, constants rather than quantiles of any data, so train
  and test are bucketed identically.
- **Selection rule.** A cell counts only if its train interval lies entirely above zero. Only such
  cells are then scored on the held-out set. The test tables print for all 120 cells, but they are
  descriptive, and nothing below is claimed from them.

### Step 2 result: no cell selected

**Selected on train: none of 120.** Every train cell's family interval includes or lies below zero.
These are the train cells with the highest point estimates; none clears its interval:

| Train, 100-contract orders | n | EV per contract | Family interval | MDE |
|---|---|---|---|---|
| 10 min, mid 0.9–1.0, buy YES | 834 | +0.4¢ | [−2.2, +3.1]¢ | 2.7¢ |
| 10 min, mid 0.0–0.1, buy NO | 780 | +0.3¢ | [−2.6, +3.1]¢ | 2.9¢ |
| 5 min, mid 0.9–1.0, buy YES | 4,599 | +0.3¢ | [−0.6, +1.1]¢ | 0.8¢ |
| 5 min, mid 0.0–0.1, buy NO | 4,484 | +0.2¢ | [−0.8, +1.2]¢ | 1.0¢ |
| 2 min, mid 0.2–0.3, buy NO | 624 | +1.3¢ | [−4.3, +6.8]¢ | 5.6¢ |
| 2 min, mid 0.0–0.1, buy NO | 5,914 | +0.1¢ | [−0.5, +0.7]¢ | 0.6¢ |

With 1-contract orders every one of these is lower, by 0.25–0.7¢ per contract: on a single
contract the fee rounds up to a whole cent. Most cells lose 2–5¢ per contract, which is roughly the spread plus the fee.

**What the null rules out.** Net of costs, at 100-contract orders, on training data:

- **Extreme buckets (mid below 0.1 or above 0.9) at 5 and 2 minutes:** an edge of more than about
  **1¢ per contract** in either direction (MDE 0.6–1.0¢).
- **Middle buckets at 10 minutes:** an edge of more than about **3¢** (MDE 2.5–3.4¢).
- **Middle buckets at 2 minutes:** only edges above **3.5–8.4¢**. Few contracts are still near 50%
  that late, so these cells are thin.

A positive edge smaller than the MDE is not ruled out. It would also have to survive real fill
sizes, which this data can't see.

**Favorite-longshot bias: it shows on train but does not hold on held-out.** The logistic
calibration slope of outcome on logit(mid) is 1 for a calibrated price and above 1 when favorites
are underpriced:

| | 10 min | 5 min | 2 min |
|---|---|---|---|
| Train slope ± SE | 1.058 ± 0.014 (z +4.1) | 1.065 ± 0.013 (z +5.0) | 1.073 ± 0.019 (z +3.8) |
| Test slope ± SE | 1.071 ± 0.029 (z +2.5) | 1.010 ± 0.018 (z +0.6) | 0.921 ± 0.016 (z **−4.9**) |

These six fits are outside the 168 counted tests. Counting them makes 174, and z barely moves
(3.63). On train, all three horizons show a mild favorite-longshot bias. On the held-out months it
is not significant at 10 or 5 minutes, and at 2 minutes it has reversed: prices are too extreme. A
pattern that flips sign out of sample is not an edge. Either way it is worth at most a few cents on
the extreme buckets, which is about what the spread and fee take.

**The held-out 2-minute reversal (slope 0.921) is a hypothesis, not a finding.** It was found by
looking at held-out data, so this dataset can't test it. It can only be tested forward, on new
contracts, in pre-registered paper trading. It is not a data-source artifact: the slope is below 1
both before the cutoff (0.876 ± 0.023, n 3,250, `/historical/`) and after it (0.953 ± 0.020,
n 5,438, batch). By calendar month it is 0.906 ± 0.027 for August and 0.928 ± 0.019 for
September–November.

### Exclusions and whether they depend on the outcome

Rows are dropped for three reasons: no candle at that exact minute (`no_quote`), a missing side
(`one_sided`), or a spread over 20¢ (`wide_spread`). Per month, from `step2`:

- **December 2025** is the worst month. At 10 minutes, 136 of 1,304 rows are dropped (28 no quote,
  26 one-sided, 82 wide spread). The median spread at 5 minutes was 4¢, and only 26% of books were
  2¢ or tighter.
- **January and February 2026** have 1–2% dropped at 10 minutes. From March on, almost nothing is
  dropped at 10 minutes. At 5 minutes, 3–5% are one-sided through March and almost none from April.
  The held-out months drop nothing at all at 10 or 5 minutes. Median spreads tighten to 1¢ by
  February.
- **The spread cap** removed 91, 85 and 89 rows at the three horizons. 86% of them (82, 68 and 77)
  are from December 2025, and it removed none after May.
- **At 2 minutes, 15% of rows are dropped**, almost all one-sided: by then many contracts are
  decided and one side of the book is gone. The 0.0–0.1 and 0.9–1.0 buckets at 2 minutes therefore
  describe contracts that still had both sides quoted, not the most decided ones.

**Missingness does not depend on the outcome.** These are YES-settled minus NO-settled exclusion
rates, with z against the family threshold of 3.62:

| | 10 min | 5 min | 2 min |
|---|---|---|---|
| All rows | +0.14 pt (z 1.5) | +0.22 pt (z 1.2) | +0.86 pt (z 2.2) |
| Spread cap alone | +0.09 pt (z 1.4) | 0.00 pt (z −0.1) | −0.06 pt (z −1.2) |

No gap is significant, and none is large enough to move a bucket's YES rate by more than a fraction
of its interval.

### Step 3 result: the model is worse than the market, and trading toward it loses

**Question:** does `kalshi.settlement_probability`, fed Coinbase prices the way the dashboard feeds
it, beat Kalshi's own mid? **No.** The market mid scores better at every horizon on train. On
held-out data it scores better at 5 and 2 minutes, and at 10 minutes the basis-adjusted model is
indistinguishable from it. The model is never better. None of the 36 trading variants is selected
on train.

**Setup.** For each contract and horizon:

- **Spot** is the close of the Coinbase 1-minute candle that *started* a minute before the horizon.
  Tests pin that the candle starting at the horizon is never used.
- **Vol** is the realized vol of the prior 120 one-minute returns, via the dashboard's own
  `kalshi.realized_volatility`.
- **Two spot variants:** raw Coinbase, and "basis-adjusted". The adjusted one rescales spot by
  strike ÷ (the Coinbase proxy for the strike's own 60-second average). That removes the
  Coinbase-vs-index gap as it stood at open, using only what was known at open.
- **Coverage.** Coinbase history is 471,969 candles with 467 minutes missing. The model could not
  price 36–41 valid rows per horizon. Those don't depend on the outcome: the YES-vs-NO gap is at
  most z 1.8 in size.

**Scoring against the mid, on the same rows.** Each row is model minus market, so positive means the
model is worse. Intervals use z = 3.62.

| | n | Brier, market | Brier diff, raw | Brier diff, adjusted | Log-loss diff, adjusted |
|---|---|---|---|---|---|
| Train, 10 min | 21,216 | 0.1877 | +0.0022 [+0.0010, +0.0034] | +0.0022 [+0.0011, +0.0033] | +0.0066 [+0.0034, +0.0099] |
| Train, 5 min | 20,807 | 0.1168 | +0.0040 [+0.0027, +0.0053] | +0.0041 [+0.0028, +0.0054] | +0.0136 [+0.0088, +0.0183] |
| Train, 2 min | 17,217 | 0.0686 | +0.0063 [+0.0044, +0.0081] | +0.0060 [+0.0042, +0.0079] | +0.0195 [+0.0131, +0.0260] |
| **Test, 10 min** | 9,207 | 0.1912 | +0.0020 [+0.0001, +0.0038] | +0.0015 [−0.0006, +0.0036] | +0.0049 [−0.0020, +0.0118] |
| **Test, 5 min** | 9,207 | 0.1283 | +0.0046 [+0.0020, +0.0071] | +0.0044 [+0.0019, +0.0069] | +0.0165 [+0.0068, +0.0262] |
| **Test, 2 min** | 8,688 | 0.0799 | +0.0071 [+0.0039, +0.0103] | +0.0066 [+0.0036, +0.0096] | +0.0298 [+0.0176, +0.0419] |

- **Held-out:** the market is significantly better at 5 and 2 minutes on both metrics. At 10
  minutes the raw model is worse and the adjusted one is not distinguishable from the market.
- **The gap grows toward the close.** Spot is the last Coinbase trade just before the horizon (the
  close of the candle [t − 60, t)), so it is not stale. What hurts late in the window is the data
  itself: minute-level Coinbase data, and the Coinbase-vs-index basis (next section). Both matter
  most once the settlement average is only a few basis points from the strike.
- **Basis adjustment helps marginally.** It improves log-loss by 0.0006–0.003, nowhere near enough to
  catch the market.
- **Calibration.** The adjusted model's calibration slope on train is 1.07, 1.20 and 1.32 at 10, 5
  and 2 minutes: too close to 50%, more so near the close. On held-out it is 0.98, 1.01 and 0.92.
- **Size of disagreement.** The model and the mid differ by 4–5¢ on average. That is the model's
  noise, not edge.
- **Counting.** These 24 intervals include the 12 train comparisons, which the 168-test family
  counted only once. Counting all 24 gives 180 tests and z = 3.635. No interval changes sign: the
  borderline raw-model 10-minute held-out intervals keep lower bounds of +0.00009 (Brier) and
  +0.00014 (log-loss).

**Trading toward the model.** Buy whichever side clears the ask (or 1 − bid) plus the per-order fee
by an extra margin. The grid is margins 0, 2¢ and 5¢ × raw or adjusted spot × 3 horizons × order
sizes 1 and 100: 36 variants.

- **Selected on train: none.** All 36 train means are negative, from −0.3¢ to −2.1¢ per contract.
  About half of the intervals (19 of 36) lie entirely below zero; none lies above.
- **The ones closest to zero** are 100 contracts, raw spot, 5¢ margin, 10 minutes: −0.3¢ over
  3,765 trades, interval [−2.6, +2.0]¢, MDE 2.3¢. And 100 contracts, adjusted, 0 margin, 2 minutes:
  −0.7¢ [−1.4, +0.1]¢.
- **What this rules out.** Following the model loses money on train, at about the cost of the
  spread. The trading MDEs say no variant has an edge above roughly 0.7–2.9¢ per contract on train.
- **Held-out** is printed in `step3` and is descriptive only. No variant was selected, so nothing is
  claimed from it.

### The Coinbase-vs-index basis

Kalshi publishes the settlement index's own 60-second averages: `floor_strike` (at open) and
`expiration_value` (at close). Each is compared with a Coinbase proxy, the mean of the open, high,
low and close of the matching minute. Units are basis points of (index ÷ Coinbase − 1).

| | n | median | p5 | p95 | mean |·| |
|---|---|---|---|---|---|
| Train, at close | 18,935 | +0.77 | −1.43 | +3.03 | 1.25 |
| Test, at close | 9,205 | +0.45 | −1.16 | +2.12 | 0.87 |

- **The index runs slightly above Coinbase,** by about half a basis point, and the gap typically
  moves by 1–3 bp over a contract.
- **There is a ceiling on any Coinbase-minute model.** Knowing Coinbase's final minute exactly still
  calls only **96.7%** of contracts right (of 28,137 with both values). The other 3.3% are decided
  inside the Coinbase-vs-index noise, plus the proxy's own error, since a 1-minute OHLC mean is not a
  60-print average. Those are close calls a Coinbase-fed model cannot get right, and they are the
  contracts whose prices sit near 50% late in the window.
- **Data gaps and errors.** `expiration_value` is missing for 2,520 contracts: all of December 2025
  and 1,212 of January 2026's. The basis exceeds 20 bp for 11 contracts. Nine of them come from 5 fast-moving minutes,
  such as the 2026-02-05 selloff, where the proxy is poor (a contract's close is the next one's open,
  so one bad minute shows twice). The other 2 are Kalshi data errors (next item).

### Data error in Kalshi's own records (affects the live dashboard)

**Two contracts' `floor_strike` is the true strike divided by 10⁴.**
`KXBTC15M-26APR131115-15` has strike 7.2437; the true strike is 72,437.29, which is the previous
contract's `expiration_value`, and the settled outcome matches it. `KXBTC15M-26APR131130-30` has
7.1988 instead of 71,987.73.

- **Effect on this study.** The model priced both at P = 1.0. Both are in train. Only
  `…1115-15` settled NO, costing the clipped −ln(10⁻⁴) = 9.21 in log-loss and 1 in Brier;
  `…1130-30` settled YES and cost nothing. Per horizon (about 17,000–21,000 rows) that adds
  0.00043–0.00053 to the model's log-loss and 0.000047–0.000058 to its Brier, measured. Had both
  settled NO it would be about twice that. They are left in, as run, and change no conclusion.
  Step 2 used market prices only and is unaffected.
- **Effect on the live tool.** `kalshi_api.parse_market` reads the same field, so the same error on a
  live contract would show **P(YES) = 100%**. *Now:* `kalshi.fair_value` refuses a strike more than
  `KALSHI_MAX_STRIKE_DISTANCE` (5%) from spot. The page then shows why, instead of a P(YES), and
  nothing is logged. Tests use both real cases: `test_kalshi.py` (strike guard) and
  `test_app.py::test_a_bad_strike_from_kalshi_shows_the_reason_instead_of_a_certain_p_yes`, which
  fails with `'100.0%'` when the guard is off.

### Step 4: not triggered

Step 4 (model improvements, then paper trading) was conditional on Step 2 or 3 showing something on
held-out data. Neither did: no cell or variant passed the train rule, so nothing reached the
held-out test.

**Bottom line:** taking prices at the top of the book and paying Kalshi's fees, this data shows no
edge, either in the market's own prices or in the dashboard's fair-value model. The market mid is at
least as good a probability as the model at every horizon tested, and significantly better at most. Don't trade the model's P(YES) against
Kalshi's quote.

**What would be needed to look further:**

- **A model that beats the mid on held-out Brier first,** before any trading. The obvious inputs
  are a real per-second BRTI or multi-venue feed instead of Coinbase minutes, and, inside the final
  minute, the running settlement average.
- **Maker strategies,** which need order-book depth and queue data that candlesticks don't have.

```bash
python review/kalshi_market_study.py step3    # needs the fetch; first run builds study_data/model_obs.csv (~10 min)
```

### Limits (both steps)

- **Fill size is unknown.** Every EV assumes the full order fills at the top-of-book price.
- **The market changes.** Fees, tick sizes and market makers can change; held-out spreads are
  already tighter than train's (1¢ vs 1.7–2¢ median).
- **Order type.** Only taker orders were tested. Resting maker orders (no fee on this series, and
  they earn the spread) are a different strategy, needing queue and fill data this study doesn't
  have.
- **Data window.** Eleven months. The series' listing may go back further than the historical
  listing returned.

```bash
python review/kalshi_market_study.py fetch    # resumable; hours the first time (the /historical/ endpoint is rate-limited)
python review/kalshi_market_study.py step2
python review/kalshi_market_study.py sources  # /historical/ vs batch quotes; held-out 2-min slope by source
```


## Scalping backtest (2026-11-08)

**Question:** does buying a side of a KXBTC15M contract and selling it back 1, 2 or 3 minutes later
earn more than the round trip costs? **Answer: not on any of the four entry rules tested.**

Code: `kalshi_scalp_study.py` (tested: `tests/test_kalshi_scalp_study.py`,
`tests/test_kalshi_scalp_backtest.py`). Run: `python review/kalshi_scalp_backtest.py`, on the
`study_data/` cache. The design is pinned in `kalshi_scalp_study.py`'s docstring and was committed
(ce5ddcf) before any scalp result had been computed.

### Design, fixed before any result was seen

- **Data, split, costs, intervals:** the edge study's. 30,688 contracts closing 2025-12-15 to
  2026-11-08; the last 30% (9,207, closing from 2026-08-02) held out; YES bought at the ask and
  sold at the bid, NO bought at 1 - bid and sold at 1 - ask; Kalshi's taker fee on both legs, rounded
  up to the cent per order; fills at the quoted top of book with zero latency; day-block bootstrap.
- **Trades:** one per contract per cell, entered 10 or 5 minutes before close and sold 1, 2 or 3
  minutes later. If the exit quote is not a real book the position is held to settlement instead and
  counted (`forced`), never dropped.
- **Entry rules** (information at entry only): `contract_momentum` / `contract_reversion` (the
  contract's mid moved at least 2¢ over the last 2 minutes; buy with or against it) and
  `btc_momentum` / `btc_reversion` (Coinbase moved at least 2 bp over the last 2 minutes).
- **Family:** 4 rules x 2 entry times x 3 holds x 2 order sizes (1 and 100) = **48 tests**, Bonferroni
  z = 3.28. Selected on train only if the interval lies entirely above zero; only selected cells are
  then scored on the held-out set.

### Result 1: the round trip costs 4-5¢, and the contract moves far more than that

Over every contract with a real book at both ends (no rule, descriptive):

| entry / hold | n | round-trip cost (1-lot) | mean abs move of the mid | moves further than cost | ORACLE ceiling (1-lot) |
|---|---|---|---|---|---|
| 10 / 1 min | 30,396 | 5.0¢ | 8.5¢ | 61.5% | +4.6¢ |
| 10 / 2 min | 30,370 | 4.8¢ | 12.2¢ | 73.4% | +8.1¢ |
| 10 / 3 min | 30,326 | 4.8¢ | 15.3¢ | 80.3% | +11.1¢ |
| 5 / 1 min | 29,509 | 3.9¢ | 8.4¢ | 58.3% | +5.3¢ |
| 5 / 2 min | 28,459 | 3.8¢ | 12.7¢ | 71.7% | +9.5¢ |
| 5 / 3 min | 25,859 | 3.7¢ | 17.4¢ | 81.2% | +14.0¢ |

Cost is (spread in + spread out) / 2 plus both fees at the mids: how far the mid must move in your
favour just to break even. The 100-lot cost is lower (2.5-4.0¢, since the per-order fee rounds up
less per contract). The ORACLE is look-ahead profit per contract if the winning side were known in
advance and the trade skipped when both lose: a ceiling on any direction rule, not a strategy. It is
large, so **the opportunity exists in hindsight; the question is only whether direction is
predictable.** As a rough back-of-envelope, if every move were the mean size, a trader would need to
call the direction right about four times in five at a 1-minute hold, and about two in three at a
3-minute hold, just to break even.

### Result 2: no rule beats the cost, in train or held out

- **Selected on train: none of 48.** Every train cell's interval lies below zero (best: contract
  momentum, entry 5, hold 3, 100-lot, -2.1¢ [-2.8, -1.4]).
- **Held out, descriptive:** all 48 cells are also negative with intervals below zero (best:
  contract momentum, entry 5, hold 3, 100-lot, -1.0¢ [-1.8, -0.1]; most cells -2 to -5¢).
- **The null is tight.** Each cell's minimum detectable edge is 0.4-0.9¢ a contract, so an edge of a
  cent or more in these cells would have shown.
- **The rules carry almost no direction.** About half of each rule's trades are YES. A rule and its
  mirror trade the same contracts on opposite sides, so half the difference between their mean
  profits (1-lot) estimates the directional information per trade: between **-1.0¢ and +0.55¢**
  across all 24 train and held-out comparisons (e.g. 10 / 1: reversion beats momentum by ~0.5¢ on
  train, 0.3¢ held out; late entries lean the other way, contract momentum beating reversion by
  ~1.0¢ at 5 / 3 held out). That is the whole signal, against a 4-5¢ round trip. Not a counted
  finding (the cells were not selected), but it is the right size to explain the null.

### Reading the best-looking cells

The least-bad cells are late entries with long holds (5 / 3). They are also the cells with the most
**forced holds**: 15.8% of train entries and 4.4% of held-out ones had no sellable book at the exit
(2 minutes before close) and were held to settlement, paying one fee and no exit spread. So part of
their relative improvement is escaping the second leg of the cost, not skill, and the edge study
already found holding to settlement loses too. Across all contracts, forced holds are 0.2-0.4%
at entry 10, and 1.8-5.3% at entry 5 for 1-2 minute holds (14.0% at 3 minutes).

### What this does and does not rule out

- **Rules out:** these four entry rules at these times and holds, for taker scalps at the quoted top
  of book. Every simplification (zero latency, full fill at the touch, one-minute decisions) flatters
  a scalp, so real results would be worse, not better.
- **Does not rule out:** exits that react inside a minute, take-profit and stop-loss exits, other
  entry times, entry signals richer than a 2-minute momentum (for example the order-book signals,
  which have no historical data), or resting maker orders (no fee on this series; needs queue and
  fill data this study doesn't have). Each would be a counted amendment, not a change to this
  design after the fact.
- **Consequence for the bot:** the paper-trading tab stays hold-to-settlement. A Close button would
  let someone try scalping by hand, but this backtest gives no reason to expect it to pay.

```bash
python review/kalshi_scalp_backtest.py        # needs the study_data/ cache from kalshi_market_study.py fetch
```


## Four further strategy hypotheses (2026-11-10)

**Question:** does any of four new entry signals, traded in either direction, earn more than it
costs on KXBTC15M? **Answer: no. None of the 40 cells is selected on train, so nothing goes to the
held-out set as a claim.**

Code: `kalshi_strategy_study.py` (tested: `tests/test_kalshi_strategy_study.py`,
`tests/test_kalshi_strategy_backtest.py`). Run: `python review/kalshi_strategy_backtest.py`. The
design is pinned in the module docstring and was committed (92ad35a) before any study data had been
fetched in the environment it was written in. No amendment was made. The held-out set was run once.

### Design, fixed before any result was seen

- **Data, split, costs, intervals:** the edge study's own functions. YES at the ask, NO at 1 − bid,
  the taker fee rounded up to the cent per order on every leg, fills at the quoted top of book,
  day-block bootstrap. The cache was rebuilt on 2026-11-10: **30,950 contracts closing 2025-12-15 →
  2026-11-10**, 262 more than the earlier studies had, so the held-out set (the last 30%, 9,285
  contracts closing from 2026-08-04) is two days later than theirs, not the identical contracts.
- **Rules,** each with its mirror, one trade per contract per cell:
  - `residual_fade` / `residual_follow`: the contract's mid moved at least 3¢ more (or less) over 3
    minutes than `kalshi.settlement_probability` did on Coinbase's move. Entry 10 or 5 minutes
    before close; held to settlement, or sold 3 minutes later (an unsaleable exit is held and
    counted).
  - `streak_follow` / `streak_fade`: the two previous contracts settled the same way. Entry 13 or 10.
  - `vol_regime` / `vol_regime_inverse`: 15-minute realized vol at least 1.5× (or at most 1/1.5×)
    the 120-minute one, mid 10–40¢ from 50¢; buy the underdog after a spike and the favorite in a
    calm, or the reverse. Entry 10 or 5.
  - `basis_fade` / `basis_follow`: the strike sat at least 1.5 bp from its trailing 24-hour median
    distance to Coinbase at open. Entry 13 or 10.
- **Family:** 20 cells × 2 order sizes = **40 tests**, Bonferroni z = 3.23. Selected on train only
  if the interval lies entirely above zero.

### Result: every train cell loses; none selected

Each rule's best cell on train (all are 100-contract orders, held to settlement), with the same
cell held out. The held-out column is descriptive only.

| Rule | Cell | Train trades | Win rate vs break-even (train) | Train mean [family interval] | Held-out mean [interval] |
|---|---|---|---|---|---|
| `residual_fade` | 10 min | 9,478 | 44.4% vs 45.3% | −0.9¢ [−2.4, +0.5] | −0.2¢ [−2.3, +2.0] |
| `residual_follow` | 5 min | 10,622 | 60.8% vs 63.2% | −2.4¢ [−3.5, −1.4] | −1.0¢ [−2.7, +0.8] |
| `streak_follow` | 10 min | 10,202 | 46.5% vs 50.8% | −4.3¢ [−5.6, −2.9] | −2.4¢ [−4.6, −0.3] |
| `streak_fade` | 10 min | 10,202 | 53.5% vs 53.6% | −0.1¢ [−1.4, +1.3] | −1.3¢ [−3.4, +0.9] |
| `vol_regime` | 10 min | 3,336 | 60.9% vs 61.9% | −1.0¢ [−3.3, +1.4] | −1.3¢ [−5.0, +2.4] |
| `vol_regime_inverse` | 5 min | 2,328 | 31.4% vs 34.7% | −3.3¢ [−5.9, −0.6] | −0.3¢ [−3.9, +3.3] |
| `basis_fade` | 10 min | 3,666 | 54.1% vs 56.1% | −1.9¢ [−4.2, +0.3] | −5.4¢ [−9.5, −1.2] |
| `basis_follow` | 10 min | 3,666 | 45.9% vs 48.2% | −2.4¢ [−4.6, −0.1] | +1.9¢ [−2.3, +6.0] |

- **All 40 train means are negative** (−0.1¢ to −6.0¢). In every settlement cell the win rate is
  below the break-even rate.
- **The residual sale 3 minutes later is the worst exit:** −2.3¢ to −6.0¢ on train, every interval
  below zero. It pays the spread and the fee twice, as the scalp study found.
- **Sample sizes are not the problem.** Every cell has n_eff above 1,000; none is flagged small.
  Train MDEs are 0.7–2.7¢, so an edge of about 3¢ a contract in any of these cells would have shown.
  Smaller edges are not ruled out.
- **Held-out, descriptive:** 36 of 40 cells are negative. The four positive ones are all
  `basis_follow` (+0.3¢ to +1.9¢), every interval spans zero, and the same cells lose 2.4–3.1¢ on
  train. A sign that flips between the two sets is noise, not a lead.

### Direction information: real on train in two pairs, smaller than the cost, and unstable

Half the difference between a rule and its mirror estimates what the signal's direction is worth
per trade before costs (100-lot; descriptive, not a counted test):

| Pair | Train | Held out |
|---|---|---|
| residual: fade over follow, 10 min, settlement | +1.3¢ | +1.6¢ |
| residual: fade over follow, 5 min, settlement | +0.7¢ | −0.4¢ |
| streak: fade over follow, 13 / 10 min | +2.2¢ / +2.1¢ | +0.6¢ / +0.6¢ |
| vol regime over its inverse, 10 / 5 min | +1.2¢ / +1.1¢ | +0.5¢ / −1.4¢ |
| basis: fade over follow, 13 / 10 min | +0.3¢ / +0.2¢ | −2.7¢ / −3.6¢ |

The largest is the streak: after two same-way settlements the contract was priced about 2¢ too far
toward a third on train. That is the size of one spread plus fee, which is why `streak_fade` lands
at −0.1¢ rather than above zero, and it shrank to 0.6¢ held out. Nothing here is tradeable by a
taker.

### What this does and does not rule out

- **Rules out:** these eight rules at these thresholds and entry times, as taker trades at the
  quoted top of book. Every simplification (zero latency, full fill at the touch, one-minute
  decisions) flatters the result.
- **Does not rule out:** other thresholds for the same signals, edges under the MDEs above, maker
  (resting) orders, anything finer than one-minute candles, other venues' prices (Binance.US and
  Bybit were unreachable where this was run), and the live order-book signals, which have no
  history.
- **Limits of "blind":** the designer had read this file, held-out tables for the earlier studies
  included, before fixing the design. No number for any of these rules was in it.
- **Consequence for the bot:** none. Paper trading stays as it is, and no rule here should be
  offered as an automatic mode.

```bash
python review/kalshi_strategy_backtest.py                 # on an existing study_data/ cache
python review/kalshi_strategy_backtest.py --fetch         # builds the cache first (resumable)
```


## Forecasting study: can P(YES) be forecast better than the market? (2026-11-10)

**Question:** is any fitted probability for a KXBTC15M contract better than Kalshi's mid, than
`kalshi.settlement_probability` as the dashboard computes it, or than the base rate? This is a
forecasting study, not a trading rule: no fee, fill or profit is in any number below.
**Answer: no model beats the market mid. Five are significantly worse, one (the fitted combiner) is
slightly but significantly worse, and recalibrating the mid changes nothing.**

Code: `kalshi_forecast_study.py` (tested: `tests/test_kalshi_forecast_study.py`,
`tests/test_kalshi_forecast_backtest.py`). Run: `python review/kalshi_forecast_backtest.py`. The
design is pinned in the module docstring and was committed (58a9f65) before any model had been
fitted to a real outcome. No amendment was made. The held-out set was scored once.

### Design, fixed before any result was seen

- **Split, chronological.** Held-out: the existing held-out set, 9,285 contracts closing
  2026-08-04 19:00 → 2026-11-10. The cache holds nothing later, so there was no fresher set to
  prefer. **It is not pristine:** earlier studies had scored the mid and the dashboard's model on
  it, and the designer had read those results. Train is everything before; its last 25% (5,417
  contracts from 2026-06-08) is the validation slice used for all tuning.
- **Rows.** One per contract per horizon (10, 5, 2 minutes before close), scored only with a real
  two-sided book, a plausible strike and every feature present: **27,326 held-out rows**, the same
  rows for every forecast. At 2 minutes 529 held-out contracts have no two-sided book and are in
  no score.
- **Baselines:** `market` (the mid), `asis` (`settlement_probability`, last Coinbase trade,
  120-minute realized vol), `base_rate` (49.7%, from train).
- **Family: 7 models,** all fitted on train only, one fit per horizon:
  `platt_asis` and `iso_asis` (Platt / isotonic recalibration of `asis`); `structural` (vol
  estimator × spot × tail chosen on validation from 36 candidates, then Platt); `features`
  (structural plus 5-minute momentum, vol ratio, basis deviation, streak and the bot's net votes,
  L2 logistic); `bot_only` (the bot's net votes alone); `platt_market` (Platt on the mid);
  `combiner` (L2 logistic on the mid, structural, `asis` and the same features). The combiner is
  the only place market, model and bot meet, and it is a fitted model compared with each component.
- **Statistics.** Skill = 1 − loss(model) / loss(reference) for Brier and log loss, pooled over
  horizons, with a day-block bootstrap interval. **54 intervals share one Bonferroni z = 3.31.**
  A model is better or worse only if its interval excludes zero.
- **Tuning (validation only).** Chosen structural model: EWMA vol with a 30-minute half-life,
  basis-adjusted spot, Student-t tail with 3 degrees of freedom (validation log loss 0.3767; the
  worst candidate 0.3829). L2 = 0.1 for both regularized models.

### Result: nothing beats the mid

| Forecast | Brier | Log loss | Brier skill vs market [interval] | Calibration a / b | vs market |
|---|---|---|---|---|---|
| **market** (baseline) | 0.13346 | 0.40752 | – | −0.037 / 0.999 | – |
| asis (baseline) | 0.13798 | 0.42614 | – | +0.104 / 0.957 | – |
| base_rate (baseline) | 0.24999 | 0.69313 | – | constant | – |
| `platt_market` | 0.13356 | 0.40803 | −0.08% [−0.20, +0.04] | −0.029 / 0.938 | not distinguishable |
| `combiner` | 0.13439 | 0.41113 | −0.70% [−1.16, −0.24] | −0.025 / 0.901 | **worse** |
| `features` | 0.13766 | 0.42366 | −3.15% [−4.45, −1.86] | −0.012 / 0.840 | **worse** |
| `structural` | 0.13777 | 0.42307 | −3.23% [−4.69, −1.78] | −0.012 / 0.853 | **worse** |
| `platt_asis` | 0.13786 | 0.42892 | −3.30% [−4.53, −2.07] | −0.024 / 0.811 | **worse** |
| `iso_asis` | 0.13854 | 0.42785 | −3.81% [−5.10, −2.52] | −0.029 / 0.820 | **worse** |
| `bot_only` | 0.19234 | 0.56726 | −44.1% [−49.4, −38.8] | +0.008 / 0.963 | **worse** |

Log-loss verdicts against the market are the same in every row.

- **The mid is already calibrated on the held-out months:** slope 0.999 ± 0.016, and every one of
  its ten reliability buckets contains its own mean forecast. There is nothing for Platt scaling to
  fix, and `platt_market` (fitted on train, where the slope was ~1.06) is marginally worse.
- **No model improves on the dashboard's calculation either.** `platt_asis`, `iso_asis`,
  `structural` and `features` are each not distinguishable from `asis` (skill between −0.65% and
  +0.72%, every interval spanning zero). The tuned vol, the fat tail, the basis adjustment and the
  extra features together buy +0.15% Brier skill over `asis`, inside ±1.2%.
- **Train-fitted recalibration over-sharpens.** Every recalibrated model has a held-out slope of
  0.81–0.90: on train, prices and the model were too close to 50%, the fits pushed them out, and
  the held-out months did not need it. This is the same train-to-held-out shift the edge study
  recorded in the favorite-longshot slope.
- **The combiner is better than every non-market component** (+2.4% over `structural` and
  `features`, +30% over `bot_only`, all intervals above zero) **and worse than the market
  component it contains** (−0.62% vs `platt_market`, −0.70% vs the mid). Adding the model and the
  bot to the price subtracts information.
- **The bot's votes carry real but small information.** `bot_only` beats the base rate by 23%
  (Brier), because its trend votes correlate with where spot sits against a strike set minutes
  earlier; it is 33–44% worse than either price-based baseline, and `features` is not
  distinguishable from `structural` without it.
- **Everything beats the base rate** by 23–47%; that is the low bar, not a finding.
- **Sample sizes.** Every skill interval has n_eff above 9,000 and no reliability bucket is flagged
  small (smallest n_eff 1,093). Intervals against the market are ±0.1–1.5%, so a model 2% better
  than the mid would have shown.

Per horizon (descriptive, no intervals): the mid has the lowest Brier at 5 and 2 minutes; at 10
minutes `platt_market` and `combiner` are 0.0001 lower (0.19021 vs 0.19032), far inside noise.

### What this does and does not say

- **Says:** with Coinbase one-minute data, the bot's indicators and these model classes, the best
  available P(YES) is Kalshi's mid, unmodified. The dashboard should keep showing its own P(YES)
  beside the mid, not in place of it, and no blend of the two is justified (principle 1 now has
  evidence behind it: the fitted blend is worse than the price alone).
- **Does not say:** that the mid cannot be beaten with per-second or multi-venue prices, Kalshi's
  own index feed, order-book inputs, or other model classes; none were tested.
- **Calibration is not the result.** `bot_only` has the best calibration slope of any fitted model here (0.963)
  and the worst score of any model. Calibration without sharpness is the base rate.
- **Not a trading result.** No cost is in these numbers. The edge study already found that taker
  costs erase differences far larger than any above.

```bash
python review/kalshi_forecast_backtest.py     # on the study_data/ cache; saves per-row probabilities next to it
```


## Trade-flow study: does Binance.US flow improve the forecast? (2026-11-10)

**Question:** does adding Binance.US signed trade flow to a P(YES) forecast make it better than the
same forecast without flow, and at least not worse than Kalshi's mid? A forecasting study: no fee,
fill or profit is in any number below. **Answer: no. Flow has no detectable effect on any of the
four models it was added to, and no model beats the mid.**

Code: `kalshi_flow_study.py` (tested: `tests/test_kalshi_flow_study.py`,
`tests/test_kalshi_flow_backtest.py`). Run: `python review/kalshi_flow_backtest.py --fetch`, then
without the flag. The design is pinned in the module docstring and was committed (03a5399) before
any Binance.US trade had been fetched or any model fitted. No amendment was made. The held-out set
was scored once.

### Design, fixed before any trade was fetched

- **Trades.** Binance.US `BTCUSDT` aggTrades for the whole span, paged by trade id into one file
  per UTC day: **1,172,250 trades over 331 days (about 3,500 a day; taker-buy share 53.5%)**. The
  id sequence has no break, no day is missing and no id is duplicated, so no window had a data hole.
- **Held-out set: the existing one, not pristine.** Only **9** contracts had settled after the
  cache's last close when checked; the design required 1,000 for a pristine set. So the held-out
  set is the 9,285 contracts closing 2026-08-04 → 2026-11-10 that two earlier families were already
  scored on.
- **Rows.** The forecasting study's usable rows, less those with fewer than 10 trades in the
  30-minute window (`thin`: excluded and counted, never imputed): **26,541 held-out rows scored,
  785 excluded as thin** (2.9%); in train, 3,667 of 59,758 usable rows were thin (6.1%). The scores
  below are therefore on a subset and are not comparable with the forecasting study's.
- **Flow features (7),** from trades with timestamp ≤ the quote minute, taker side: notional
  imbalance over 5 and 30 minutes, trade-count imbalance, large-trade ($10,000+) imbalance, a
  short-window surge normalised by long-window activity, a flow-versus-return divergence normalised
  by volatility, and a flag for an empty 5-minute window (3.8–4.8% of scored held-out rows).
- **Family: 8 fitted models,** four with flow, each paired with the same model without its flow
  columns: on the mid (`market_flow` / `market_base`), on the mid with volatility and US-hours
  regime interactions (`market_flow_regime` / `market_regime_base`), on the dashboard's model
  (`model_flow` / `model_base`), and on both (`both_flow` / `both_base`). L2 logistic, one fit per
  horizon, tuned on a validation slice inside train.
- **Statistics.** 60 skill intervals (day-block bootstrap) share one Bonferroni z = 3.34. Flow
  counts as helping only if both the Brier and log-loss intervals against the flow-free counterpart
  lie above zero and the model is not worse than the market.

### Result: no detectable effect

| Model | Brier | Log loss | Brier skill vs market [interval] | Brier skill vs flow-free counterpart [interval] | Calibration a / b |
|---|---|---|---|---|---|
| **market** (baseline) | 0.13334 | 0.40717 | – | – | −0.038 / 0.998 |
| asis (baseline) | 0.13771 | 0.42544 | – | – | +0.100 / 0.956 |
| base_rate (baseline) | 0.24999 | 0.69313 | – | – | constant |
| `market_base` | 0.13344 | 0.40769 | −0.08% [−0.20, +0.04] | (flow-free) | −0.029 / 0.938 |
| `market_flow` | 0.13348 | 0.40753 | −0.10% [−0.31, +0.10] | −0.02% [−0.21, +0.16] | −0.040 / 0.957 |
| `market_regime_base` | 0.13345 | 0.40768 | −0.09% [−0.22, +0.04] | (flow-free) | −0.024 / 0.940 |
| `market_flow_regime` | 0.13352 | 0.40767 | −0.14% [−0.37, +0.09] | −0.05% [−0.25, +0.14] | −0.035 / 0.955 |
| `model_base` | 0.13752 | 0.42815 | −3.14% [−4.37, −1.91] | (flow-free) | −0.025 / 0.812 |
| `model_flow` | 0.13774 | 0.42864 | −3.30% [−4.51, −2.09] | −0.16% [−0.45, +0.14] | −0.039 / 0.809 |
| `both_base` | 0.13370 | 0.40908 | −0.27% [−0.49, −0.05] | (flow-free) | −0.027 / 0.915 |
| `both_flow` | 0.13388 | 0.40955 | −0.40% [−0.70, −0.11] | −0.13% [−0.36, +0.10] | −0.039 / 0.912 |

- **The effect of flow is zero to within ±0.2–0.4%.** All eight counterpart intervals (four models,
  Brier and log loss) span zero; the point estimates run from −0.16% to +0.04%, six of eight on
  the wrong side. A flow effect of half a percent of the loss would have shown.
- **It already failed on validation.** Every flow model had a higher validation log loss than its
  counterpart (e.g. 0.37063 vs 0.37001 on the mid), before the held-out set was touched.
- **No model beats the mid.** The four models built on the mid alone are not distinguishable from
  it; `model_*` and `both_*` are worse. `both_flow` is worse than its `market_flow` component
  (−0.30% [−0.52, −0.08]): as in the forecasting study, adding the dashboard's model to the price
  subtracts.
- **The fitted flow coefficients are small and do not point one way.** On the mid, at 10 minutes,
  30-minute imbalance is −0.09 and divergence +0.11 logit per standard deviation, largely
  offsetting; the rest are within ±0.05, with signs that change across horizons. Descriptive only.
- **Sample sizes.** Every skill interval has n_eff above 10,000 and no reliability bucket is
  flagged small (smallest n_eff 1,310).

### What this does and does not say

- **Says:** Binance.US taker flow over the previous 5 and 30 minutes carries no information about
  KXBTC15M settlement that the Kalshi mid, or the dashboard's own model, does not already have.
- **Does not say** that trade flow is useless. This is one thin venue (about 2.5 trades a minute)
  quoting USDT, not the three-venue flow the live `agg_delta` pools: Bybit was geo-blocked where
  this ran and Coinbase has no time-ranged trade history. Deeper venues, order-book inputs (no
  history exists), sub-minute sampling (Kalshi quotes here are one-minute closes), other windows
  and non-linear models were not tested.
- **Not pristine.** The held-out contracts had been scored by two earlier families and the designer
  had read those results. For a null this matters less than for a positive claim, but it is stated.
- **Consequence for the bot:** none. `agg_delta` stays a vote in the composite; nothing here
  supports feeding it into the Kalshi P(YES).

```bash
python review/kalshi_flow_backtest.py --fetch   # counts new contracts, caches Binance.US trades (resumable, minutes)
python review/kalshi_flow_backtest.py           # fits on train, scores the held-out set
```
