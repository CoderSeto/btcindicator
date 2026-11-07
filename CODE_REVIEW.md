# Full code review: instructions for Claude Opus

You are doing a complete, evidence-based review of this repository, a BTC
(and other pairs) technical-indicator bot with a Streamlit dashboard, a
console mode, multi-exchange trade and level-2 feeds, a backtester, and a
Kalshi 15-minute binary fair-value calculator. Read `CLAUDE.md` first. It is
the source of truth for architecture, design principles and past bugs. This
file only adds how to run the review.

The bar: **every finding must be demonstrated, not asserted.** A finding
without a reproduction (a failing test, a script's output, or a quoted line
plus a worked example) is a hypothesis. Label it as one.

---

## 0. Setup

```bash
pip install -r requirements.txt pytest scipy   # scipy is test-only (cross-checks)
python -m pytest -q                            # baseline: must be green before you start
```

Record the baseline test count. Do not start from a red suite without saying so.

Constraints that apply to everything you do:

- **No API keys, ever.** Every integration uses public, unauthenticated
  endpoints. Adding auth is a different, bigger change. Flag it, don't slip it in.
- **Never blend independently sourced probabilities** (design principle 1).
  A "fix" that merges the Kalshi P(YES), the backtest hit rate, the weighted
  conviction or the order-book readout into one number is wrong by this
  project's rules.
- Network access in a sandbox may be restricted. Say what you could and could
  not reach, and don't claim live behaviour you didn't observe.
- Work on the branch you're given. Do not open a PR unless asked.

---

## 1. Method

1. **Map first.** Skim every module and note its inputs, outputs, shared
   state and threads. Build a short data-flow picture in your head before
   judging any single file.
2. **Review by risk, not alphabetically** (order in section 3).
3. **For every suspected bug, write the failing test first.** Put it in
   `tests/`, matching the existing style (see section 5). Confirm it fails for
   the reason you think. Only then fix, and confirm it passes.
4. **Distinguish three things in your report:** confirmed bug (failing test),
   design concern (judgment, no failing test possible), and nit.
5. **Check the tests too.** A test that can't fail (`or True`, asserting what
   it just set up, mocking the thing under test) is a finding. For any
   important test, ask what bug would make it fail. If none, say so.
6. **Don't rediscover known items** (section 4). Verify they're still accurate,
   then spend your time elsewhere.

---

## 2. What good looks like in this repo

Review against the project's own principles (details in `CLAUDE.md`):

- **Absent, not neutral.** A signal with no data is left out of
  `snapshot.individual`, never recorded as a neutral vote.
- **Fixed CSV schemas, blank not zero.** Any logged row whose fields can vary
  uses a declared header, and a changed layout rotates the old file aside.
- **Time is evidence.** Prefer "held for N seconds" to a single sampled frame.
- **State limitations in the output, not just docstrings.**
- **Small samples are flagged, not hidden.**
- **Depth in basis points and notional**, never level counts. **Venue books are
  never pooled**, only per-venue ratios are averaged.

A change or an existing behaviour that violates one of these is a finding even
if nothing crashes.

---

## 3. Review areas (in priority order)

### 3.1 Correctness of the numbers people may act on
- `kalshi.py`: the lognormal crossing probability, annualization
  (365.25 x 24 x 60), the no-drift and point-in-time-settlement assumptions,
  and zero or negative time remaining. Check units end to end:
  minutes -> years, 1-minute returns -> annual vol.
- `kalshi_api.py` + `kalshi.py` together: does the model price the **same
  event the contract settles on?** Kalshi's KXBTC15M rules (see
  `tests/fixtures/kalshi_open_markets.json`, field `rules_primary`) compare the
  average of the 60 seconds before close against the average of the 60
  seconds before open. `floor_strike` is a reference value. Decide whether
  treating it as a fixed barrier is sound, and state the size of any error.
- `kalshi_tracking.py`: calibration bucketing, Brier score, one row per
  ticker (closest to close), outcome resolution and caching.
- `weighting.py`: weights come from the same window they're applied to
  (acknowledged in the docstring). Check whether `hit_rate - 0.5` is the right
  edge when the base rate of up-moves isn't 50%, and how ties are handled.
- `backtest.py` / `backtest_agg_delta.py`: look-ahead leakage, off-by-one on
  the lookahead index, overlapping outcome windows (samples are not
  independent, so the stated `n` overstates the evidence), and survivorship in
  the fetched history. State quantitatively how much the overlap matters.

### 3.2 Concurrency and the live feeds
- `exchange_feeds.py`, `orderbook_feeds.py`: lock discipline in
  `TradeAggregator` and `BookAggregator`, callbacks invoked outside the lock,
  sampler-thread death and stale-read handling, `run_with_reconnect` backoff
  and stop-event behaviour, thread and socket leaks on stop.
- Binance diff-depth procedure (gotcha #8): the `U`/`u` checks, buffering
  during the snapshot fetch, resync on a gap. Walk the code against Binance's
  documented procedure, not against the tests (the runner threads are
  **untested**).
- Liquidation payload normalization (gotcha #3) and Coinbase `level2_batch`
  (gotcha #4).
- Streamlit: `@st.cache_resource` feed handles are never evicted. Check what
  happens when the pair changes repeatedly, and whether anything leaks.

### 3.3 Data integrity
- CSV logging in `main.py`, `csv_utils.py`, `kalshi_tracking.py`: fixed
  headers, rotation, and **concurrent writers** (a Streamlit rerun plus a
  console run, or two browser sessions appending to the same file).
- pandas 3.x datetime resolution (gotcha #2). `grep` for every
  `astype("int64")`, `/ 1e9` and `unit="s"` and check each one.
- Float handling in sliding windows and accumulations (a residue bug was
  already found in `reconstruct_agg_delta_series`; look for siblings, e.g. in
  `TradeAggregator`).

### 3.4 Signals and indicators
- `indicators.py` conventions are pinned by tests (RSI is Cutler's, not
  Wilder's; all-gains RSI reports 50; volume average includes the current
  candle). Decide whether each is a **defensible choice** or a **latent
  bug**, and say which.
- `signals.py`: crossover logic at the boundaries (`>` vs `>=` on `prev`),
  persistence gating, and the fallback to the legacy REST book.
- `divergence.py`: swing detection edge cases and window handling.

### 3.5 Dashboard (`app.py`, `dashboard_data.py`, `charts.py`)
- Rerun cost: how many network calls does one refresh make? Any chance of
  hitting rate limits? (Kalshi returned HTTP 429 during one session.)
- `st.session_state` logic for the Kalshi block: strike persistence,
  window-change reset, auto-track off.
- Every string containing `$` reaching `st.markdown`/`st.caption`/
  `st.warning` must go through `_esc()`. Grep for misses.
- Deprecated Streamlit APIs (`use_container_width`) and the
  `streamlit>=1.35` floor.

### 3.6 Security and robustness (it is a read-only tool, so keep this proportionate)
- Anything interpolated into a URL or file path (tickers, product IDs, config).
- Behaviour on malformed, empty or hostile API payloads, timeouts, and HTTP
  errors. Every `requests` call should have a timeout.
- Dependencies: unpinned versions, anything unmaintained or unnecessary.

### 3.7 The test suite itself
- Run it three ways: normally, with `-p no:randomly` (if installed) or in
  reverse order, and twice in a row. Order-dependence, shared global state
  (the `config` module is mutated by `app.py`), and flakiness are findings.
- Coverage gaps worth stating: the WebSocket runner threads, reconnect logic,
  and visual layout. Say which gaps matter most.

---

## 4. Known items: verify, don't rediscover

These were noticed during test-writing. Confirm each is still true, assess its
real impact, and recommend (don't silently decide).

| # | Item | Status |
|---|------|--------|
| 1 | `kalshi_api.BASE_URL` is `external-api.kalshi.com`; the payload fixtures came from `api.elections.kalshi.com`. The former was blocked by the sandbox proxy, so it is unknown whether it works. | Unresolved. Needs a live check outside the sandbox. |
| 2 | `signals.py` hard-codes `sma_9`, `sma_21`, `ema_12`, `ema_26`, while `indicators.py` names columns from config. Changing those config values raises `KeyError`. | Pinned by a test, not fixed. |
| 3 | RSI uses a simple rolling mean, and a window with gains and no losses reports 50, not 100. | Pinned as current behaviour. Decide if intended. |
| 4 | Volume-spike average includes the current candle. | Pinned as current behaviour. |
| 5 | `app.py` uses `use_container_width` (9 places), deprecated in Streamlit. | Unfixed. |
| 6 | Suspected, **unverified**: `parse_market` would raise `TypeError` for a `close_time` with no timezone (naive minus aware). Real Kalshi data always has `Z`. | Verify, then judge severity. |
| 7 | Backtest samples overlap (lookahead of N candles, evaluated every candle), so `n` overstates independent evidence. | Not quantified. |

Bugs already found and fixed (watch for regressions and siblings): CSV header
drift (gotcha #1); pandas datetime resolution (gotcha #2);
`compute_calibration` mixing quotes across rows via `groupby().first()`;
`compute_calibration` crashing when only some contracts had quotes; float
residue in `reconstruct_agg_delta_series`; unescaped `$` in divergence text;
`kalshi_predictions.csv` not rotating on a layout change.

---

## 5. Test style (match it)

Follow `CLAUDE.md` > Testing approach. In short:

- Math is cross-checked against an **independent** reference (scipy, numpy,
  `statistics`, a brute force), never against a re-implementation of the same
  formula.
- External schemas are tested against **verbatim real payloads** with their
  source named in the docstring (`tests/fixtures/` holds two Kalshi captures).
- CSV writers get the "field appears or disappears mid-session" case and the
  "layout changed between releases" case.
- Derived/statistical code gets hand-built data with an exact expected answer,
  and the arithmetic in a comment.
- Prefer testing a property (antisymmetry, monotonicity, causality) where the
  property matters more than a value.
- The dashboard is tested with Streamlit's `AppTest` (`tests/test_app.py`).

---

## 6. Deliverable

1. **`REVIEW_FINDINGS.md`** in the repo root, with:
   - A one-paragraph verdict: is this safe to rely on, and for what?
   - A table of findings: `ID | Severity | Type | Location (file:line) | Summary | Evidence`.
     - Severity: **Critical** (wrong number someone may act on, data loss),
       **High**, **Medium**, **Low**, **Nit**.
     - Type: *Confirmed bug* / *Design concern* / *Test gap* / *Hypothesis*.
     - Evidence: the test name or command that demonstrates it.
   - For each Critical/High: the failing test, the fix, and the passing result.
   - A short "what I could not verify" section (no live network, untested
     threads, etc.).
   - The items from section 4, each with your verdict.
2. **Fixes** only for confirmed bugs, one commit per fix, each with its
   regression test. Leave design concerns as recommendations. Don't refactor
   opportunistically.
3. Update `CLAUDE.md` if you add a gotcha that already caused (or would
   cause) a real bug.
4. Final message: the verdict, counts by severity, the test count before and
   after, and the 3 things you'd do next.

**Before you finish:** the full suite is green, nothing is left untracked,
and every claim in `REVIEW_FINDINGS.md` points at evidence a reader can
re-run.
