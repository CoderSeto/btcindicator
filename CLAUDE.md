# CLAUDE.md

Guidance for Claude Code (or any future contributor) working in this repo.
This is the developer-facing companion to README.md, which is user-facing
(how to run the bot). This file is about how the code fits together, why
certain things are the way they are, and what's already gone wrong once.

## What this is

A BTC (and other pairs) technical-indicator bot with a Streamlit dashboard
and a console mode. It computes 7-9 signals from live Coinbase data plus
multi-exchange trade flow and multi-exchange level-2 depth, backtests them against real historical data,
weights them by empirical accuracy, and includes a standalone fair-value
probability calculator for Kalshi's 15-minute BTC binary contracts. No
API keys anywhere -- every integration (Coinbase, Binance.US, Bybit,
Kalshi) uses public, unauthenticated market-data endpoints. Keep it that
way unless a feature genuinely requires auth; that's a different, bigger
kind of change (credential storage, auth flow) and should be flagged as
such, not slipped in quietly.

## Architecture (read in roughly this order to understand the codebase)

**Data layer**
- `config.py` -- every tunable setting lives here. When in doubt about a
  magic number, check here first; almost nothing is hardcoded elsewhere.
- `data_fetch.py` -- Coinbase REST: latest candles, order book, and a
  time-ranged historical candle fetch used by the backtest. Also
  `drop_forming_candle()`: a candle is stamped with its bucket's start, so
  the last one may still be forming. Signals and the backtest use completed
  candles only. The displayed price and the Kalshi model's spot come from
  `/ticker` via `latest_price()` (`DashboardState.live_price`), never from a
  candle: see gotcha #19. `price_df` (raw candles) is only the vol input.
- `exchange_feeds.py` -- live WebSocket feeds (Coinbase, Binance.US,
  Bybit trades + liquidations) feeding into `TradeAggregator`, a
  thread-safe rolling buy/sell aggregator. Also has the Binance.US REST
  historical-trades client used by `backtest_agg_delta.py`, and
  `run_with_reconnect()`, the shared reconnect/backoff loop that
  `orderbook_feeds.py` reuses.
- `orderbook_feeds.py` -- live level-2 depth WebSockets (Coinbase
  `level2_batch`, Binance.US diff-depth + REST snapshot, Bybit
  `orderbook.200`) maintained as `LocalBook`s inside a `BookAggregator`,
  which samples the aggregated metrics once a second into a rolling
  history. That history is the whole point: persistence, resting-liquidity
  flow and wall age are all things a single snapshot cannot answer.

**Signal computation**
- `indicators.py` -- SMA/EMA/RSI/MACD/Bollinger/volume-spike, computed
  once per refresh on the full candle DataFrame (vectorized, not
  per-row).
- `orderbook.py` -- all order-book math, as pure functions over a
  normalized book dict. Depth is measured in **basis points from mid** and
  in **notional**, never in level counts or base volume -- see gotcha #6.
  Also microprice, spread, wall detection, and `aggregate_metrics()`,
  which combines per-venue reads (see gotcha #7 for why it averages ratios
  rather than merging books).
- `signals.py` -- two entry points, both important:
  - `evaluate()` -- the live path, adding `orderbook`, `book_flow` and
    `agg_delta` to the 6 candle signals (7-9 total, depending on which
    feeds have warmed up). The `orderbook` vote is persistence-gated: a
    lean only counts once it has held past `BOOK_IMBALANCE_THRESHOLD` for
    `BOOK_PERSISTENCE_SECONDS`. Falls back to the legacy single-snapshot
    Coinbase REST reading when the depth feeds aren't up.
  - `evaluate_price_only()` -- the backtestable path, only the 6
    candle-derived signals. The two order-book signals and agg_delta have
    no historical data source (no exchange publishes historical level-2
    snapshots; no exchange has a practical historical multi-exchange trade
    feed), so they're deliberately excluded here. `backtest_agg_delta.py`
    is the one exception -- see below.
- `divergence.py` -- RSI/price divergence (reversal) detection.

**Backtesting and weighting**
- `backtest.py` -- pulls historical candles, replays `evaluate_price_only()`
  at every point, checks whether price actually moved as predicted within
  `BACKTEST_LOOKAHEAD_CANDLES`. Produces per-indicator AND per-composite
  hit-rate buckets, saved to `backtest_results.json`. Every bucket also
  carries `n_eff` (`effective_sample_size()`: neighbouring rows share their
  outcome window, so raw n overstates the evidence ~3.5x for state votes)
  and `base_rate` (how often any scored window moved that way) -- see
  gotcha #15.
- `backtest_agg_delta.py` -- separately backtests `agg_delta`, using
  Binance.US's historical trade API (the only one of the three live
  exchanges with a practical time-range historical-trades endpoint --
  Coinbase's `/trades` is cursor-paginated by trade ID, not time; Bybit's
  caps at the last 60 trades). Replays the exact same rolling-window math
  as the live `TradeAggregator` against historical trades, then merges
  its hit-rate buckets into the same `backtest_results.json`. This
  reflects Binance.US alone, not the combined 3-exchange live signal --
  an approximation, not a reconstruction.
- `weighting.py` -- turns per-indicator hit rates into weights
  (`weight = 1.0 + SCALE * (hit_rate - base_rate)`, clipped; stays 1.0 for
  no data, `n_eff < BACKTEST_MIN_SAMPLE`, or an edge within
  `INDICATOR_WEIGHT_MIN_Z` standard errors -- `weight_details()` says which),
  and computes a `weighted_conviction` score.
  **This is a separate, additive score. It never modifies
  `snapshot.composite`/`strength`, which still drives alerts and CSV
  logging.** See "Design principles" below for why.
- `outlook.py` -- turns snapshot + divergence + backtest confidence +
  weighted conviction into the plain-language narrative shown in the UI.

**Interfaces**
- `main.py` -- console loop. Also owns `log_alert()` and
  `log_signal_snapshot()`, the `ALL_SIGNAL_NAMES` / `BOOK_LOG_FIELDS`
  fixed-schema constants both use. Every log -- these two and
  `kalshi_tracking.log_prediction` -- writes through
  `csv_utils.append_row()`, which rotates a log aside when its column
  layout changes between releases (gotcha #1) and does the whole append
  under a cross-process file lock (gotcha #16). Both loggers skip a row that repeats the file's last one
  within `LOG_MIN_INTERVAL_SECONDS` (a different alert is always written),
  because the dashboard reruns on every tick, click and tab.
- `dashboard_data.py` -- `load_state()`, the fetch+compute pipeline the
  dashboard calls once per refresh.
- `app.py` -- the Streamlit dashboard. Works on a per-run copy of
  config.py's settings (`cfg`), never the shared module -- see gotcha #14.
  Layout is deliberately
  front-loaded: Kalshi calculator and top metrics first, heavier
  sections (chart, live feeds, alert log) collapsed in expanders.
- `charts.py` -- Plotly figure builder.
- `analyze_correlation.py` -- reads `signal_log.csv` (opt-in via
  `LOG_ALL_SIGNALS`) to measure how much `agg_delta` overlaps with the
  other indicators empirically, rather than by reasoning about it.

**Kalshi**
- `kalshi.py` -- fair-value probability calculator for KXBTC15M-style
  binary contracts. `settlement_probability()` prices what actually
  settles -- the average of 60 one-second prints before close -- under
  driftless GBM; `crossing_probability()` is the textbook point-in-time
  N(d2), kept for comparison (`p_yes_point`). Pure math, no network calls
  except in its own CLI entry point.
- `kalshi_api.py` -- public (no-auth) Kalshi market-data client.
  `SERIES_TICKER = "KXBTC15M"` -- **not** `KXBTCD`, which is a different,
  separate hourly series (easy to conflate; cross-checked multiple
  sources before committing to this).
- `kalshi_tracking.py` -- logs fair-value predictions (opt-in via
  `LOG_KALSHI_PREDICTIONS`) and checks them against real settled
  outcomes for calibration (bucketed hit-rate table + Brier score,
  compared against Kalshi's own market price as a baseline).
- `paper_trading.py` -- paper trading on KXBTC15M (dashboard tab "Paper
  trading", or `python paper_trading.py`). An order buys YES at the ask or NO
  at 1 - bid and pays `kalshi_study.order_fee` (reused, so the fee model has
  one home), and settles on Kalshi's own recorded `result` via
  `kalshi_tracking.resolve_outcomes`. Never fill at the mid or skip the fee:
  the edge study's whole finding is that costs erase the apparent edge, so a
  frictionless paper account would be a flattering lie. The ledger is an
  append-only EVENT log (`open` / `settle` rows joined on `trade_id` by
  `load_trades()`), written only through `csv_utils.append_row`; settling uses
  its `skip_if` so two tabs can't settle one trade twice. Orders price on a
  fresh `current_btc15m_market()`, not the 10s poll cache. `summarize()`
  reports the win rate against the break-even rate with its standard error
  (counting each market once) and flags small samples -- keep it that way.
  Not modelled, and the tab says so: size at the touch, latency.
  Automatic mode (`auto_step()`, driven from the dashboard's refresh loop, so
  it only trades while a page is open) enters once per market inside
  `PAPER_AUTO_ENTRY_MINUTES`..`PAPER_AUTO_LAST_MINUTES` before close under one
  of two rules in `decide()`: "composite" (follow the composite when it is
  strong enough to alert) or "model" (calls `kalshi_study.model_trades`, the
  study's own rule, not a copy). Trades carry a `source` column
  ("manual" / "auto:<rule>") and `summarize_by_source()` keeps each rule's
  record apart -- never pool them, and never present either rule as having
  edge (principle 1; the study found none). Once-per-market is enforced by
  `place_order(once_per_market=True)` through `append_row`'s locked `skip_if`,
  so two open tabs place one order; the order is re-decided on a fresh quote.
- `kalshi_study.py` -- the KXBTC15M edge study (REVIEW_FINDINGS.md, "Is
  there an edge?"): fetches and caches every settled market and its
  1-minute candlesticks, plus Coinbase 1-minute history, into
  `study_data/` (gitignored); per-order fees; train/test split;
  day-block bootstrap; the model-vs-market comparison. Its docstring holds
  the pre-registered design. Change it only by adding a labelled
  amendment, never after looking at outcomes. Kalshi API facts it
  depends on: the batch `GET /markets/candlesticks` caps a request at
  10,000 candles (markets x minutes in the range) and returns nothing for
  markets older than ~2026-08-31, which come one at a time from
  `/historical/markets/{ticker}/candlesticks`. A candle is stamped with
  the END of its minute. Some market records have no `floor_strike`, and
  two have one scaled by 10^-4.
  `review/kalshi_market_study.py` runs it.
- `kalshi_scalp_study.py` -- the scalping backtest: buy a side, sell it back
  1-3 minutes later at the then-standing bid, on one of four fixed entry rules
  (contract momentum/reversion, BTC momentum/reversion). Same machinery and
  discipline as `kalshi_study.py` (reuses its fee, `valid_book`, split,
  day-block bootstrap and Bonferroni; fixed design in the docstring, changed
  only by a labelled amendment that is counted in the family, never after
  looking at outcomes). A scalp pays the spread and the taker fee twice, which
  is the whole reason to test before building; an exit with no sellable book is
  held to settlement and counted (`forced`), never dropped. `cost_floor()` gives
  the break-even move and an ORACLE ceiling (look-ahead, an upper bound on any
  direction rule, not a strategy). `review/kalshi_scalp_backtest.py` runs it on
  the `study_data/` cache.
- `edge_recorder.py` / `kalshi_forward_study.py` -- the forward edge hunt
  (`EDGE_HYPOTHESES.md`, REVIEW_FINDINGS.md "Forward edge hunt"). The recorder
  writes one fixed-schema row a second joining the live Kalshi YES top of book
  (public orderbook endpoint; YES ask = 1 - best NO bid), `latest_price()`,
  the depth and flow aggregators, liquidations, and `settlement_probability`
  with its raw inputs and a model-version tag; blanks and an `*_up` flag per
  source, a `gap` row when the loop stalls, Kalshi prints to a second log,
  everything via `csv_utils.append_row`. Standalone, key-free, not wired into
  the dashboard or alerts. The evaluator transcribes the pre-registered doc:
  five hold-to-settlement hypotheses, one entry per market, taker fills,
  per-order fee, day-block intervals, Bonferroni over 5, and refuses rows
  stamped before the pre-registration. Changing a constant in either the doc
  or the evaluator after data exists is a labelled amendment counted in the
  family, never a quiet edit. `kalshi_api.parse_orderbook` / `parse_trades`
  raise `KalshiSchemaError` on an unrecognised payload and are UNTRUSTED
  until `python edge_recorder.py --capture` has saved real payloads to
  `tests/fixtures/` (Kalshi was unreachable where they were written).
  `review/kalshi_forward_edge.py` runs the evaluation; its report says
  "INTERIM, NOT A VERDICT" until a hypothesis reaches its stopping rule.

## Design principles (the "why" behind non-obvious choices)

1. **Never blend independently-sourced probability/confidence numbers
   without validated evidence for the blending method.** This shows up
   four times now: weighted conviction is shown *separately* from the raw
   composite; the Kalshi model's P(YES) is shown *separately* from the
   bot's backtested confidence; the "resting liquidity between spot and
   the strike" readout sits *next to* an unchanged P(YES) rather than
   adjusting it; correlation analysis and hit-rate backtesting are kept as
   distinct tools. We have real evidence some
   signals have historical edge. We do not have validated evidence for
   the *correct way* to combine multiple such numbers into one. Don't
   invent a formula and present it with more confidence than it's
   earned. If you're tempted to blend two of this project's probability
   outputs into one, that's a sign to stop and think, not a quick win.

2. **State limitations in the actual output, not just a docstring.** The
   Kalshi calculator prints its basis-risk/settlement-averaging/no-drift
   caveats every time it runs, not once in a README someone might not
   read. Match this standard for new features that produce a number
   someone might act on.

3. **Small-sample results are flagged, not hidden.** `BACKTEST_MIN_SAMPLE`,
   `KALSHI_CALIBRATION_MIN_SAMPLE`, etc. -- every hit-rate/calibration
   table marks low-n buckets rather than presenting them at face value.

4. **Fixed CSV schemas, always.** Any logged row where a field might not
   always be present (running examples: `agg_delta` and `book_flow`, which
   only exist once their feeds have warmed up) must use a fixed,
   pre-declared column list with blank cells for missing values -- never
   derive the header from whatever happens to be in the first row. Note
   *blank*, not zero: for a numeric column like `book_imbalance`, zero is
   a real reading (a balanced book) and must stay distinguishable from
   "the feed was down". See gotcha #1.

5. **Absent, not neutral.** A signal with no data is left out of
   `snapshot.individual` entirely rather than recorded as a neutral vote.
   A neutral vote is a claim ("I looked, and it's balanced"); silence
   isn't. The loggers turn that absence into a blank cell, and the
   composite denominator shrinks accordingly.

6. **Time is evidence.** Where a reading can be sampled over time, prefer
   "this has held for N seconds" to "this is true in the frame I
   sampled". The order-book persistence gate and wall ageing both exist
   for this reason, and both cost nothing but a rolling deque.

## Known gotchas (already caused real bugs -- don't reintroduce)

1. **CSV header drift when a field appears mid-session.** `log_alert()`
   in `main.py` originally derived its CSV header from whichever
   indicators were present on the *first* alert. Once `agg_delta`
   started appearing later (after feeds warmed up), later rows had one
   more column than the header declared, and `pd.read_csv()` broke in
   production. Fixed by introducing `ALL_SIGNAL_NAMES`, a fixed list
   both `log_alert()` and `log_signal_snapshot()` use, writing `""` for
   any indicator not present in a given snapshot. Any new logging
   function needs the same treatment if its row shape can vary.

   Same corruption, second route: **adding a column in a new release**
   appends wider rows to a file already written under the narrower
   header. `csv_utils.needs_header()` handles that now -- if the on-disk header
   doesn't match the expected one, the old file is renamed to
   `<path>.<timestamp>.bak` and a fresh one is started, so both files
   stay internally consistent and readable. Never just append.

2. **pandas datetime resolution is not reliably nanoseconds.**
   `pd.to_datetime(x, unit="s")` does not consistently produce
   `datetime64[ns]` -- pandas 3.x can produce `datetime64[s]` from that
   call. Code that assumed nanoseconds and divided an `int64`-cast
   timestamp by `1e9` silently computed dates near the Unix epoch (1970)
   instead of raising an error, because dividing an already-in-seconds
   value by another billion just produces a tiny-but-valid number. This
   shipped in `backtest_agg_delta.py` and caused a real "zero trades,
   no error" failure. Fix: force resolution explicitly before an
   epoch conversion --
   `series.values.astype("datetime64[ns]").astype("int64") / 1e9` --
   rather than assuming any particular resolution. `backtest_agg_delta.py`
   also has a sanity check (`_MIN_PLAUSIBLE_MS`) that fails loudly if a
   computed date is implausible, specifically to catch this class of bug
   early next time.

3. **Binance's and Bybit's liquidation payloads use the same field
   letter (`S`) for opposite concepts.** Binance's `S` is the side of
   the *closing order* (`SELL` closes a long). Bybit's `S` is the
   *position side itself* (`Buy` means a long was liquidated). Both are
   normalized to `liquidated_side: "long"/"short"` inside
   `parse_binance_liquidation()`/`parse_bybit_liquidation()` in
   `exchange_feeds.py`. Any new code touching raw liquidation payloads
   should go through those functions, not re-derive this logic.

4. **Coinbase Exchange's `level2` WebSocket channel requires
   authentication** (since 2023-08-01, along with the Full and Level3
   channels). `level2_batch` is the unauthenticated equivalent -- same
   JSON schema, batched every 50ms -- and is what `orderbook_feeds.py`
   subscribes to. If depth ever silently stops arriving from Coinbase,
   check this first; subscribing to `level2` will look like a connection
   that works and then never sends a snapshot. Keeping this project
   key-free is the constraint that decides it.

5. **Binance.com blocks US IPs from its spot API/WebSocket.** This
   project uses Binance.US for spot trades specifically because of this
   (see `_binance_us_trade_runner` in `exchange_feeds.py`). Binance's
   futures liquidation stream (`fstream.binance.com`) is a different
   domain and isn't affected the same way -- that one still uses
   binance.com directly.

6. **Order-book depth measured in "levels" is not comparable across
   exchanges.** Coinbase quotes BTC-USD in $0.01 increments, so 25 levels
   can span well under a dollar; Bybit's spot book is much coarser. The
   original signal summed base volume over the top 25 levels, which meant
   it was measuring a different slice of the book on every venue and
   couldn't be aggregated honestly. Everything in `orderbook.py` is now in
   **basis points from mid** (venue-independent) and in **notional**
   (`price * size`, which is what actually has to be absorbed). If you add
   a metric here, keep both conventions.

7. **Never merge order books across venues by pooling price levels.**
   Coinbase quotes BTC-USD; Binance.US and Bybit quote BTCUSDT. USDT is
   not a dollar and the basis drifts by tens of basis points, so a pooled
   book has a fictional mid and a near band that mixes levels 2bps from
   mid on one venue with levels 30bps away on another. It produces
   plausible-looking numbers that mean nothing. `aggregate_metrics()`
   instead averages each venue's *dimensionless* imbalance ratio --
   measured against its own mid, so basis-invariant -- weighted by in-band
   notional. The same trap applies to any future cross-venue price
   arithmetic here.

8. **Binance's local-order-book procedure is not optional bookkeeping.**
   `<symbol>@depth@100ms` is a diff stream: it must be spliced onto a REST
   snapshot, and every event's `U`/`u` checked against the local update ID.
   `U > local + 1` means events were missed and the only correct response
   is to *discard the book and re-seed* -- patching over the hole leaves a
   book that looks fine and is quietly wrong, with no error anywhere. That
   logic lives in `BookAggregator.apply_sequenced_update()` /
   `seed_sequenced_book()` (methods rather than inline callback code
   specifically so it can be unit tested without a socket), and is covered
   in `tests/test_orderbook_feeds.py`.

9. **Windows dev environment needs `py`, not `python`/`pip` directly.**
   Not a code issue, but worth knowing if giving the user shell
   commands: their machine has multiple Python installs, and only the
   `py` launcher (`py`, `py -m pip`) reliably hits the one with this
   project's dependencies installed.

10. **A hole in a sampled history is not evidence.** `BookAggregator` only
    samples while some venue has a usable book, so an outage (or the
    machine asleep) leaves a gap. Persistence and wall age used to walk
    straight across it -- a lean seen before and after a 5-minute outage
    "held" for 5 minutes and passed the vote gate on its first sample back.
    Anything that measures "how long has X held" over a sampled series must
    break the run at a gap (`_max_sample_gap()`), and must expire stale
    state *before* refreshing it with the current sample.

11. **A sum over "whichever venues are up" changes when a venue drops.**
    Book flow compared the summed notional of the usable venues at each end
    of its window, so one bid-heavy venue leaving the aggregate (a
    disconnect, or the spread gate during volatility) read as bids being
    pulled -- a fabricated -0.69 flow vote. Compare a quantity over time only
    across the venues present at both ends (samples carry `venue_notional`
    for this). Ratios averaged per venue (gotcha #7) don't have this problem;
    raw totals do.

12. **A key derived from a value scaled by itself is a constant.** Wall
    tracking matched walls with `round(price / (price * bps / 1e4))`, which
    is `round(1e4 / bps)` for every price: all walls on a side shared one
    key, and a new wall inherited an old one's age -- defeating the spoof
    filter that wall age exists for. Bucket in log price instead.

13. **Kalshi's 60-second settlement average makes the true probability
    MORE extreme than the point-in-time model, not less.** The caveat was
    once written backwards in four places (docstring, README, CLI,
    dashboard). Averaging lowers the terminal variance (to sigma^2 * (T -
    ~39.5s)), so a point-in-time N(d2) is too close to 50%: ~79% vs ~92% at 1
    minute left, spot 5bp above the strike. `settlement_probability()` now
    models the average and matches a Monte Carlo of the real rule
    (`tests/test_kalshi.py`; `review/settlement_averaging.py` prints the
    table). The one remaining approximation is the final minute, where the
    already-taken prints are assumed equal to spot -- keep saying so in the
    output if you touch it.

14. **Streamlit sessions share module state.** Every browser tab's script
    runs in one process, so `app.py` writing sidebar values into the
    `config` module made a new tab open on the last tab's pair and
    threshold, and let two live tabs render one pair's data under the
    other's label. `app.py` now works on a per-run copy (`cfg`); anything
    added to the page must take `cfg` explicitly and never assign to a
    module attribute. The same goes for the logs: a rerun is not an
    observation (hence `LOG_MIN_INTERVAL_SECONDS`).

15. **Overlapping outcome windows make n lie.** The backtest scores every
    candle against the next `BACKTEST_LOOKAHEAD_CANDLES`, so neighbouring
    rows share most of their outcome and state votes fire on long runs:
    on random walks the raw n overstated the evidence ~3.5x, and weights
    built on it moved +/-0.08-0.16 from noise alone. Use a bucket's `n_eff`
    for any sample-size judgement, and measure edge against its
    `base_rate`, never against 0.5.

16. **Two writers on one log corrupt it unless the append is locked.** The
    dashboard, the console bot and every open tab write the same CSVs.
    Unlocked, "check the header, then append" let 8 simultaneous writers
    put 7-8 header lines in a new file (numeric columns then read back as
    strings), let a writer read a half-written header and rotate a good
    file aside, and on a layout change crashed the rename losers and lost
    their rows. `csv_utils.append_row()` holds `<log>.lock` (O_CREAT |
    O_EXCL, stdlib-only so it works on Windows) across the header check,
    rotation, duplicate check and write. Never open a log for append
    anywhere else. `review/csv_writer_race.py` shows before and after;
    `tests/test_csv_utils.py` stress-tests it with separate processes.
    On Windows, a lock deleted while another process still has it open is
    "delete pending", and re-creating it raises `PermissionError`, not
    `FileExistsError`; `file_lock` treats that as busy there (only there --
    elsewhere it's a real permission error) and waits like any other.
    First seen on the user's Windows machine, where it crashed writers.

17. **Coinbase's `match.side` is the MAKER's side.** A match with side
    `"sell"` filled a resting sell order, so the taker *bought*.
    `parse_coinbase_match()` once passed it through as the taker side,
    which counted every Coinbase trade backwards in the live `agg_delta`
    (and labelled Coinbase whale trades backwards) while Binance's `m`
    and Bybit's `S` were read correctly. Same family as gotcha #3: three
    venues, three conventions for "who was the aggressor". Every trade
    parser must return the TAKER side; `tests/test_exchange_feeds.py`
    pins it per venue and checks that one taker-buy normalizes to "buy"
    on all three.

18. **Windows doesn't read text as UTF-8 by default.** `open()` and
    `Path.read_text()` without `encoding=` use the locale code page there
    (cp1252), and `app.py` contains UTF-8 characters, so a bare
    `read_text()` of it in `tests/test_app.py` stopped the whole suite at
    collection on the user's machine while passing on Linux. Pass
    `encoding="utf-8"` whenever a file can hold non-ASCII text.
    `tests/test_portability.py` re-runs collection (and the tests that read
    `app.py`) in a child interpreter with UTF-8 mode off and an ASCII locale.

19. **Coinbase's bare `/candles` URL is CDN-cached for 5 minutes.** It
    comes back `cache-control: public, max-age=300`, a Cloudflare `HIT`.
    Measured from the user's machine, the newest 1-minute candle advanced
    23:17 -> 23:22 -> 23:27, with its close frozen in between (even a forming
    candle's 24-second-old snapshot), up to 5 minutes old and $48 off
    `/ticker`. The same request with `start`/`end` was a `MISS` in all 18 of
    18 polls, up to the forming minute, so `get_candles` always sends a range
    ending now.
    Never drop it to "simplify". Any price someone acts on still comes from
    `data_fetch.latest_price()` (the ticker, with a flagged fallback),
    because even a fresh candle lags the last trade. Never call the network
    from tests: `tests/conftest.py` makes any real HTTP request raise.

## Testing approach

There **is** now a committed suite, in `tests/`, run with `pytest` (add
`pip install pytest`; it's the only dev-only dependency). It was added
alongside the order-book work and currently covers `orderbook.py`,
`orderbook_feeds.py`, the order-book paths through `signals.py`, and the
CSV logging in `main.py`.

Everything before that was tested with throwaway scripts, run once and
discarded. The standard those held to is the standard the suite holds to,
and is what any new tests should match:

- Anything touching real math gets cross-validated against an independent
  reference, not just checked for "runs without crashing". Examples:
  `kalshi.py`'s normal CDF was checked against `scipy.stats.norm.cdf` to
  machine precision; `reconstruct_agg_delta_series` against a naive
  O(n*m) brute force; and in the committed suite, the banded/decayed depth
  sums against a deliberately naive brute-force implementation (no sorting
  assumptions, no early exit) across randomized books -- which is
  specifically what would catch a bug in `side_depth`'s early `break`.
- Anything parsing an external API's schema is tested against real,
  verbatim example payloads from that API's own documentation -- never
  guessed field names. `tests/test_orderbook_feeds.py` names its sources
  in the module docstring; keep doing that, so the next person can
  re-verify them.
- Anything writing a CSV log is tested for the "a field appears or
  disappears partway through a session" case explicitly, and for the
  "column layout changed between releases" case -- see gotcha #1, both
  routes to the same corruption.
- Statistical/derived code is tested against hand-constructed data with a
  known, exact expected answer. The persistence and flow tests build a
  history by hand and assert the exact seconds and the exact `log` value.
- Where a property matters more than a value, test the property: flow is
  asserted to be antisymmetric and to cancel a book-wide liquidity swing;
  wall detection is asserted to use the median rather than the mean, via a
  wall large enough to hide itself under a mean.

Also covered now: `kalshi.py`, `weighting.py`, `backtest.py`, `backtest_agg_delta.py`,
`kalshi_tracking.py`, `paper_trading.py`, `kalshi_scalp_study.py`, `kalshi_api.py`, `indicators.py`, `divergence.py`,
`outlook.py`, `dashboard_data.py`, `charts.py` and `app.py`. Writing them turned up three real bugs, all fixed:
`compute_calibration` borrowed an older row's Kalshi quote (`groupby().first()`
is per-column) and crashed when only some contracts had quotes, and
`reconstruct_agg_delta_series` let float residue pose as a reading in an empty
window.

Every module now has tests. The dashboard is covered by `tests/test_app.py`
using Streamlit's `AppTest` (the real `app.py` runs; only Coinbase REST, the
feed handles and Kalshi's API are stubbed, and the live-feed scenario drives a
real `BookAggregator` on a fake clock). `app.py`'s `_esc` and
`_strike_path_liquidity` are lifted out with `ast` since a script can't be
imported. `exchange_feeds.py`'s parsers and `TradeAggregator` are covered in
`tests/test_exchange_feeds.py`, whose docstring names the source of every payload
(the exchanges' docs sites were unreachable from the sandbox, so Binance and
Bybit payloads come verbatim from their docs' GitHub sources, Binance futures
from its official SDK's fixture, and Coinbase's `match` schema and meaning from
its official client -- re-verify that one against the live docs when you can).
Every WebSocket runner's callbacks are covered too, by driving the real
closures with the socket, threads, sleeps and REST call faked: the five trade
and liquidation runners and the Coinbase and Bybit depth runners in
`tests/test_feed_runners.py` (what each subscribes to, and what every kind of
message, error and close does to the aggregator), the Binance.US depth
runner's resync in `tests/test_orderbook_feeds.py` (`_BinanceHarness`), and
`run_with_reconnect`'s backoff in `tests/test_exchange_feeds.py`. Not covered:
visual layout/CSS, which needs a browser, and the feeds against the live
exchanges, which no test here reaches.

`REVIEW_FINDINGS.md` holds the 2026-11-07 full review and its follow-up: what
was fixed (including the shared-config, log-per-rerun, sample-overlap,
base-rate and settlement-averaging items, addressed in the follow-up), what is
still only a recommendation, and the evidence for each. `review/` has the
scripts behind the quantified claims.

Streamlit renders `$...$` as LaTeX, so every string with dollar amounts that
reaches `st.caption`/`st.markdown`/`st.warning` must go through `_esc()` --
including the divergence text inside the outlook, which was missed once and is
now pinned by a test.

`tests/test_kalshi_api.py` runs against verbatim Kalshi payloads saved in
`tests/fixtures/` (captured 2026-11-07; the capture URLs are in the test's
docstring -- re-capture if Kalshi changes its schema). Host note: the tests
mock the network. `kalshi_api.BASE_URL` is `external-api.kalshi.com`, confirmed
live from the user's machine on 2026-11-07. The payloads were captured from
`api.elections.kalshi.com`, which serves the same schema. Cloud sandboxes may
reach neither: their proxy has refused both hosts.

`signals.py` reads the moving-average columns under the names config implies
(`sma_{SMA_FAST}` ...), the same way `indicators.py` builds them, so changing
those periods in config.py just works (`tests/test_signals_price.py`). It used
to read literal `sma_9`/`ema_12` and raise `KeyError` instead.

## Common commands

```bash
streamlit run app.py              # dashboard (primary interface)
python main.py                    # console alternative
python backtest.py                # historical hit-rate backtest (run before backtest_agg_delta.py)
python backtest_agg_delta.py      # adds agg_delta hit rates to the same results file
pytest                            # run the test suite (needs: pip install pytest)
python exchange_feeds.py          # sanity-check trade WebSocket connectivity
python orderbook_feeds.py         # sanity-check level-2 depth WebSocket connectivity
python kalshi_api.py              # sanity-check Kalshi API connectivity
python kalshi.py --strike X --minutes Y   # one-off Kalshi fair-value check
python analyze_correlation.py     # needs LOG_ALL_SIGNALS=True first
python kalshi_tracking.py         # needs LOG_KALSHI_PREDICTIONS=True first
python paper_trading.py          # settle due paper trades and print the account
python review/kalshi_scalp_backtest.py   # scalping backtest on the cached study data (run kalshi_market_study.py fetch first)
python edge_recorder.py --capture        # save verbatim Kalshi orderbook/trades payloads to tests/fixtures/ (do this first, then pytest)
python edge_recorder.py                  # forward recorder for the edge hunt (EDGE_HYPOTHESES.md); runs for months
python review/kalshi_forward_edge.py     # evaluate the pre-registered hypotheses on edge_record.csv
```

## Runtime-generated files (gitignored -- not source, don't hand-edit)

`alerts_log.csv`, `signal_log.csv`, `backtest_results.json`,
`kalshi_predictions.csv`, `kalshi_outcomes_cache.json`, `paper_trades.csv`,
`edge_record.csv`, `edge_trades.csv` -- all created
automatically by the scripts above on first use. Plus
`*.csv.<timestamp>.bak`, written when a log's column layout changes
between releases (gotcha #1) -- old data, still readable, safe to delete
once you've pulled anything you want out of it. And `*.csv.lock`, which
exists only for the milliseconds of an append; one left behind by a
crashed writer is cleared automatically after 10s.
