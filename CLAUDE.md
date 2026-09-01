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
  time-ranged historical candle fetch used by the backtest.
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
  hit-rate buckets, saved to `backtest_results.json`.
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
  (`weight = 1.0 + SCALE * (hit_rate - 0.5)`, clipped, falls back to 1.0
  for no-data/small-sample), and computes a `weighted_conviction` score.
  **This is a separate, additive score. It never modifies
  `snapshot.composite`/`strength`, which still drives alerts and CSV
  logging.** See "Design principles" below for why.
- `outlook.py` -- turns snapshot + divergence + backtest confidence +
  weighted conviction into the plain-language narrative shown in the UI.

**Interfaces**
- `main.py` -- console loop. Also owns `log_alert()` and
  `log_signal_snapshot()`, the `ALL_SIGNAL_NAMES` / `BOOK_LOG_FIELDS`
  fixed-schema constants both use, and `_needs_header()`, which rotates a
  log aside when its column layout changes between releases (see gotcha
  #1 below).
- `dashboard_data.py` -- `load_state()`, the fetch+compute pipeline the
  dashboard calls once per refresh.
- `app.py` -- the Streamlit dashboard. Layout is deliberately
  front-loaded: Kalshi calculator and top metrics first, heavier
  sections (chart, live feeds, alert log) collapsed in expanders.
- `charts.py` -- Plotly figure builder.
- `analyze_correlation.py` -- reads `signal_log.csv` (opt-in via
  `LOG_ALL_SIGNALS`) to measure how much `agg_delta` overlaps with the
  other indicators empirically, rather than by reasoning about it.

**Kalshi**
- `kalshi.py` -- fair-value probability calculator for KXBTC15M-style
  binary contracts. Standard lognormal/GBM crossing probability (the
  N(d2) term from Black-Scholes). Pure math, no network calls except in
  its own CLI entry point.
- `kalshi_api.py` -- public (no-auth) Kalshi market-data client.
  `SERIES_TICKER = "KXBTC15M"` -- **not** `KXBTCD`, which is a different,
  separate hourly series (easy to conflate; cross-checked multiple
  sources before committing to this).
- `kalshi_tracking.py` -- logs fair-value predictions (opt-in via
  `LOG_KALSHI_PREDICTIONS`) and checks them against real settled
  outcomes for calibration (bucketed hit-rate table + Brier score,
  compared against Kalshi's own market price as a baseline).

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
   header. `_needs_header()` handles that now -- if the on-disk header
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

Still uncovered, in rough priority order if you're adding more:
`kalshi.py`, `weighting.py`, `backtest_agg_delta.py`'s
`reconstruct_agg_delta_series`, and `kalshi_tracking.py`'s logging.

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
```

## Runtime-generated files (gitignored -- not source, don't hand-edit)

`alerts_log.csv`, `signal_log.csv`, `backtest_results.json`,
`kalshi_predictions.csv`, `kalshi_outcomes_cache.json` -- all created
automatically by the scripts above on first use. Plus
`*.csv.<timestamp>.bak`, written when a log's column layout changes
between releases (gotcha #1) -- old data, still readable, safe to delete
once you've pulled anything you want out of it.
