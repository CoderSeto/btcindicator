# CLAUDE.md

Guidance for Claude Code (or any future contributor) working in this repo.
This is the developer-facing companion to README.md, which is user-facing
(how to run the bot). This file is about how the code fits together, why
certain things are the way they are, and what's already gone wrong once.

## What this is

A BTC (and other pairs) technical-indicator bot with a Streamlit dashboard
and a console mode. It computes 7-8 signals from live Coinbase data plus
multi-exchange trade flow, backtests them against real historical data,
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
  historical-trades client used by `backtest_agg_delta.py`.

**Signal computation**
- `indicators.py` -- SMA/EMA/RSI/MACD/Bollinger/volume-spike, computed
  once per refresh on the full candle DataFrame (vectorized, not
  per-row).
- `orderbook.py` -- order-book imbalance from a level-2 snapshot.
- `signals.py` -- two entry points, both important:
  - `evaluate()` -- the live path, includes order-book imbalance and
    `agg_delta` (7-8 signals total).
  - `evaluate_price_only()` -- the backtestable path, only the 6
    candle-derived signals. Order-book and agg_delta have no historical
    data source (Coinbase exposes no historical order book; no exchange
    has a practical historical multi-exchange trade feed), so they're
    deliberately excluded here. `backtest_agg_delta.py` is the one
    exception -- see below.
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
  `log_signal_snapshot()`, and the `ALL_SIGNAL_NAMES` fixed-schema
  constant both use (see gotcha #1 below).
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
   three times: weighted conviction is shown *separately* from the raw
   composite; the Kalshi model's P(YES) is shown *separately* from the
   bot's backtested confidence; correlation analysis and hit-rate
   backtesting are kept as distinct tools. We have real evidence some
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
   always be present (the running example: `agg_delta`, which only
   exists once the multi-exchange feeds have collected trades) must use
   a fixed, pre-declared column list with blank cells for missing
   values -- never derive the header from whatever happens to be in the
   first row. See gotcha #1.

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

4. **Binance.com blocks US IPs from its spot API/WebSocket.** This
   project uses Binance.US for spot trades specifically because of this
   (see `_binance_us_trade_runner` in `exchange_feeds.py`). Binance's
   futures liquidation stream (`fstream.binance.com`) is a different
   domain and isn't affected the same way -- that one still uses
   binance.com directly.

5. **Windows dev environment needs `py`, not `python`/`pip` directly.**
   Not a code issue, but worth knowing if giving the user shell
   commands: their machine has multiple Python installs, and only the
   `py` launcher (`py`, `py -m pip`) reliably hits the one with this
   project's dependencies installed.

## Testing approach

There's no committed test suite (no `pytest`/`unittest` files in the
repo) -- everything so far was tested via throwaway scripts during
development, run once, discarded. The standard those scripts held to,
worth matching if you add a real suite:

- Anything touching real math (probability formulas, statistical
  estimators, sliding-window aggregations) gets cross-validated against
  an independent reference. Example: `kalshi.py`'s normal CDF was
  checked against `scipy.stats.norm.cdf` to machine precision; the
  agg_delta sliding-window reconstruction was checked against a naive
  O(n*m) brute-force implementation across random data.
- Anything parsing an external API's schema gets tested against real,
  verbatim example payloads from that API's own documentation -- never
  guessed field names.
- Anything writing a CSV log gets tested for the "a field appears or
  disappears partway through a session" case explicitly (see gotcha #1
  -- this is exactly the scenario that wasn't tested and broke).
- Calibration/statistical code gets tested against a hand-constructed
  dataset with a known, exact expected answer (not just "runs without
  crashing").

If you add `pytest`, the functions in `kalshi.py`, `weighting.py`,
`backtest_agg_delta.py`'s `reconstruct_agg_delta_series`, and the CSV
logging functions in `main.py`/`kalshi_tracking.py` are the highest-value
places to start, since they've already caused real bugs once.

## Common commands

```bash
streamlit run app.py              # dashboard (primary interface)
python main.py                    # console alternative
python backtest.py                # historical hit-rate backtest (run before backtest_agg_delta.py)
python backtest_agg_delta.py      # adds agg_delta hit rates to the same results file
python exchange_feeds.py          # sanity-check exchange WebSocket connectivity
python kalshi_api.py              # sanity-check Kalshi API connectivity
python kalshi.py --strike X --minutes Y   # one-off Kalshi fair-value check
python analyze_correlation.py     # needs LOG_ALL_SIGNALS=True first
python kalshi_tracking.py         # needs LOG_KALSHI_PREDICTIONS=True first
```

## Runtime-generated files (gitignored -- not source, don't hand-edit)

`alerts_log.csv`, `signal_log.csv`, `backtest_results.json`,
`kalshi_predictions.csv`, `kalshi_outcomes_cache.json` -- all created
automatically by the scripts above on first use.
