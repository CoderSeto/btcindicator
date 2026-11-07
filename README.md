# BTC Indicator Bot

A local tool that watches Bitcoin price action and tells you when technical
indicators and order-book pressure line up on a direction. It's a
**monitor/alert tool** — it surfaces signals and logs alerts to a CSV, but
it does not place any trades.

Data comes from Coinbase Exchange's public market data API (candles), plus
Coinbase's, Binance.US's, and Bybit's public WebSocket feeds -- trades and
liquidations for the cross-exchange flow signals, and level-2 depth for the
order-book signals. None of it needs an API key or account.

There are two ways to run it:

## Option A: Dashboard (recommended)

A browser-based UI with live price charts, colored indicator badges, and
an alert log — auto-refreshes on its own.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows with more than one Python installed, go through the `py`
launcher so you hit the interpreter that actually has these dependencies:
`py -m pip install -r requirements.txt` then `py -m streamlit run app.py`.

This opens a browser tab at `http://localhost:8501` on its own. The layout is
built to fit one screen -- the live state is always visible, and the price
chart is the one panel you scroll for:

- **Header strip** — price, the composite signal (bullish/bearish/neutral with
  how many indicators agree), aggregated order-book imbalance, and the Kalshi
  P(YES) for the tracked contract, as four compact cards.
- **Signal row** — every indicator as a coloured dot. 7 to start, rising to 9
  as the feeds warm up: `book_flow` appears once the depth feeds have
  `BOOK_FLOW_WINDOW_SECONDS` of history (~1 min) and `agg_delta` once the
  trade feeds have collected trades. Hover a dot for the full name and vote.
- **Kalshi box** — strike, minutes remaining and auto-track, with P(YES),
  P(NO) and the market's own quote right beneath them.
- **Outlook box** — a plain-language read on which way things are leaning and
  why, plus a "reversal watch" callout when an RSI/price divergence pattern is
  detected. Commentary on the current pattern, not a forecast.
- **Tabs** — Order Book, Feeds, Chart, Alerts, and Notes & limits. Order Book
  opens by default because the chart is tall enough to push the page past a
  screen; it's one click away when you want it.
- **Sidebar** — switch trading pair, adjust the refresh interval (as low as
  5s, useful for watching a Kalshi window's final minute), adjust how many
  aligned indicators trigger an alert, and run a backtest — all without
  touching code.

The three boxes and the tab panel are fixed-height and scroll internally, so
the page height doesn't move around as text wraps or as you switch tabs. If
you have more vertical room than the ~900px this targets, raise `BOX_HEIGHT`
and `TAB_BODY_HEIGHT` near the top of `app.py`.

## Option B: Console bot

The original plain-terminal version — lighter weight, good for running
headless or piping into other tools.

```bash
pip install -r requirements.txt
python main.py
```

(On Windows: `py -m pip install -r requirements.txt` then `py main.py`.)

It prints a status line every `POLL_INTERVAL` seconds:

```
[14:32:05] $61,204.50  composite=BULLISH (5/9)  book=+0.24 (3v, 100% agree, held 41s)  sma_cross:BULL ema_cross:BULL rsi:NEUT macd:BULL bollinger:NEUT volume:NEUT orderbook:BULL book_flow:NEUT agg_delta:BULL
  book: bid $8,412,660 / ask $5,109,003 within band  |  near +0.31, far +0.19, microprice +0.04bps
  flow: resting bids +6.2%, asks -3.1% over 60s (net +0.09)
  wall: $1,340,000 resting 18bps above mid on coinbase (resistance, held 74s)
  outlook: A majority of indicators agree on upside pressure (5/9).
```

(The count starts at 7 and rises to 9 as the feeds warm up -- `book_flow`
needs about a minute of depth history and `agg_delta` needs collected
trades. Until then those two are absent rather than voting neutral. The
`book=` reading is replaced by `imbalance=` if the depth feeds are off or
still connecting -- see "Multi-exchange order book" below.)

Both options share the same alert threshold logic and the same
`alerts_log.csv` — you can run either one (or switch between them) without
losing your alert history.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable parameters — poll interval, indicator periods, thresholds. Start here. |
| `data_fetch.py` | Pulls candles, order book, and ticker from Coinbase's public REST API. |
| `indicators.py` | Computes SMA, EMA, RSI, MACD, Bollinger Bands, and volume-spike detection on the candle history. |
| `orderbook.py` | Order-book metrics: distance-banded, notional, depth-weighted imbalance, microprice, spread, liquidity walls, and cross-venue aggregation. |
| `orderbook_feeds.py` | Live level-2 depth feeds from Coinbase, Binance.US, and Bybit, maintained as local books and sampled over time so persistence, resting-liquidity flow, and wall age can be measured. Run standalone (`python orderbook_feeds.py`) to sanity-check connectivity. |
| `signals.py` | Converts each indicator's raw value into bullish/bearish/neutral, and combines them into a composite read. |
| `main.py` | Console version: the polling loop, status printing, alert logging, and (when enabled) per-cycle signal logging for correlation analysis. |
| `dashboard_data.py` | Fetch + compute wrapper used by the dashboard (same pipeline as `main.py`, adapted to run once per page refresh). |
| `divergence.py` | Detects RSI/price divergence -- a classic pattern for spotting potential reversals. |
| `backtest.py` | Backtests signal patterns against real historical data to compute actual hit rates, including per-indicator hit rates used by `weighting.py`. Runs standalone (`python backtest.py`) or from the dashboard sidebar. |
| `backtest_agg_delta.py` | Backtests `agg_delta` specifically, using Binance.US's historical trade data (the only one of the three live exchanges with a practical historical-trades API). Adds to `backtest.py`'s results -- run `backtest.py` first. |
| `weighting.py` | Turns each indicator's backtested hit rate into a weight, and computes a separate "weighted conviction" score alongside the raw composite. |
| `outlook.py` | Turns the signal snapshot + divergence read + backtest confidence + weighted conviction into a plain-language "what's this suggesting" summary. |
| `charts.py` | Builds the Plotly price/RSI/MACD chart shown in the dashboard. |
| `exchange_feeds.py` | Connects to Coinbase, Binance.US, and Bybit's public WebSocket feeds for aggregated buy/sell volume, whale trades, and liquidations. Run standalone (`python exchange_feeds.py`) to sanity-check connectivity. |
| `analyze_correlation.py` | Analyzes how much `agg_delta` overlaps with the other 8 indicators, from logged signal history. Run standalone (`python analyze_correlation.py`) once you've accumulated some data. |
| `kalshi.py` | Fair-value probability calculator for Kalshi's KXBTC15M-style binary contracts. Runs standalone (`python kalshi.py --strike ... --minutes ...`) or from the dashboard. |
| `kalshi_api.py` | Public (no-auth) client for Kalshi's own market data API -- auto-discovers the currently open KXBTC15M contract's strike, close time, and quoted price. Run standalone (`python kalshi_api.py`) to sanity-check connectivity. |
| `kalshi_tracking.py` | Logs fair-value predictions and checks them against real settled outcomes, to measure calibration empirically. Run standalone (`python kalshi_tracking.py`) once you've accumulated some tracked predictions. |
| `app.py` | The Streamlit dashboard itself. |
| `tests/` | Pytest suite covering the order-book math (cross-validated against brute-force references), every exchange payload parser (against the exchanges' own documented examples), the Binance sequence-gap handling, and the CSV log schema. Run with `pytest`. |

## Indicators used

**Trend**
- SMA crossover (9 vs 21 period) — fast average crossing the slow average
- EMA crossover (12 vs 26 period) — same idea, more weight on recent price
- MACD (12/26/9) — momentum via the MACD line crossing its signal line

**Mean reversion / overextension**
- RSI (14 period) — flags overbought (>70) / oversold (<30) conditions
- Bollinger Bands (20 period, 2 std) — flags price pushing outside the bands

**Confirmation**
- Volume spike — current volume vs. a rolling average, direction taken from
  whether that candle closed up or down
- Order-book imbalance — resting bid vs. ask *notional* within a
  basis-point band of mid, weighted toward the touch, aggregated across
  Coinbase/Binance.US/Bybit, and only counted once the lean has *held*
  for `BOOK_PERSISTENCE_SECONDS` (see below)
- Book flow — whether resting liquidity is being added to one side or
  pulled from the other over the last `BOOK_FLOW_WINDOW_SECONDS` (only
  present once the depth feeds have that much history)
- Aggregated exchange delta — net taker buy vs. sell volume across
  Coinbase, Binance, and Bybit combined (only present once the
  multi-exchange feeds have collected trades -- see below)

Each indicator votes bullish, bearish, or neutral. The composite direction
is whichever side has more votes, and "strength" is how many indicators
agree. Up to 9 can be present at once; the two feed-dependent ones
(`book_flow`, `agg_delta`) are simply absent until their feeds have
warmed up rather than voting neutral by default. This is a simple voting
scheme, not a weighted or backtested model —
tune `COMPOSITE_ALERT_THRESHOLD` and the individual thresholds in
`config.py` to fit how noisy or strict you want it.

## Outlook and reversal detection

On top of the indicator vote, the bot generates a short plain-language
**outlook**: which way the indicators are leaning, how strong the agreement
is, and which specific indicators are driving it (e.g. "RSI is in oversold
territory, which has often preceded a short-term bounce").

It also runs a separate **RSI/price divergence** check (`divergence.py`)
over the last `DIVERGENCE_LOOKBACK` candles (40 by default). Divergence is
a classic reversal pattern:

- **Bearish divergence** — price makes a higher high, but RSI makes a
  *lower* high at the same point. Upward momentum is fading even as price
  pushes up.
- **Bullish divergence** — price makes a lower low, but RSI makes a
  *higher* low at the same point. Downward momentum is fading even as
  price pushes down.

When a divergence is found and it runs counter to the current composite
lean, it shows up as a "reversal watch" caution — the bot's way of
flagging "the trend indicators still say X, but there's an early sign it
might be turning."

**This is pattern-based commentary, not a forecast.** It describes what
the indicators are showing right now and what that combination has
tended to precede historically — it isn't a guarantee of what price will
do next, and it isn't financial advice.

## Multi-exchange live feeds

Coinbase alone only shows one exchange's worth of activity. `exchange_feeds.py`
connects directly to Coinbase, Binance.US, and Bybit's public trade
WebSocket feeds (plus Binance's and Bybit's liquidation feeds) and
aggregates them into:

- **`agg_delta`** — a signal of its own: the net taker buy vs. sell volume across
  every connected exchange over a rolling window (`AGG_WINDOW_SECONDS`, 60s
  by default). This is a different thing from the order-book imbalance
  signal — order-book imbalance reflects *resting* orders, `agg_delta`
  reflects *actual executed* aggression across exchanges. Both are kept as
  separate signals rather than one replacing the other.
- **Whale trades** — any single trade at or above `WHALE_TRADE_USD_THRESHOLD`
  ($100k by default) is logged immediately, with a real-time console alert
  in the terminal version and a live table in the dashboard.
- **Liquidations** — forced position closures on Binance's and Bybit's
  USDT-perpetual futures, shown the same way.

This was built directly against each exchange's own public API rather than
depending on a third-party aggregator like aggr.trade, which doesn't expose
a public API for external use (it connects browser-to-exchange directly,
and its own historical-data server isn't publicly hosted). The same
approach is used for depth — see the next section.

**Why Binance.US instead of binance.com:** Binance.com blocks spot
API/WebSocket access from the US for regulatory reasons. Binance.US is a
separate, independently licensed exchange with its own distinct order
book, built specifically to serve that market instead — so that's the one
connected here. (binance.com's spot feed, and its higher liquidity, is
available outside the US; if you're running this from outside the US and
want it back, `_binance_us_trade_runner()` in `exchange_feeds.py` is a
ready-made template for it — same shape, different URL.) Binance's
liquidation feed (`binance_liq`) is a separate domain from its spot
feed and isn't affected by this either way.

**Toggle:** set `ENABLE_EXCHANGE_FEEDS = False` in `config.py` to run on
Coinbase REST data alone, no background WebSocket connections.

**Check connectivity yourself:** run `python exchange_feeds.py` — it
connects for 20 seconds, prints each exchange's status and running totals,
then exits. If an exchange never reaches `connected`, that's usually a
network/firewall issue on your end blocking that specific domain, not a
bug in the bot.

**Limitations worth knowing:**
- **Liquidations are a derivatives thing.** They reflect leveraged
  perpetual-futures positions on Binance/Bybit being force-closed — not
  spot trading, and not Coinbase (which doesn't offer retail perpetuals on
  this product). A liquidation is a related-market signal, not literally
  "someone sold your exact BTC-USD pair."
- **Not every pair is covered on every exchange.** Binance/Bybit use
  different symbols than Coinbase (`BTCUSDT` vs `BTC-USD`); pairs with no
  clean equivalent (e.g. `BTC-EUR`) just run on Coinbase alone — the
  dashboard and console bot both handle this gracefully, they don't error.
- **Switching pairs repeatedly in one dashboard session** leaves the
  previous pair's background connections running (a minor resource cost,
  not a correctness issue) until you restart the app.

## Multi-exchange order book

`orderbook_feeds.py` keeps a **live level-2 order book** for each of
Coinbase, Binance.US, and Bybit, and `orderbook.py` measures it.

**On aggr.trade, since it's the usual reference point here:** aggr
aggregates *trades*, not order books. It has no order-book view — the
request for one ([Tucsky/aggr#346](https://github.com/Tucsky/aggr/issues/346))
was closed without an implementation — and no public API for outside
consumers either way. So there is no aggr.trade order book to pull from.
What aggr *does* do is connect straight to each exchange's own public
WebSocket from the browser and merge the streams client-side, and that's
exactly the approach taken here, extended to depth: same public feeds,
same merge-it-yourself idea, no third-party dependency, still no API keys.

### What changed, and why it matters

The original order-book signal was one Coinbase REST snapshot per polling
cycle: sum the base volume over the top 25 levels on each side, take the
ratio. Four things were wrong with that.

1. **A "level" isn't a unit of anything.** Coinbase quotes BTC-USD in
   $0.01 increments, so 25 levels can span well under a dollar. Bybit's
   spot book is coarser. "Top 25 levels" covers a completely different
   slice of the book on each exchange, which makes the numbers
   non-comparable and impossible to aggregate honestly. Depth is now
   measured in **basis points from mid**, which means the same thing
   everywhere.
2. **Distance was ignored.** Size resting 1bp from mid counted the same as
   size 80bps away, though only the first will interact with price on this
   bot's timescale. Levels are now weighted `exp(-distance / BOOK_DECAY_BPS)`.
3. **Base volume, not notional.** Summing `size` weights a level by coin
   count. It's dollars that have to be absorbed, so it's notional now.
4. **No time dimension at all.** This is the big one. Resting size is
   placed and pulled constantly, so a book that looks lopsided in the one
   frame you sampled is close to noise.

### What the live book makes possible

- **Persistence.** The order-book signal now only votes once a lean has
  held the same side past `BOOK_IMBALANCE_THRESHOLD` *continuously* for
  `BOOK_PERSISTENCE_SECONDS` (20s by default). "It has looked like this
  for the last half a minute" is a far stronger statement than "it looks
  like this right now", and this single gate removes most of the noise the
  old signal had. Set it to `0` to get the old instantaneous behavior.
- **Flow (`book_flow`).** Whether resting size is being *added* to the bid
  or *pulled* from the ask. Both look identical in a snapshot — you need
  two points in time to tell them apart — and both tend to precede
  movement. Reported as `log(bid growth) - log(ask growth)` over
  `BOOK_FLOW_WINDOW_SECONDS`, so a doubling on one side and a halving on
  the other are equal and opposite, and a book-wide liquidity swing
  (everyone widening at once) cancels rather than reading as direction.
- **Wall age.** A large level that has sat there for two minutes is
  evidence of something; one that appears when price approaches and
  vanishes before it arrives is a spoof. Only walls held for at least
  `BOOK_WALL_MIN_AGE_SECONDS` are reported at all.
- **Venue agreement.** Three exchanges leaning the same way at 0.3 is a
  different picture from one exchange at 0.9 dragging two flat ones along.
  That's reported as its own number rather than being folded into the
  imbalance, because collapsing both into one figure would hide exactly
  that difference.
- **Microprice.** The size-weighted mid, which leans toward whichever side
  has *less* resting size (that's the side that clears first).

### Why per-venue imbalances are averaged instead of the books being merged

The obvious approach — concatenate every exchange's levels into one big
book and run the same math over it — is wrong, and quietly so. Coinbase
quotes BTC-USD; Binance.US and Bybit quote BTCUSDT. USDT is not a dollar,
and the basis between those pairs drifts by tens of basis points. Pooling
their price levels builds a book whose "mid" is a fiction and whose near
band mixes levels 2bps from mid on one venue with levels 30bps away on
another.

Each venue's imbalance is a dimensionless ratio measured against *its own*
mid, so those are directly comparable no matter what the basis is doing.
They're averaged, weighted by each venue's in-band notional — and a venue
whose spread has blown out past `BOOK_MAX_SPREAD_BPS` is dropped for that
refresh, since that usually means a stale or dislocated feed rather than
information.

### How each feed is maintained

| Exchange | Stream | Notes |
|---|---|---|
| Coinbase | `level2_batch` | Full-book snapshot then incremental updates, batched every 50ms. `level2_batch` rather than `level2` because the plain `level2` channel has required authentication since 2023-08-01, and this project uses no API keys anywhere. Same JSON schema either way. |
| Binance.US | `<symbol>@depth@100ms` + REST snapshot | Follows Binance's own documented "how to manage a local order book correctly" procedure: buffer stream events, fetch the snapshot, discard what it already covers, replay the rest, then check first/last update IDs for gaps on every event and rebuild from scratch if one appears. The partial-book stream (`@depth20`) would be far less code, but 20 levels of BTCUSDT span a fraction of a basis point — nowhere near the band the signal is read from. |
| Bybit | `orderbook.200.{symbol}` (spot) | Snapshot then deltas; size `0` removes a level, and `u == 1` means Bybit's own service restarted and the book must be re-seeded. Self-seeding, so no REST bootstrap. |

**Toggle:** set `ENABLE_BOOK_FEEDS = False` in `config.py` to skip the
depth WebSockets entirely. The order-book signal then falls back to the
original single Coinbase REST snapshot — still a working signal, just one
venue, one frame per poll, and with no persistence gate available. Set
`ENABLE_BOOK_FLOW_SIGNAL = False` to keep the depth feeds but leave
`book_flow` out of the composite (it still shows in the dashboard).

**Check connectivity yourself:** run `python orderbook_feeds.py` — it
connects for 45 seconds and prints the aggregated read each second.

**Limitations worth knowing:**
- **Neither order-book signal can be backtested.** No exchange publishes
  historical level-2 snapshots, so unlike the candle indicators there is
  no hit rate behind `orderbook` or `book_flow` at all. Both always carry
  weight 1.00 in weighted conviction. Nothing here is a measured edge — it
  is a better-constructed *description* of the book, which is not the same
  claim. Recording the live feeds going forward would eventually build
  such a history; nothing downloadable exists to check against today.
- **Resting size is not a barrier.** A wall that has held for two minutes
  can still be pulled the instant price arrives, and it says nothing about
  how much aggressive flow is coming the other way.
- **Flow is measured on a mid-anchored band.** When price moves, the set
  of levels inside the band changes for mechanical reasons unrelated to
  anyone adding or pulling size. Both sides shift together so much of it
  cancels, but not all of it. Treat `book_flow` as suggestive, never as a
  clean measurement of intent.
- **Spoofing is real and this can't fully see through it.** Wall age is
  the only evidence available, and it's evidence, not proof.
- **Warm-up.** The persistence gate needs `BOOK_PERSISTENCE_SECONDS` and
  flow needs `BOOK_FLOW_WINDOW_SECONDS` of history, so both stay neutral
  or absent for roughly the first minute after startup.

## Backtested confidence

The outlook above is qualitative -- "a majority of indicators agree."
`backtest.py` adds an actual number to that: it pulls a large chunk of
real historical BTC candles, replays the exact same signal and divergence
logic at every point in that history, and checks what price actually did
`BACKTEST_LOOKAHEAD_CANDLES` (5, by default) candles later. That gives an
empirical hit rate for each signal pattern -- e.g. "this pattern has been
followed by an upward move within 5 candles 63% of the time, against a 51%
base rate for any window (n=140, about 45 independent)."

Two numbers sit next to every hit rate, because a hit rate alone overstates
what it shows:

- **The base rate** -- how often *any* window in the backtest moved that way.
  In a stretch that drifted up, every bullish call looks good; a call is
  only worth something to the extent it beats that.
- **The effective sample size** ("about N independent"). Every candle is
  scored against the next 5, so neighbouring rows share most of their
  outcome window, and state votes like SMA cross fire on long runs of
  consecutive candles. On simulated random walks the raw n overstated the
  evidence ~3.5x for those signals. `backtest.effective_sample_size()`
  corrects for the overlap.

Run it two ways:

```bash
python backtest.py          # standalone, prints a summary table
```

or click **Run backtest now** in the dashboard sidebar. Either way, results
are cached to `backtest_results.json` and reused until you re-run it --
the dashboard doesn't re-backtest on every refresh, since it's a much
heavier operation than the live polling loop (it pulls thousands of
historical candles in paginated requests, so it takes 10-30 seconds).

**Important limitations, on purpose, not glossed over:**

- **No order-book history at all; agg_delta separately, if you run it.**
  Coinbase's public API doesn't expose historical order-book snapshots, so
  order-book imbalance can never be backtested. `backtest.py` on its own
  also doesn't cover `agg_delta` (same reason -- no practical multi-exchange
  historical trade source), but `backtest_agg_delta.py` can add it in
  separately using Binance.US's history -- see "Backtesting agg_delta"
  below. Until you run that, `backtest.py` covers the 6 candle-based
  indicators (SMA/EMA cross, RSI, MACD, Bollinger, volume spike). The
  confidence numbers are matched against a price-only recomputation of the
  current signal state for this reason.
- **Small samples happen, and are labeled.** Some signal combinations are
  rare, especially at high alignment counts (5-6 of 6 indicators agreeing
  at once). Any hit rate built from fewer than `BACKTEST_MIN_SAMPLE`
  *effective* (independent) samples (15, by default) is flagged "small
  sample -- treat with caution" rather than presented with false confidence.
- **A hit rate is not a probability of the future.** It's a description of
  what happened in the specific historical window you backtested. Markets
  change regime; a pattern that worked in the last few days of data isn't
  guaranteed to keep working. Re-run the backtest periodically, and treat
  the numbers as one more input, not a signal to act on by itself.
- **Results are pair/granularity-specific.** Switching trading pairs in
  the dashboard sidebar won't show stale confidence numbers from a
  different pair -- cached results are only used when they match the
  current `PRODUCT_ID` and `GRANULARITY`, otherwise you'll see "no
  backtest yet" until you re-run one for that pair.

## Weighted conviction

Everything above treats all 7-8 indicators equally -- the composite is a
raw count, "5 of 8 say bullish." `weighting.py` adds a second opinion:
what if each indicator counted in proportion to how often it's actually
been right, based on the same backtest?

Once you've run a backtest, `backtest.py` also tracks a hit rate for each
of the 6 backtestable indicators' bullish and bearish calls individually
(not just the composite). `weighting.py` turns each one into a weight:

```
weight = 1.0 + INDICATOR_WEIGHT_SCALE * (hit_rate - base_rate)
```

`base_rate` is how often the backtested window moved in that direction at
all, so an indicator that does no better than "always say bullish" gets
weight 1.0 -- same as unweighted. Beating the base rate scales its weight
up; trailing it scales it down (floored at `INDICATOR_WEIGHT_MIN` rather
than going to zero or negative, so a historically weak indicator still
counts a little rather than vanishing from the story).

A weight stays at 1.0 unless the evidence is there: fewer than
`BACKTEST_MIN_SAMPLE` effective samples, or an edge within
`INDICATOR_WEIGHT_MIN_Z` (2) standard errors of zero, means "no measured
edge". That second gate matters: on backtests of pure random walks,
where no indicator can have an edge, ungated weights still spread
+/-0.08 to +/-0.16 from noise alone. The Notes tab's weights table says
why each weight is what it is, and when nothing voting has a measured
edge the outlook says weighting changes nothing rather than quoting a
"weighted conviction" figure.

The dashboard shows this as a line under the Outlook -- "weighted
conviction is 72% bullish" -- with an expander showing exactly which
indicators contributed which weight. **Most of the time it'll agree with
the raw composite.** The interesting case is when it doesn't: a handful
of indicators with strong historical track records can outweigh a larger
number of indicators that individually don't add much value, which the
raw count alone can't see. When that happens, both the raw and weighted
reads are shown side by side -- neither one silently overrides the other,
and nothing about the raw composite (the alert threshold, the CSV log,
the outlook headline) changes because of this.

**Order-book imbalance always gets weight 1.0** -- no historical order-book
data exists to backtest it, full stop. **The aggregated exchange delta
starts at weight 1.0 too, but doesn't have to stay there** -- run
`backtest_agg_delta.py` (see below) and it'll get a real backtested weight
like everything else.

**A real methodological caveat, not glossed over:** these weights come
from the *same* historical window the backtest already covers. That's a
genuine historical edge, but it is *not* an independent, out-of-sample
test proving the weighted score actually predicts better than the raw
count -- it's entirely possible to construct a weighting scheme that
looks great on the data it was built from and adds nothing (or even hurts)
going forward. Treat the weighted line as one more input to sit alongside
the raw composite, not a replacement for your own judgment.

## Backtesting agg_delta

`agg_delta` starts out unbacktestable for the same reason order-book
imbalance is: it needs historical tick-by-tick trade data across
exchanges, and there's no practical source for that live-multi-exchange
combination. But checked one exchange at a time, it turns out **one of
the three does** have a usable historical API:

| Exchange | Historical trades API | Usable for backtesting? |
|---|---|---|
| Coinbase | Cursor-paginated by trade ID, not time-range | No -- would take an impractical number of requests to cover multiple days for a liquid pair |
| Bybit | Caps at the last 60 trades, no pagination | No |
| Binance.US | `startTime`/`endTime` range queries, 1000 trades/request, paginated | **Yes** |

`backtest_agg_delta.py` uses Binance.US's history to reconstruct what the
rolling-window `agg_delta` reading would have been at each candle in your
existing backtest, replaying the exact same window math
`exchange_feeds.TradeAggregator` uses live, then checks the same "did
price move as predicted" outcome as every other indicator.

```bash
python backtest.py              # if you haven't already
python backtest_agg_delta.py    # adds agg_delta hit-rate buckets to the same results file
```

The result gets merged into `backtest_results.json` — `weighting.py` picks
it up automatically from there, and `analyze_correlation.py` now has real
historical data to compare `agg_delta` against instead of only whatever's
accumulated live.

**Be clear-eyed about what this is measuring.** The live `agg_delta`
combines Coinbase + Binance.US + Bybit. This backtest only reflects
Binance.US's own trade flow — a real historical signal, but an
*approximation* of the live one, not a reconstruction of it. Binance.US
also has thinner liquidity than Coinbase, so some candles may have little
or no Binance.US coverage in their window; those are skipped rather than
guessed at (the report tells you how many candles had coverage). Treat the
resulting hit rate as a reasonable proxy, not a promise that it matches
how the full 3-exchange signal would have performed.

## How much does agg_delta overlap with the other indicators?

A fair question once you've got 8 signals instead of 7: is `agg_delta`
telling you anything the others don't, or just restating what a couple of
them already say? `analyze_correlation.py` answers this from real logged
data instead of reasoning about it.

(This is a different question from the hit-rate backtesting above --
`backtest_agg_delta.py` tells you whether `agg_delta` has historically
been *right*; this tells you whether it's telling you something
*different* from the other indicators. Worth running both.)

**Setup (off by default, since it's a continuously-growing file):**

```python
# in config.py
LOG_ALL_SIGNALS = True
```

Then run `main.py` or the dashboard for a while — every cycle's full set
of indicator votes gets appended to `signal_log.csv` (not just
alert-triggering moments, so the sample isn't biased toward
strong-alignment periods). Once you've got a reasonable stretch of data
with the multi-exchange feeds connected:

```bash
python analyze_correlation.py
```

For each of the other 8 indicators, against every cycle where `agg_delta`
was present, it reports:

- **Agreement rate** — of the cycles where both had a directional
  (non-neutral) vote, how often they matched.
- **Correlation** — Pearson correlation between the two, encoded
  bullish=+1 / neutral=0 / bearish=-1, across every cycle.

**What this can and can't tell you:** high overlap means the two
indicators are carrying similar information right now — it does *not*
automatically mean one is safe to drop. A short collection window, one
particular market regime, or plain coincidence can all inflate agreement
that wouldn't hold up over a longer stretch. `volume` (similar timescale,
same idea of detecting a burst of activity) is the one I'd expect to show
the most overlap going in; `orderbook` (resting liquidity, a genuinely
different data type from executed trades) the least — but that's a
starting hypothesis for you to check against the actual numbers, not a
conclusion baked into the tool.

## Kalshi 15-minute BTC fair value

Kalshi's `KXBTC15M` series opens a fresh binary contract every 15 minutes:
does BTC close above a fixed strike (set near the price at the window's
open) at the end of the window? This is a genuinely different question
from everything else in this bot. The Outlook section answers "is price
leaning up or down right now" — Kalshi's contract asks "what's the actual
probability price is above a *specific fixed number* at a *specific
future time*." A qualitative lean isn't a probability. `kalshi.py` tries
to give an actual one.

**The model:** the standard approach for pricing exactly this kind of
binary/digital contract — the same math behind the N(d2) term in
Black-Scholes. Given the current price, the strike, time remaining, and
recent realized volatility (estimated from your own candle history), it
computes the theoretical probability of finishing above the strike under
a driftless random walk. This is the same formula independently-built
tools for analogous Kalshi markets (e.g. their 15-minute silver contracts)
use for fair value, which was a useful sanity check while building this.

**Two ways to use it:**

```bash
python kalshi.py --strike 105000 --minutes 8
```

or use the **Kalshi 15m Fair Value** section in the dashboard, which by
default auto-tracks the currently open KXBTC15M window (see below) --
untick "Auto-track current window" to type in a strike/time by hand
instead.

**Deliberately NOT blended with the bot's own signals.** The dashboard
also shows the bot's backtested confidence numbers alongside the model
output, clearly labeled as a separate read — not mathematically combined
into one probability. We have real backtested evidence for how often our
signals have been *right*; we don't have validated evidence for the
*correct way to blend that into a probability estimate*, and inventing a
blending formula would be exactly the kind of unjustified-precision this
whole project has tried to avoid. Compare the two numbers yourself. (For
the comparison to mean anything, set `BACKTEST_LOOKAHEAD_CANDLES = 15` in
`config.py` and re-run `python backtest.py` first — the default is tuned
for a 5-minute horizon, not 15.)

## Auto-tracking the live Kalshi market

`kalshi_api.py` connects to Kalshi's own public market-data API (no
account or API key needed) to find whichever `KXBTC15M` contract is
currently open and pull its exact strike, close time, and Kalshi's own
quoted YES/NO price -- so you don't have to copy those numbers over by
hand while a 15-minute window is ticking down.

With "Auto-track current window" checked (the dashboard default):

- **Strike** syncs from Kalshi automatically, but only resets when the
  window itself actually rolls over to a new contract -- so if you edit
  it by hand mid-window to explore a hypothetical, your edit sticks
  instead of being overwritten on the next refresh.
- **Minutes remaining** always updates live off Kalshi's real close time
  for the current window (more precise than assuming a clean quarter-hour
  boundary, since it comes straight from the market record).
- **Kalshi's own quoted price** shows up as a 4th metric next to the
  model's P(YES)/P(NO), with the gap between them called out directly --
  that gap is the actual "is there an edge here" question, though see the
  caveat below before reading too much into it.

**Sanity-check it yourself first.** This was built against Kalshi's
documented API schema with thorough mocked testing, but I can't reach
`external-api.kalshi.com` from the environment this was built in, so it's
never actually been verified against Kalshi's live API. Run
`python kalshi_api.py` standalone once -- it should print a real ticker,
a strike near the current BTC price, and a close time within the next 15
minutes. If something looks off (particularly if `SERIES_TICKER` in the
file, currently `"KXBTC15M"`, ever needs to change -- Kalshi does
restructure series occasionally), that's the first thing to check.

**The gap is a starting point, not a signal.** A meaningful difference
between the model's P(YES) and Kalshi's own quote could reflect a real
mispricing -- or just this model's own approximations (the basis-risk and
settlement-averaging caveats above) being wrong instead of the market.
Worth understanding *why* a gap exists before trusting it.

**Limitations, stated plainly:**

- **Basis risk.** Kalshi settles on CF Benchmarks' Real-Time Index — a
  multi-exchange composite — not Coinbase alone. This model uses
  Coinbase's price as a proxy, which can diverge from the real settlement
  source by a few dollars, especially during a volatile close. On a
  contract decided by a small margin, that gap is exactly what matters.
- **Settlement is an average, not a snapshot.** Kalshi's reference price
  is a 60-second average at the window's close, not a single tick — the
  well-known trap where price ticks above the strike right at the buzzer
  and the contract still settles NO, because the average of the closing
  minute decided it, not the last print. P(YES) prices that average
  (`kalshi.settlement_probability`). Averaging lowers the variance of what
  settles, so this sits further from 50% than the textbook point-in-time
  N(d2) -- under a point apart early in the window, large at the end: with
  1 minute left and spot 5bp above the strike, point-in-time says ~79%
  and the averaged settlement ~92%. The point-in-time figure is still
  shown by the CLI and logged, for comparison. **What's still approximate
  is the final minute:** some of the 60 prints have already been taken,
  their real values aren't available here, and they're assumed equal to
  the current price. The dashboard flags it.
- **No drift, no jumps.** Standard simplifying assumption for pricing
  this kind of contract — not a claim that BTC actually moves this way.
- **Not financial advice.** A calculator, not a guarantee, and this
  project is not a licensed financial advisor.

`KALSHI_VOL_LOOKBACK_MINUTES` (in `config.py`, default 120) controls how
much recent price history feeds the volatility estimate — longer is more
stable but slower to react to a recent volatility regime change; shorter
is more responsive but noisier.

## Is the fair-value model actually any good?

One resolved contract isn't evidence either way. A prediction of "82%"
is *expected* to lose about 1 time in 6 — a single loss at that price is
completely consistent with the model working fine. What actually answers
"is this any good" is checking, across many resolved contracts, whether
"82% predictions" really do resolve YES about 82% of the time.
`kalshi_tracking.py` builds that track record.

**Turn on logging** (off by default, since it's a continuously-growing file):

```python
# in config.py
LOG_KALSHI_PREDICTIONS = True
```

With that on, every fair-value computation made while the dashboard is
auto-tracking a real Kalshi contract gets logged — ticker, strike,
current price, minutes remaining, the model's P(YES), and Kalshi's own
quoted price at that moment. (Manual, non-auto-tracked entries aren't
logged, since without a real ticker there's no way to look up what
actually happened later.)

**Check the results:**

```bash
python kalshi_tracking.py
```

This looks up each logged contract's real outcome via Kalshi's API
(caching resolved ones locally so they're not re-queried every run), then
reports two things:

- **A calibration table** — bucketed by predicted probability (0-20%,
  20-40%, ..., 80-100% by default), showing what fraction of predictions
  in each bucket actually resolved YES. A well-calibrated model's
  80-100% bucket should resolve YES close to 80-100% of the time, not
  systematically lower or higher.
- **Brier score** — the standard proper scoring rule for probabilistic
  forecasts (mean squared error between the predicted probability and
  the actual 0/1 outcome; lower is better, 0 is perfect, 0.25 is what
  you'd get always guessing 50%) — computed for **both** the model's own
  predictions **and** Kalshi's own quoted price at the same moments. That
  second number is a real baseline, not a strawman: prediction markets
  with real money on the line are usually reasonably well-calibrated
  already, so "does the model's Brier score beat the market's own" is a
  meaningful bar to clear, not an easy one.

Each contract can rack up many logged rows (one per dashboard refresh
while it was open); the report scores one row per contract, so a contract
you watched closely doesn't get counted many times over. That row is the
one logged nearest `KALSHI_CALIBRATION_HORIZON_MINUTES` (5) before close,
within `KALSHI_CALIBRATION_HORIZON_TOLERANCE_MINUTES` (1) -- every contract
is judged at the same distance from settlement. (The closest-to-close row
would be nearly decided already, with P(YES) near 0 or 1, which says little
about the model.) The report also scores the logged point-in-time
probability on the same rows, so you can see whether modelling the
settlement average actually helped.

**Sample size matters here more than almost anywhere else in this repo.**
A handful of resolved contracts is nowhere near enough to distinguish a
real calibration problem from ordinary variance — buckets below
`KALSHI_CALIBRATION_MIN_SAMPLE` (5, by default) are flagged as low-sample
in the output. Treat any of this as informative once you're well into the
dozens of resolved contracts per bucket, not before.

## Tuning tips

- `GRANULARITY` + refresh interval: signals are scored on **completed**
  candles only -- the same kind of candle the backtest scored -- so with
  1-minute candles they change once a minute, however often you refresh.
  The price in the header (and the Kalshi model) still uses the latest
  print from the candle that's still forming. Coinbase asks that you not
  poll the candles endpoint too aggressively; every 15–30s is reasonable
  for 1-minute candles.
- Alert threshold: lower it (e.g. 2) for more frequent, noisier alerts;
  raise it (e.g. 5) to only see strong alignment.
- `IMBALANCE_THRESHOLD` and `ORDER_BOOK_DEPTH` (in `config.py`): these now
  only apply to the fallback path, when `ENABLE_BOOK_FEEDS = False` or the
  depth feeds haven't connected. They're on a different (unweighted base
  volume) scale from `BOOK_IMBALANCE_THRESHOLD` and are not interchangeable
  with it.
- `BOOK_PERSISTENCE_SECONDS` (in `config.py`): the single most useful knob
  on the order-book signal. Raise it for fewer, more considered order-book
  votes; lower it to react faster at the cost of more noise; set it to `0`
  to vote on the instantaneous reading the way the old signal did.
- `BOOK_BAND_BPS` and `BOOK_DECAY_BPS` (in `config.py`): the band is how
  far from mid resting size is counted at all, the decay is how sharply
  closer levels are favoured within it. A tighter band (or smaller decay)
  makes the signal more about the executable top of book; a wider one
  brings in more passive, more easily faked size.
- `BOOK_FLOW_WINDOW_SECONDS` and `BOOK_FLOW_THRESHOLD` (in `config.py`): a
  shorter window catches liquidity being pulled faster but picks up more
  churn. Note the signal is simply absent until the feeds have that much
  history.
- `BOOK_WALL_MULTIPLE`, `BOOK_WALL_MIN_NOTIONAL`, `BOOK_WALL_MIN_AGE_SECONDS`
  (in `config.py`): how big relative to its neighbours, how big in dollars,
  and how long present a level has to be before it's called a wall. Raise
  the age if you're seeing walls reported that vanish as soon as price
  approaches.
- `BACKTEST_HISTORY_CANDLES` (in `config.py`): more candles = larger
  samples per bucket but a slower backtest run. 4000 (a few days of 1-min
  candles) is a reasonable default; drop it if the backtest feels too slow.
- `AGG_WINDOW_SECONDS` and `AGG_DELTA_THRESHOLD` (in `config.py`): a
  shorter window reacts faster but is noisier; a wider one smooths things
  out but lags. `WHALE_TRADE_USD_THRESHOLD` controls how big a single
  trade needs to be before it counts as a whale alert -- lower it if you
  want to see smaller trades, or raise it on quieter pairs where $100k
  trades are rare.
- `INDICATOR_WEIGHT_SCALE`, `INDICATOR_WEIGHT_MIN`, `INDICATOR_WEIGHT_MAX`
  (in `config.py`): a higher scale makes the weighted score react more
  sharply to an indicator's historical edge; the min/max floor and cap how
  far any single indicator's weight can swing. Widen them if the weighted
  and raw reads almost never diverge and you want more separation between
  them; narrow them if the weighted score feels too volatile between
  backtest runs.
- `AGG_DELTA_BACKTEST_HOURS` (in `config.py`): more hours means more
  Binance.US trade history to pull (more requests, slower run) but a
  larger sample. 24h is a reasonable default; `AGG_DELTA_BACKTEST_MAX_REQUESTS`
  is a safety cap in case an unusually active window needs more requests
  than expected to cover.

## Notes and caveats

- This reflects short-term technical signals, not fundamentals or news —
  treat it as one input, not a trading decision by itself.
- Coinbase's public candles endpoint isn't meant for real-time tick data;
  for genuinely low-latency use you'd want the WebSocket feed instead of
  polling REST. This script polls REST for simplicity.
- If you want to extend this toward your Polymarket 5-minute bot later, the
  order-book imbalance piece in `orderbook.py` uses the same core idea as
  what you scoped out for that project — it'd be a straightforward reuse,
  and `orderbook_feeds.py` is venue-agnostic enough to point at another
  exchange without much surgery.
- **The order-book signals are better-built, not proven.** The persistence
  gate, notional weighting and cross-venue aggregation are all arguments
  from how order books actually work, not measurements — there's no
  historical level-2 data to check any of it against. Treat them as a
  cleaner description of the book than the old signal gave, not as an
  established edge.

## Running the tests

```bash
pip install pytest
pytest
```

The suite covers the parts that have already caused real bugs once, or are
easy to get quietly wrong:

- **Order-book math** — the banded, distance-decayed depth sums are
  cross-validated against a deliberately naive brute-force implementation
  across randomized books, so the sorted-order early exit can't silently
  drop levels. Microprice, spread and wall detection are checked against
  hand-computed values.
- **Exchange payload parsers** — every one is tested against that
  exchange's *own documented example payload*, copied verbatim, never a
  field name guessed from memory.
- **Binance sequence handling** — the applied/stale/gap/re-seed logic is
  exercised directly, since a mishandled gap produces a book that looks
  fine and is quietly wrong.
- **CSV log schema** — including the exact scenario that broke
  `pd.read_csv()` in production once: a signal that only appears partway
  through a session.
