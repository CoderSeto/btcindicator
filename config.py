"""
Configuration for the BTC indicator bot.
Edit these values to change behavior without touching the core logic.
"""

# Coinbase Exchange product to track (BTC-USD, BTC-EUR, ETH-USD, etc.)
PRODUCT_ID = "BTC-USD"

# Candle granularity in seconds. Coinbase supports: 60, 300, 900, 3600, 21600, 86400
GRANULARITY = 60  # 1-minute candles

# How many candles to keep in the rolling window for indicator calculations
CANDLE_HISTORY = 200

# How often (seconds) the bot polls for new data
POLL_INTERVAL = 30

# Order book depth (levels) used for the bid/ask imbalance signal
ORDER_BOOK_DEPTH = 25

# --- Indicator parameters ---
SMA_FAST = 9
SMA_SLOW = 21
EMA_FAST = 12
EMA_SLOW = 26

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

VOLUME_SPIKE_WINDOW = 20
VOLUME_SPIKE_MULTIPLIER = 2.0

IMBALANCE_THRESHOLD = 0.20  # +/-20% skew between bid/ask volume triggers a signal

# --- Multi-exchange order book (see orderbook.py / orderbook_feeds.py) ---
# Master switch for the live level-2 depth WebSockets (Coinbase, Binance.US,
# Bybit). With this False the bot falls back to the single Coinbase REST
# book snapshot it polls each cycle -- still a working order-book signal,
# just one venue, one snapshot per POLL_INTERVAL, and with no persistence
# or flow history behind it.
ENABLE_BOOK_FEEDS = True

# Depth bands, in basis points from mid (1 bp = 0.01%). Book depth is
# measured by distance from mid rather than by level count because a
# "level" means something different on every exchange -- Coinbase's
# BTC-USD book is quoted in $0.01 steps, Bybit's spot book is coarser, so
# "top 25 levels" covers a totally different slice of each. Basis points
# are comparable across venues; level counts are not.
BOOK_BAND_BPS_NEAR = 5.0    # the executable book -- what a market order hits now
BOOK_BAND_BPS = 25.0        # the working band the directional signal is read from
BOOK_BAND_BPS_FAR = 100.0   # context: does the wider book agree with the near one?

# Within a band, each level is weighted exp(-distance_bps / BOOK_DECAY_BPS),
# so resting size at the touch counts for much more than size at the band
# edge. Set to 0 to weight everything inside a band equally.
BOOK_DECAY_BPS = 10.0

# A venue whose spread exceeds this is dropped from the aggregate for that
# refresh -- a blown-out spread usually means a stale or dislocated feed
# rather than real information.
BOOK_MAX_SPREAD_BPS = 8.0

# Depth subscribed per venue. Binance's partial-book depth stream supports
# only 5, 10 or 20; Bybit spot supports 1, 50, 200, 1000. Coinbase's
# level2_batch channel sends the whole book and isn't configurable.
BINANCE_BOOK_DEPTH = 20
BYBIT_BOOK_DEPTH = 200

# How often the aggregated book state is sampled into the rolling history
# that persistence and flow are computed from, and how much of that
# history to keep.
BOOK_SAMPLE_INTERVAL_SECONDS = 1.0
# A venue whose book hasn't changed for this long is left out of the sample.
# Coinbase's level2_batch pushes every 50ms and the others push every change,
# so 15s of silence on a BTC book is a frozen feed, not a quiet market. The
# socket's ping timeout (~30s) used to be the only thing that removed one.
BOOK_STALE_SECONDS = 15.0
BOOK_HISTORY_SECONDS = 900

# How much of each local book is kept when it's materialized for the
# metrics above. Coinbase's level2 feed carries the entire book (tens of
# thousands of levels), almost none of which is relevant on this bot's
# timescale; trimming to a wide guard band keeps the per-sample work small
# while still comfortably covering BOOK_BAND_BPS_FAR and any plausible
# Kalshi strike distance (500 bps = 5%).
BOOK_RETAIN_BPS = 500.0

# Imbalance past this (in either direction) counts as a directional read.
# Separate from IMBALANCE_THRESHOLD above, which applies to the legacy
# single-snapshot fallback and is on a different (unweighted base-volume)
# scale -- the two numbers are not interchangeable.
BOOK_IMBALANCE_THRESHOLD = 0.18

# ...but it only becomes a directional VOTE once it has held that side
# continuously for this long. A single snapshot of a lopsided book is
# mostly noise: resting size appears and vanishes constantly, and the
# whole point of having a time series is to be able to tell a lean that
# has stuck around from one that flickered. Set to 0 to vote on the
# instantaneous reading (the old behavior).
BOOK_PERSISTENCE_SECONDS = 20.0

# --- Resting-liquidity flow (the book_flow signal) ---
# Compares how each side's resting notional has grown or shrunk over this
# window: bids stacking while asks get pulled is the classic setup ahead
# of an upward move, and vice versa.
BOOK_FLOW_WINDOW_SECONDS = 60.0
# log(bid growth) - log(ask growth) past this magnitude counts as a signal.
# 0.20 is roughly "one side's resting notional grew ~22% more than the other's".
BOOK_FLOW_THRESHOLD = 0.20
# Adds book_flow as its own vote in the composite. Turn this off to keep
# the composite exactly as it was and treat flow as display-only context.
# Like the order-book imbalance signal, it cannot be backtested (see
# backtest.py) -- so it always carries weight 1.0 in weighted conviction.
ENABLE_BOOK_FLOW_SIGNAL = True

# --- Liquidity walls ---
# A price level counts as a wall when its resting notional is at least
# BOOK_WALL_MULTIPLE times the median level inside the band AND at least
# BOOK_WALL_MIN_NOTIONAL in absolute terms.
BOOK_WALL_MULTIPLE = 4.0
BOOK_WALL_MIN_NOTIONAL = 250_000
# A wall is only reported as "held" once it has been continuously present
# for this long. A wall that appears and vanishes as price approaches is
# indistinguishable from a spoof, and age is the only evidence available.
BOOK_WALL_MIN_AGE_SECONDS = 15.0
# How close two sightings must be (in bps) to count as the same wall
# across samples, so a wall isn't "reset" by a one-tick reprice.
BOOK_WALL_MATCH_BPS = 2.0

# Minimum number of aligned individual signals required to fire a composite alert
COMPOSITE_ALERT_THRESHOLD = 3

# --- Divergence / reversal detection ---
# How many recent candles to scan for a price/RSI divergence pattern
DIVERGENCE_LOOKBACK = 40
# A candle counts as a local swing high/low if it's the extreme point
# within +/- this many candles on either side
DIVERGENCE_SWING_ORDER = 3

# --- Backtesting (historical confidence) ---
# How many historical candles to pull for the backtest. Coinbase caps each
# request at 300, so this gets paginated into multiple requests.
BACKTEST_HISTORY_CANDLES = 4000
# "Did price turn" is checked this many candles after each historical point
# (a short lookahead, per your choice of short-term/noisier turns)
BACKTEST_LOOKAHEAD_CANDLES = 5
# Below this many historical occurrences, a hit rate is flagged as low-sample
BACKTEST_MIN_SAMPLE = 15
# Pause (seconds) between paginated historical requests, to be polite to Coinbase
BACKTEST_REQUEST_PAUSE = 0.3
# Where backtest results are cached between runs
BACKTEST_RESULTS_PATH = "backtest_results.json"

# CSV file alerts get appended to
ALERT_LOG_PATH = "alerts_log.csv"

# Minimum seconds between two rows in the alert and signal logs. The dashboard
# reruns its whole script on every refresh tick, every widget click and in
# every open tab, and the console bot writes the same files; without this each
# rerun appended a row. A *different* alert (direction or strength changed) is
# always logged straight away; only repeats are held back. Checked against the
# file's own last row, so it also holds across processes. 0 disables it.
LOG_MIN_INTERVAL_SECONDS = 20

# --- Multi-exchange live feeds (aggregated volume, whale trades, liquidations) ---
# Master switch -- set False to run on Coinbase REST data alone, no background
# WebSocket connections.
ENABLE_EXCHANGE_FEEDS = True

# Rolling window (seconds) used to compute the aggregated buy/sell delta
AGG_WINDOW_SECONDS = 60
# +/- skew in that window needed to call the aggregated delta bullish/bearish
AGG_DELTA_THRESHOLD = 0.15

# A single trade at or above this notional value (USD) counts as a "whale" trade
WHALE_TRADE_USD_THRESHOLD = 100_000
# How many recent whale trades / liquidations to keep in memory for display
WHALE_LIST_MAXLEN = 50
LIQUIDATION_LIST_MAXLEN = 50

# --- Indicator weighting (weighted conviction score) ---
# Turns each indicator's backtested hit rate into a weight for a SEPARATE
# "weighted conviction" score, alongside (not replacing) the raw indicator
# count used for the composite/alert threshold. Weight = 1.0 + SCALE *
# (hit_rate - base_rate): a call that does no better than the window's own
# drift gets weight 1.0 (same as unweighted); better/worse scales up/down.
INDICATOR_WEIGHT_SCALE = 2.0
INDICATOR_WEIGHT_MIN = 0.2  # floor -- a bad-track-record indicator still counts a little
INDICATOR_WEIGHT_MAX = 2.0  # cap -- one strong indicator can't fully dominate
# The edge is measured against the backtested window's base rate for that
# direction (not 50%), on the overlap-corrected effective sample size, and a
# weight only moves off 1.0 when that edge is at least this many standard
# errors from zero. Below it the edge is indistinguishable from sampling noise
# -- on a pure random walk, ungated weights still spread +/-0.08 to +/-0.16.
INDICATOR_WEIGHT_MIN_Z = 2.0

# --- Signal logging (for correlation analysis -- see analyze_correlation.py) ---
# When True, logs EVERY cycle's full set of indicator votes to
# SIGNAL_LOG_PATH -- not just alert-triggering moments like ALERT_LOG_PATH
# does. Off by default since it's a continuously-growing file; turn on
# when you want to accumulate data to analyze indicator overlap/correlation.
LOG_ALL_SIGNALS = False
SIGNAL_LOG_PATH = "signal_log.csv"

# --- agg_delta historical backtest (Binance.US only -- see backtest_agg_delta.py) ---
# How far back to reconstruct historical agg_delta readings from Binance.US's
# aggTrades REST history. Binance.US is the only one of the three live
# exchanges with a practical time-range historical trades endpoint (Coinbase's
# is cursor/trade-ID paginated, not time-range; Bybit's caps at the last 60
# trades) -- so this backtest reflects Binance.US alone, not the full
# 3-exchange live signal. Still real historical grounding, just not a perfect
# reconstruction of what the live signal would have shown.
AGG_DELTA_BACKTEST_HOURS = 24
AGG_DELTA_BACKTEST_REQUEST_PAUSE = 0.3
AGG_DELTA_BACKTEST_MAX_REQUESTS = 300

# --- Kalshi 15m BTC fair-value calculator (see kalshi.py) ---
# How many minutes of recent 1-minute candle history to use for the
# realized-volatility estimate that feeds the fair-value calculation.
# Longer = more stable estimate but slower to react to a recent vol
# regime change; shorter = more responsive but noisier. 120 (2 hours) is
# a reasonable middle ground for a contract this short-dated.
KALSHI_VOL_LOOKBACK_MINUTES = 120
# The dashboard asks Kalshi for the open KXBTC15M market at most once per
# this many seconds, shared across every open tab (the countdown is still
# recomputed on every refresh). Kalshi has answered HTTP 429 to per-refresh
# polling before. 0 polls on every refresh.
KALSHI_POLL_SECONDS = 10
# Refuse to price a strike further than this fraction from spot (0.05 = 5%).
# BTC doesn't move 5% inside one 15-minute window, so a strike that far off is
# almost certainly a bad record: Kalshi's own API has served two KXBTC15M
# strikes divided by 10,000 (7.2437 for 72,437.29), which priced at
# P(YES) = 100%. See REVIEW_FINDINGS.md, "Is there an edge?".
KALSHI_MAX_STRIKE_DISTANCE = 0.05

# --- Kalshi prediction tracking / calibration (see kalshi_tracking.py) ---
# When True, every fair-value computation (each dashboard refresh, while
# a Kalshi market is being auto-tracked) is logged to
# KALSHI_PREDICTIONS_LOG_PATH. Off by default since it's a continuously-
# growing file -- turn on to start building a track record you can check
# for calibration: do "82% predictions" actually resolve YES about 82%
# of the time, across many contracts?
LOG_KALSHI_PREDICTIONS = False
KALSHI_PREDICTIONS_LOG_PATH = "kalshi_predictions.csv"
# Cache of resolved outcomes (ticker -> yes/no), so re-running the
# calibration report doesn't re-query Kalshi for contracts already known
# to be settled.
KALSHI_OUTCOMES_CACHE_PATH = "kalshi_outcomes_cache.json"
# Width of each probability bucket in the calibration report (0.2 = five
# buckets: 0-20%, 20-40%, ..., 80-100%). Narrower buckets are more
# precise but need more resolved contracts per bucket to mean anything.
KALSHI_CALIBRATION_BUCKET_SIZE = 0.2
# Below this many resolved predictions in a bucket, its calibration rate
# is flagged as low-sample rather than trusted at face value.
KALSHI_CALIBRATION_MIN_SAMPLE = 5
# Which logged row per contract the calibration report scores: the one
# nearest this many minutes before close, if one lies within the tolerance
# (contracts with none are left out). A fixed horizon compares like with like.
# The closest-to-close row (the old choice) is nearly decided by then -- p near
# 0 or 1 -- so it measured the model where it matters least.
KALSHI_CALIBRATION_HORIZON_MINUTES = 5.0
KALSHI_CALIBRATION_HORIZON_TOLERANCE_MINUTES = 1.0

# --- Paper trading on KXBTC15M (see paper_trading.py) ---
# Where simulated orders and their settlements are kept: an append-only CSV of
# events, written only through csv_utils.append_row.
PAPER_TRADE_LOG_PATH = "paper_trades.csv"
# Below this many settled MARKETS the account summary is flagged as too small
# to tell skill from luck (the same idea as BACKTEST_MIN_SAMPLE).
PAPER_MIN_SAMPLE = 30
# The win rate has to beat the break-even rate by this many standard errors
# before the summary calls it anything but noise.
PAPER_NOISE_Z = 2.0
# Automatic mode: on every dashboard refresh inside the window below, enter ONE
# position per market if the rule says to trade. It runs from the dashboard, so
# it only trades while a dashboard page is open (and refreshing). This is only
# the checkbox's starting value; the Paper trading tab can switch it per session.
PAPER_AUTO_ENABLED = False
# "composite": follow the bot's composite signal when it is strong enough to
# alert (the sidebar's alert threshold): bullish -> YES, bearish -> NO.
# "model": trade toward the Kalshi fair-value model when its edge over the
# price, after the fee, exceeds PAPER_AUTO_MODEL_MARGIN -- the rule the edge
# study tested and found loses money. Neither is a recommendation.
PAPER_AUTO_STRATEGY = "composite"
# Enter from this many minutes before close (the first refresh at or below it
# where the rule fires), but not later than PAPER_AUTO_LAST_MINUTES before
# close. Keep the dashboard's refresh interval well under the gap between them.
PAPER_AUTO_ENTRY_MINUTES = 5.0
PAPER_AUTO_LAST_MINUTES = 2.0
PAPER_AUTO_CONTRACTS = 1
PAPER_AUTO_MODEL_MARGIN = 0.02

# ---------------------------------------------------------------------------
# Forward edge recorder (edge_recorder.py; EDGE_HYPOTHESES.md). Records the
# live KXBTC15M quote next to spot, depth and flow once a second so the
# pre-registered hypotheses can be tested on data none of them has seen.
# Standalone script, no auth; off unless you run it.
# ---------------------------------------------------------------------------
EDGE_RECORD_PATH = "edge_record.csv"          # one row per tick (and one per gap)
EDGE_TRADES_PATH = "edge_trades.csv"          # Kalshi prints, one row each, deduplicated by trade_id
EDGE_TICK_SECONDS = 1.0                       # tick interval; Kalshi book is polled every tick
EDGE_MARKET_REFRESH_SECONDS = 30.0            # how often the open-market list is re-fetched
EDGE_TRADES_REFRESH_SECONDS = 5.0             # how often Kalshi prints are polled
EDGE_VOL_REFRESH_SECONDS = 60.0               # how often the Coinbase vol window is re-fetched
EDGE_GAP_SECONDS = 5.0                        # a tick-to-tick interval longer than this logs a gap row
EDGE_LIQ_WINDOW_SECONDS = 30.0                # liquidation notional summed over this trailing window
