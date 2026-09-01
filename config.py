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
# (hit_rate - 0.5): a coin-flip (50%) indicator gets weight 1.0 (same as
# unweighted); above/below 50% scales up/down from there.
INDICATOR_WEIGHT_SCALE = 2.0
INDICATOR_WEIGHT_MIN = 0.2  # floor -- a bad-track-record indicator still counts a little
INDICATOR_WEIGHT_MAX = 2.0  # cap -- one strong indicator can't fully dominate

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
