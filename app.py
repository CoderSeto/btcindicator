"""
BTC Indicator Bot -- Dashboard
--------------------------------
A browser-based, auto-refreshing UI on top of the same data/indicator/
signal pipeline used by main.py (the console version). Run with:

    streamlit run app.py

This is a monitoring/alerting tool -- it does NOT place any real trades. The
Paper trading tab simulates Kalshi orders only (see paper_trading.py).
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import backtest
import config as shared_config
import dashboard_ui as ui
import edge_recorder
import exchange_feeds
import kalshi
import kalshi_api
import kalshi_decision
import kalshi_tracking
import orderbook_feeds
import paper_trading
from dashboard_data import load_state
from main import log_alert, log_signal_snapshot

st.set_page_config(page_title="BTC Indicator Bot", page_icon="\u20bf", layout="wide")

# A private copy of config.py's settings for this run. Streamlit runs every
# browser session's script in the same process, so writing the sidebar values
# into the shared `config` module (as this page once did) made a new tab open
# on the previous tab's pair and threshold, and let two live tabs render or log
# one pair's data under the other's label mid-run. Everything below reads and
# writes `cfg`; the shared module is only ever read, here.
cfg = SimpleNamespace(**{k: getattr(shared_config, k) for k in dir(shared_config) if k.isupper()})

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BTC-EUR"]


# --- Sidebar controls ---
st.sidebar.header("Settings")
product_id = st.sidebar.selectbox(
    "Pair", PAIRS, index=PAIRS.index(cfg.PRODUCT_ID) if cfg.PRODUCT_ID in PAIRS else 0,
)
# Floor lowered to 5s (from 10s) to support watching a Kalshi window's
# closing minute more closely -- the "minutes remaining" default in the
# Kalshi section recomputes on every refresh, so a shorter interval gives
# a more live-feeling countdown. Still gentle on Coinbase's REST API.
refresh_seconds = st.sidebar.slider("Refresh interval (s)", 5, 120, cfg.POLL_INTERVAL, 5)
# Upper bound is the full signal set: 6 candle-based + orderbook + book_flow + agg_delta.
alert_threshold = st.sidebar.slider("Alert threshold (indicators aligned)", 1, 9, cfg.COMPOSITE_ALERT_THRESHOLD)
st.sidebar.button("Refresh now", width="stretch")
st.sidebar.caption("Changes here apply immediately. Edit config.py to change other defaults "
                    "(indicator periods, order book depth, etc).")

# This session's copy only (see the top of the file). Everything this page
# calls is handed `cfg` explicitly; main.py's loggers read only the shared log
# paths, which the sidebar never changes.
cfg.PRODUCT_ID = product_id
cfg.POLL_INTERVAL = refresh_seconds
cfg.COMPOSITE_ALERT_THRESHOLD = alert_threshold


@st.cache_resource(show_spinner=False)
def _get_feed_handle(pair: str, enabled: bool):
    # Cached per (pair, enabled) so switching pairs starts a fresh set of
    # feeds. Note: previously-selected pairs' background connections keep
    # running for the life of the app process (Streamlit's resource cache
    # doesn't evict them) -- a minor resource cost, not a correctness issue,
    # but worth knowing if you switch pairs repeatedly in one long session.
    if not enabled:
        return None
    return exchange_feeds.start_feeds(cfg)


@st.cache_resource(show_spinner=False)
def _get_book_handle(pair: str, enabled: bool):
    # Same caching story as the trade feeds above. The order-book feeds are
    # more expensive to warm up (Coinbase sends the whole book on connect,
    # Binance needs a REST snapshot spliced onto the stream), so they're
    # deliberately started once and kept for the life of the process rather
    # than rebuilt per rerun.
    if not enabled:
        return None
    return orderbook_feeds.start_book_feeds(cfg)


feed_handle = _get_feed_handle(product_id, cfg.ENABLE_EXCHANGE_FEEDS)
feed_snapshot = feed_handle.snapshot() if feed_handle else None

book_handle = _get_book_handle(product_id, cfg.ENABLE_BOOK_FEEDS)
book_snapshot = book_handle.snapshot() if book_handle else None

@st.cache_resource(show_spinner=False)
def _get_recorder(enabled: bool, path: str):
    """
    The background recorder (edge_recorder.start_background), once per process, on BTC-USD (the
    KXBTC15M underlying) and only while the dashboard is on BTC-USD, so the feeds it reads are the
    BTC ones. It shares this dashboard's Kalshi poll, ticker and feeds; see edge_recorder.
    """
    if not enabled:
        return None
    rec_cfg = SimpleNamespace(**vars(cfg))
    rec_cfg.PRODUCT_ID = "BTC-USD"
    return edge_recorder.start_background(
        rec_cfg, fetch_quote=lambda: kalshi_api.live_market(rec_cfg),
        fetch_spot=lambda: ui.shared_latest_price("BTC-USD", rec_cfg.STRIP_REFRESH_SECONDS / 2),
        book_snapshot=book_handle.snapshot if book_handle else None,
        trade_snapshot=feed_handle.snapshot if feed_handle else None)


recorder = _get_recorder(bool(cfg.EDGE_RECORD_IN_DASHBOARD) and product_id == "BTC-USD", cfg.EDGE_RECORD_PATH)

# --- Sidebar: backtest controls ---
st.sidebar.divider()
if recorder is not None:
    st.sidebar.caption(f"\u23FA Recording to {cfg.EDGE_RECORD_PATH} while this dashboard runs "
                       f"({recorder[0].rows_written:,} rows this session, {recorder[0].gaps_written} gaps).")
st.sidebar.divider()
st.sidebar.subheader("Backtest confidence")
cached_results = backtest.load_results(cfg)
if cached_results and "error" not in cached_results:
    gen_time = cached_results.get("generated_at", "")[:16].replace("T", " ")
    st.sidebar.caption(
        f"Last backtest: {cached_results['candles_used']} candles of "
        f"{cached_results['product_id']}, generated {gen_time} UTC"
    )
    if "base_rates" not in cached_results:
        st.sidebar.caption(
            "\u26A0\uFE0F These results predate overlap-corrected sample sizes and base rates -- "
            "re-run the backtest so the hit rates and weights are measured honestly."
        )
elif cached_results and "error" in cached_results:
    st.sidebar.caption(f"Last backtest attempt failed: {cached_results['error']}")
else:
    st.sidebar.caption(f"No backtest yet for {product_id} at this granularity.")

run_backtest_clicked = st.sidebar.button("Run backtest now", width="stretch")
st.sidebar.caption(
    f"Pulls ~{cfg.BACKTEST_HISTORY_CANDLES} historical candles and checks how often each "
    "signal pattern actually preceded a move. Takes 10-30s. Can also run standalone: "
    "`python backtest.py`."
)

if run_backtest_clicked:
    with st.spinner("Running backtest -- pulling historical data and checking hit rates..."):
        try:
            backtest.run_backtest(cfg)
        except Exception as e:
            st.sidebar.error(f"Backtest failed: {e}")
        else:
            st.rerun()

st_autorefresh(interval=refresh_seconds * 1000, key="autorefresh")

st.markdown(ui.CSS, unsafe_allow_html=True)

# --- Load data ---
# The page refresh reads the Kalshi market (through the shared poll) alongside load_state's Coinbase
# fetches. The live strip below re-reads price and quote on its own, faster timer.
auto_track = st.session_state.get("kalshi_auto_track", True)
kalshi_pair_ok = product_id == ui.KALSHI_PAIR
with ThreadPoolExecutor(max_workers=1) as _pool:
    _kalshi_poll = _pool.submit(kalshi_api.live_market, cfg) if auto_track and kalshi_pair_ok else None
    state = load_state(cfg, feed_snapshot=feed_snapshot, book_snapshot=book_snapshot)

if state.error:
    st.error(f"Couldn't fetch live data: {state.error}")
    st.stop()

df, imbalance, snapshot = state.df, state.imbalance, state.snapshot
# Signals are scored on completed candles (`df`); the price and the Kalshi model use the ticker's
# last trade. Seeding the shared ticker cache means the strip doesn't ask again this run.
live_price, price_df = state.live_price, state.price_df
ui.seed_price(product_id, {"price": live_price, "source": state.live_price_source,
                           "time": None, "reason": state.live_price_note})
log_signal_snapshot(snapshot)
alert_fired = snapshot.strength >= cfg.COMPOSITE_ALERT_THRESHOLD
if alert_fired:
    log_alert(snapshot)

# --- Kalshi inputs ------------------------------------------------------------
# KXBTC15M is a BTC/USD contract, so auto-tracking only runs on BTC-USD. The strike resets only
# when the window changes (a manual edit mid-window survives); minutes left always follows the clock.
kalshi_live = _kalshi_poll.result() if _kalshi_poll is not None else None
if kalshi_live is not None and kalshi_live["available"]:
    if st.session_state.get("kalshi_last_ticker") != kalshi_live["ticker"]:
        st.session_state["kalshi_strike"] = kalshi_live["strike"]
        st.session_state["kalshi_last_ticker"] = kalshi_live["ticker"]
    st.session_state["kalshi_minutes"] = round(kalshi_live["minutes_remaining"], 2)
if "kalshi_strike" not in st.session_state:
    st.session_state["kalshi_strike"] = float(round(live_price))
if "kalshi_minutes" not in st.session_state:
    st.session_state["kalshi_minutes"] = round(kalshi.minutes_until_next_kalshi_close(), 1)

kalshi_result = kalshi.fair_value(
    price_df, st.session_state["kalshi_strike"], st.session_state["kalshi_minutes"],
    cfg.KALSHI_VOL_LOOKBACK_MINUTES, candle_seconds=cfg.GRANULARITY, current_price=live_price,
    max_strike_distance=cfg.KALSHI_MAX_STRIKE_DISTANCE,
)
# Only log a row that prices the tracked market's own strike (a hand-edited strike answers a
# different question from the one the ticker settles).
if (kalshi_result["available"] and kalshi_live and kalshi_live["available"]
        and kalshi_result["strike"] == kalshi_live["strike"]):
    kalshi_tracking.log_prediction(
        cfg, ticker=kalshi_live["ticker"], strike=kalshi_result["strike"],
        current_price=kalshi_result["current_price"], minutes_remaining=kalshi_result["minutes_remaining"],
        p_yes=kalshi_result["p_yes"], annual_vol=kalshi_result["annual_vol"],
        kalshi_yes_bid=kalshi_live["yes_bid"], kalshi_yes_ask=kalshi_live["yes_ask"],
        kalshi_last_price=kalshi_live.get("last_price"), p_yes_point=kalshi_result["p_yes_point"],
    )

decision_rule, decision_result = kalshi_decision.load_active(cfg.DECISION_BACKTEST_PATH)

# --- 1. Live strip: price, quote, P(YES), time left, decision (own timer) --------
ui.header_line(product_id, refresh_seconds)
strip_ctx = {
    "product_id": product_id, "candles": df, "price_df": price_df, "auto_track": auto_track and kalshi_pair_ok,
    "strike": st.session_state["kalshi_strike"], "minutes": st.session_state["kalshi_minutes"],
    "rule": decision_rule, "rule_result": decision_result,
    "open_trades": [t for t in paper_trading.load_trades(cfg.PAPER_TRADE_LOG_PATH) if t["status"] == "open"],
}
with st.container(border=True):
    st.fragment(ui.live_strip, run_every=cfg.STRIP_REFRESH_SECONDS)(cfg, strip_ctx)

# --- 2. Composite, order book, signal badges ----------------------------------
with st.container(border=True):
    ui.render_signal_row(cfg, snapshot, imbalance, snapshot.book or {}, alert_fired)

# --- 3. Kalshi inputs | Outlook -------------------------------------------------
left, right = st.columns(2)
with left:
    ui.render_kalshi_box(cfg, product_id, auto_track, kalshi_live, kalshi_result, book_snapshot)
with right:
    ui.render_outlook(snapshot, state.outlook)

# --- 4. Deep-dive tabs ------------------------------------------------------------
# Order Book leads: Streamlit opens the first tab, and landing on the tall chart would scroll the page.
tab_book, tab_feeds, tab_chart, tab_alerts, tab_paper, tab_notes = st.tabs(
    ["📒 Order Book", "📡 Feeds", "📈 Chart",
     "🔔 Alerts", "🧪 Paper trading", "ℹ️ Notes & limits"]
)
with tab_chart:
    ui.render_chart_tab(cfg, df, product_id)
with tab_book, st.container(height=ui.TAB_BODY_HEIGHT, border=False):
    ui.render_book_tab(cfg, book_snapshot)
with tab_feeds, st.container(height=ui.TAB_BODY_HEIGHT, border=False):
    ui.render_feeds_tab(cfg, feed_snapshot)
with tab_alerts, st.container(height=ui.TAB_BODY_HEIGHT, border=False):
    ui.render_alerts_tab(cfg)
with tab_paper, st.container(height=ui.TAB_BODY_HEIGHT, border=False):
    ui.render_paper_tab(cfg, kalshi_live, kalshi_result, snapshot, live_price)
with tab_notes, st.container(height=ui.TAB_BODY_HEIGHT, border=False):
    ui.render_notes_tab(cfg, state, kalshi_result, book_snapshot)
