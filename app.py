"""
BTC Indicator Bot -- Dashboard
--------------------------------
A browser-based, auto-refreshing UI on top of the same data/indicator/
signal pipeline used by main.py (the console version). Run with:

    streamlit run app.py

This is a monitoring/alerting tool -- it does NOT place any trades.
"""
import os
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import backtest
import data_fetch
import config as shared_config
import exchange_feeds
import kalshi
import kalshi_api
import kalshi_tracking
import orderbook
import orderbook_feeds
import weighting
from charts import build_figure
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


# --- Small rendering helpers ------------------------------------------------

def _esc(text: str) -> str:
    """
    Escapes dollar signs for Streamlit's markdown.

    Streamlit renders `$...$` as LaTeX math, so any caption carrying two
    dollar amounts on one line -- "$100,008.85 vs strike $100,000.00" --
    silently turns the text between them into italic gibberish rather than
    erroring. Every interpolated money string that reaches st.markdown or
    st.caption goes through this. Metric values are not markdown-rendered
    and don't need it.
    """
    return text.replace("$", r"\$")


def _venue_status(status: dict):
    """
    One compact row of connection dots. Both the depth feeds and the trade
    feeds report status the same way, so they share this rather than keeping
    two near-identical blocks that drift apart.
    """
    if not status:
        st.caption("Connecting...")
        return
    cols = st.columns(len(status))
    for col, (name, val) in zip(cols, status.items()):
        with col:
            icon = "\U0001F7E2" if val == "connected" else (
                "\U0001F7E1" if val in ("connecting", "reconnecting", "syncing book")
                else "\U0001F534")
            st.markdown(
                f"<div style='font-size:0.72rem;line-height:1.4;margin-bottom:0.3rem' "
                f"title='{name}: {val}'>{icon} <b>{name}</b>"
                f"<span style='opacity:0.5'> {val[:18]}</span></div>",
                unsafe_allow_html=True,
            )


def _strike_path_liquidity(book_snapshot, kalshi_result, cfg):
    """
    How much resting size sits between spot and the Kalshi strike, or None if
    that can't be answered (no depth feed, or a strike further out than the
    retained band).

    Deliberately kept as a plain factual readout rather than anything that
    feeds back into P(YES): this project's rule is that independently sourced
    numbers don't get blended without validated evidence for the blending
    method, and there is none here -- no historical order-book data exists to
    calibrate what a given amount of resting size does to a crossing
    probability.
    """
    if not (book_snapshot and book_snapshot.get("available") and kalshi_result.get("available")):
        return None

    venue_books = book_snapshot.get("books", {})
    # Prefer Coinbase: it's the venue the price and the fair-value model both
    # come from, and it's quoted in actual dollars rather than USDT.
    venue = "coinbase" if "coinbase" in venue_books else next(iter(venue_books), None)
    book = venue_books.get(venue) if venue else None
    if not book:
        return None

    spot = kalshi_result["current_price"]
    strike = kalshi_result["strike"]
    if abs(strike - spot) / max(spot, 1) > cfg.BOOK_RETAIN_BPS / 10_000.0:
        return None

    side = "ask" if strike > spot else "bid"
    low, high = min(spot, strike), max(spot, strike)
    return {
        "venue": venue,
        "side": side,
        "low": low,
        "high": high,
        "notional": orderbook.notional_between(book, spot, strike, side),
        "direction_word": "up through" if strike > spot else "down through",
        "settles": "YES" if strike > spot else "NO",
        "walls": [
            w for w in book_snapshot.get("walls", [])
            if w.get("held") and w.get("venue") == venue and w["side"] == side
            and low <= w["price"] <= high
        ],
    }


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

# --- Sidebar: backtest controls ---
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

# --- Compact layout styling -------------------------------------------------
# Streamlit's defaults are generous with vertical space: a ~6rem top pad,
# large metric type, wide gaps between blocks. On a dashboard whose whole job
# is "see the current state at a glance" that costs a screenful of scrolling.
# These rules tighten the spacing so the live state fits in one viewport. The
# price chart is the deliberate exception -- it lives in a tab, where scrolling
# is expected and fine.
#
# This hooks Streamlit's internal test IDs, which do change between releases.
# A selector that stops matching simply does nothing: the layout still works,
# it just gets roomier again. Nothing here is load-bearing.
st.markdown(
    """
    <style>
      /* Reclaim the default ~6rem top pad and the gaps between blocks. */
      .block-container {padding-top: 2.4rem; padding-bottom: 0.8rem;}
      [data-testid="stVerticalBlock"] {gap: 0.4rem;}
      [data-testid="stHorizontalBlock"] {gap: 0.5rem;}
      [data-testid="stVerticalBlockBorderWrapper"] {padding: 0.1rem 0.15rem;}
      /* Metrics carry the headline numbers; the default type is oversized
         for four of them in a row. */
      [data-testid="stMetricValue"] {font-size: 1.35rem; line-height: 1.5rem;}
      [data-testid="stMetricLabel"] p {font-size: 0.72rem;}
      [data-testid="stMetricDelta"] {font-size: 0.7rem;}
      [data-testid="stTabs"] button p {font-size: 0.8rem;}
      h1, h2, h3, h4 {margin-bottom: 0.3rem; padding-top: 0.1rem;}
      hr {margin: 0.35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load data ---
state = load_state(cfg, feed_snapshot=feed_snapshot, book_snapshot=book_snapshot)

if state.error:
    st.error(f"Couldn't fetch live data: {state.error}")
    st.stop()

df, imbalance, snapshot = state.df, state.imbalance, state.snapshot
# Signals are scored on completed candles (`df`); the header price and the
# Kalshi model use the ticker's last trade (a candle's close can lag it).
live_price, price_df = state.live_price, state.price_df
stale_price_warning = None
if state.live_price_source != "ticker":
    stale_price_warning = (f"\u26A0\uFE0F Price from the newest candle, not the ticker "
                           f"({state.live_price_note}). {data_fetch.CANDLE_CACHE_NOTE}.")
book_read = snapshot.book or {}
outlook_result = state.outlook
log_signal_snapshot(snapshot)
alert_fired = snapshot.strength >= cfg.COMPOSITE_ALERT_THRESHOLD
if alert_fired:
    log_alert(snapshot)

# --- Kalshi fair value, computed before anything renders --------------------
# The headline strip shows P(YES) alongside price and the composite, but the
# strike/minutes inputs live further down the page. Reading them from
# session_state here rather than from the widgets lets the number appear at
# the top while the controls stay next to the rest of the Kalshi block.
# session_state already holds the current values at the start of a rerun, so
# there's no one-frame lag: editing a strike reruns the script, and this
# reads the new value before the widget is drawn again.
auto_track = st.session_state.get("kalshi_auto_track", True)
# KXBTC15M is a BTC/USD contract. Auto-tracking it on any other pair would
# price a ~$75k BTC strike against that pair's candles (ETH, SOL, or BTC in
# EUR) and log the result under the BTC ticker, so it only runs on BTC-USD.
KALSHI_PAIR = "BTC-USD"
kalshi_pair_ok = product_id == KALSHI_PAIR

kalshi_live = None
if auto_track and kalshi_pair_ok:
    kalshi_live = kalshi_api.cached_btc15m_market(cfg.KALSHI_POLL_SECONDS)
    if kalshi_live["available"]:
        # Strike only resets when the window itself actually changes -- so a
        # manual edit mid-window survives instead of being fought every
        # refresh. Minutes-remaining is a live countdown, so it always updates.
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

# Only log a row that prices the tracked market's own strike. A hand-edited
# strike survives refreshes on purpose, but its P(YES) answers a different
# question from the one the ticker settles, and the calibration report would
# score it against that ticker's outcome anyway.
if (kalshi_result["available"] and kalshi_live and kalshi_live["available"]
        and kalshi_result["strike"] == kalshi_live["strike"]):
    kalshi_tracking.log_prediction(
        cfg,
        ticker=kalshi_live["ticker"],
        strike=kalshi_result["strike"],
        current_price=kalshi_result["current_price"],
        minutes_remaining=kalshi_result["minutes_remaining"],
        p_yes=kalshi_result["p_yes"],
        annual_vol=kalshi_result["annual_vol"],
        kalshi_yes_bid=kalshi_live["yes_bid"],
        kalshi_yes_ask=kalshi_live["yes_ask"],
        kalshi_last_price=kalshi_live["last_price"],
        p_yes_point=kalshi_result["p_yes_point"],
    )

# The study's answer to "is the model-vs-market gap an edge?", shown wherever the gap is.
MODEL_VS_MARKET_NOTE = (
    "Model minus market is not a trading signal. On 30,688 settled KXBTC15M contracts the model "
    "scored worse than Kalshi's own mid on held-out data, and trading toward it lost money after "
    "fees (REVIEW_FINDINGS.md, \"Is there an edge?\")."
)

kalshi_quote = None
if (kalshi_live and kalshi_live["available"]
        and kalshi_live["yes_bid"] is not None and kalshi_live["yes_ask"] is not None):
    kalshi_quote = (kalshi_live["yes_bid"] + kalshi_live["yes_ask"]) / 2

# --- Header strip -----------------------------------------------------------
title_col, alert_col = st.columns([3, 2])
with title_col:
    st.markdown(
        f"#### \u20bf {product_id}"
        f"<span style='font-weight:400;font-size:0.75rem;opacity:0.6'>"
        f" &nbsp;·&nbsp; {datetime.now().strftime('%H:%M:%S')}"
        f" &nbsp;·&nbsp; refresh {refresh_seconds}s</span>",
        unsafe_allow_html=True,
    )
with alert_col:
    if alert_fired:
        st.markdown(
            f"<div style='text-align:right;padding-top:0.5rem;font-size:0.85rem'>"
            f"\U0001F6A8 <b>{snapshot.strength} aligned {snapshot.composite.upper()}</b>"
            f"<span style='opacity:0.6'> &nbsp;· logged to {cfg.ALERT_LOG_PATH}</span></div>",
            unsafe_allow_html=True,
        )

emoji_map = {"bullish": "\U0001F7E2", "bearish": "\U0001F534", "neutral": "\u26aa"}

head_cols = st.columns(4)
with head_cols[0]:
    with st.container(border=True):
        st.metric(product_id, f"${live_price:,.2f}",
                  help="Last trade, from Coinbase's ticker. The signals below are scored on "
                       "completed candles only, like the backtest, so they can lag this by up "
                       "to one candle.")
        if stale_price_warning:
            st.caption(_esc(stale_price_warning))
with head_cols[1]:
    with st.container(border=True):
        st.metric(
            "Composite",
            f"{emoji_map[snapshot.composite]} {snapshot.composite.upper()}",
            f"{snapshot.strength}/{len(snapshot.individual)} aligned",
        )
with head_cols[2]:
    with st.container(border=True):
        if book_read.get("available"):
            held = book_read["persistence_seconds"]
            sub = (
                f"held {held:.0f}s \u00b7 {book_read['venue_count']} venues"
                if book_read["persistence_direction"]
                else f"{book_read['venue_count']} venues \u00b7 no held lean"
            )
            st.metric("Order book", f"{book_read['imbalance']:+.2f}", sub,
                      help="Notional depth within the configured band of mid, distance-weighted "
                           "and averaged across venues. Only votes once a lean has held.")
        elif imbalance is None:
            st.metric("Order book", "--", "waiting on depth feeds")
        else:
            sub = ("buy pressure" if imbalance > 0.02
                   else "sell pressure" if imbalance < -0.02 else "balanced")
            st.metric("Order book (1 venue)", f"{imbalance:+.2f}", sub,
                      help="Fallback reading: one Coinbase REST snapshot, unweighted base "
                           "volume over a fixed level count. Not the same scale as the "
                           "aggregated reading.")
with head_cols[3]:
    with st.container(border=True):
        if not kalshi_result["available"]:
            st.metric("Kalshi P(YES)", "--", kalshi_result["reason"][:38])
        else:
            gap = (f"{(kalshi_result['p_yes'] - kalshi_quote) * 100:+.0f} pts vs market"
                   if kalshi_quote is not None else
                   f"{kalshi_result['minutes_remaining']:.1f} min left")
            st.metric("Kalshi P(YES)", f"{kalshi_result['p_yes'] * 100:.1f}%", gap,
                      help="Model fair value for the strike below. Never blended with the "
                           "bot's directional signals -- see the Notes tab."
                           + (" " + MODEL_VS_MARKET_NOTE if kalshi_quote is not None else ""))

# --- Signal badges ----------------------------------------------------------
short_labels = {
    "sma_cross": "SMA", "ema_cross": "EMA", "rsi": "RSI", "macd": "MACD",
    "bollinger": "BOLL", "volume": "VOL", "orderbook": "BOOK",
    "book_flow": "FLOW", "agg_delta": "DELTA",
}
long_labels = {
    "sma_cross": "SMA Cross", "ema_cross": "EMA Cross", "rsi": "RSI",
    "macd": "MACD", "bollinger": "Bollinger", "volume": "Volume Spike",
    "orderbook": "Order Book", "book_flow": "Book Flow",
    "agg_delta": "Agg. Exchange Delta",
}

with st.container(border=True):
    badge_cols = st.columns(max(len(snapshot.individual), 1))
    for col, (name, val) in zip(badge_cols, snapshot.individual.items()):
        with col:
            st.markdown(
                f"<div style='text-align:center;line-height:1.25' "
                f"title='{long_labels.get(name, name)}: {val}'>"
                f"<span style='font-size:0.68rem;opacity:0.65'>{short_labels.get(name, name)}</span><br>"
                f"<span style='font-size:0.95rem'>{emoji_map[val]}</span></div>",
                unsafe_allow_html=True,
            )
    missing = [n for n in long_labels if n not in snapshot.individual]
    if missing:
        st.caption(
            "Warming up: " + ", ".join(long_labels[n] for n in missing)
            + " -- absent rather than voting neutral until their feeds have enough history."
        )

# The two mid-row boxes and the tab panel are all fixed-height scrollers, so
# the page height is a constant rather than a function of how much the outlook
# text happens to wrap or which tab is open. That is what makes "no page
# scroll" hold in every state instead of only the tidy ones. Raise either if
# you have the vertical room and would rather scroll the page than a box.
BOX_HEIGHT = 205

# --- Main body: Kalshi | Outlook --------------------------------------------
left, right = st.columns(2)

with left:
    with st.container(border=True, height=BOX_HEIGHT):
        st.markdown("**\u23F1\uFE0F Kalshi 15m fair value**")
        k1, k2, k3 = st.columns([1, 1, 1])
        with k1:
            st.number_input("Strike ($)", min_value=0.0, step=10.0, key="kalshi_strike")
        with k2:
            st.number_input("Minutes left", min_value=0.0, step=0.5, key="kalshi_minutes")
        with k3:
            st.checkbox("Auto-track", key="kalshi_auto_track", value=True,
                        help="Pulls the currently open KXBTC15M contract's strike and close "
                             "time from Kalshi's public API each refresh.")

        if auto_track and not kalshi_pair_ok:
            st.caption(f"Auto-track only tracks KXBTC15M on {KALSHI_PAIR} -- on {product_id} "
                       "this is a plain calculator on the values above.")
        if auto_track and kalshi_live and not kalshi_live["available"]:
            st.caption(f"\u26A0\uFE0F Auto-track couldn't reach Kalshi ({kalshi_live['reason']}) "
                       "-- using the values above.")

        if not kalshi_result["available"]:
            st.warning(_esc(f"Can't compute fair value: {kalshi_result['reason']}"))
        else:
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("P(YES)", f"{kalshi_result['p_yes'] * 100:.1f}%")
            with r2:
                st.metric("P(NO)", f"{kalshi_result['p_no'] * 100:.1f}%")
            with r3:
                if kalshi_quote is not None:
                    st.metric("Market YES", f"{kalshi_quote * 100:.0f}\u00a2",
                              f"{(kalshi_result['p_yes'] - kalshi_quote) * 100:+.1f} pts")
                else:
                    st.metric("Ann. vol", f"{kalshi_result['annual_vol'] * 100:.1f}%")

            bits = [
                f"${kalshi_result['current_price']:,.2f} vs ${kalshi_result['strike']:,.2f}",
                f"\u00b1${kalshi_result['price_std']:,.2f} sd",
                f"vol {kalshi_result['annual_vol'] * 100:.0f}%",
            ]
            if kalshi_live and kalshi_live["available"]:
                bits.append(f"`{kalshi_live['ticker']}`")
            st.caption(_esc(" \u00b7 ".join(bits)))
            if stale_price_warning:
                st.caption(_esc(stale_price_warning))
            if kalshi_quote is not None:
                st.caption(_esc(MODEL_VS_MARKET_NOTE))
            if kalshi_result["final_minute"]:
                st.caption("\u26A0\uFE0F Final minute: part of the settlement average is already "
                           "fixed, and it's assumed to equal the current price. Rough.")

            path = _strike_path_liquidity(book_snapshot, kalshi_result, cfg)
            if path:
                st.caption(_esc(
                    f"\U0001F9F1 **${path['notional']:,.0f}** of resting {path['side']}s between "
                    f"spot and strike on {path['venue']} -- context only, deliberately **not** "
                    f"folded into P(YES). See Notes."
                ))

with right:
    with st.container(border=True, height=BOX_HEIGHT):
        lean_icon = {"bullish": "\U0001F4C8", "bearish": "\U0001F4C9",
                     "neutral": "\u2696\ufe0f"}[snapshot.composite]
        st.markdown(f"**{lean_icon} {outlook_result['headline']}**")
        if outlook_result["detail"]:
            st.caption(_esc(outlook_result["detail"]))
        if outlook_result["caution"]:
            st.warning(_esc(outlook_result["caution"]), icon="\u26A0\uFE0F")
        if outlook_result.get("book_note"):
            st.caption(_esc(f"\U0001F9F1 {outlook_result['book_note']}"))
        if outlook_result["weighted_line"]:
            st.caption(f"\u2696\ufe0f {outlook_result['weighted_line']}")
        st.caption(
            "Commentary on the current indicator pattern, not a prediction and not financial "
            "advice. Backtested hit rates are in the Notes tab."
        )

# --- Deep-dive tabs ---------------------------------------------------------
# Tabs rather than stacked expanders: seven collapsed expanders still cost
# seven rows of vertical space, and opening one pushes everything below it
# down the page. A tab strip costs one row and swaps content in place.
# Order Book leads deliberately: Streamlit opens the first tab by default,
# and the chart is the one panel tall enough to push the page past a
# viewport. Landing on it would mean the dashboard loads already scrolled,
# which is the whole thing this layout is trying to avoid.
# Sized so the whole page lands just under a 900px viewport -- the header,
# badges, both boxes and a usable panel with no page scroll. Raise it if you
# have the vertical room and would rather see more of a panel at once; the
# only cost is scrolling the page.
TAB_BODY_HEIGHT = 300

tab_book, tab_feeds, tab_chart, tab_alerts, tab_notes = st.tabs(
    ["\U0001F4D2 Order Book", "\U0001F4E1 Feeds", "\U0001F4C8 Chart",
     "\U0001F514 Alerts", "\u2139\uFE0F Notes & limits"]
)

with tab_chart:
    # The one panel deliberately allowed to overflow the fold. Everything else
    # on the page is sized to fit a viewport; a price chart squeezed into
    # 330px would be useless, so this is the scroll the layout budgets for.
    st.plotly_chart(build_figure(df, cfg), width="stretch")

with tab_book, st.container(height=TAB_BODY_HEIGHT, border=False):
    if not cfg.ENABLE_BOOK_FEEDS:
        st.caption("Level-2 depth feeds are disabled (ENABLE_BOOK_FEEDS = False in config.py). "
                   "The order-book signal falls back to a single Coinbase REST snapshot.")
    elif not book_snapshot:
        st.caption("Depth feeds not available for this pair.")
    elif not book_snapshot.get("available"):
        st.caption(f"Waiting on depth feeds -- {book_snapshot.get('reason', 'no book yet')}.")
        _venue_status(book_snapshot.get("status", {}))
    else:
        agg = book_snapshot["aggregate"]
        persistence = book_snapshot["persistence"]
        flow = book_snapshot["flow"]

        _venue_status(book_snapshot["status"])

        book_cols = st.columns(4)
        with book_cols[0]:
            st.metric(f"Imbalance ({agg['band_bps']:.0f}bps)", f"{agg['imbalance']:+.2f}",
                      f"near {agg['imbalance_near']:+.2f} / far {agg['imbalance_far']:+.2f}")
        with book_cols[1]:
            st.metric("Venue agreement", f"{agg['agreement'] * 100:.0f}%",
                      f"{agg['venue_count']} venue(s) in aggregate")
        with book_cols[2]:
            st.metric("Held for", f"{persistence['seconds']:.0f}s",
                      persistence["direction"] or "no lean past threshold")
        with book_cols[3]:
            if flow.get("available"):
                st.metric("Resting flow", f"{flow['value']:+.2f}",
                          f"bids {flow['bid_change_pct']:+.0f}% / asks {flow['ask_change_pct']:+.0f}%")
            else:
                st.metric("Resting flow", "--", flow.get("reason", "unavailable"))

        st.caption(
            f"Depth-weighted across venues by in-band notional. The order-book signal only votes "
            f"once a lean has held past {cfg.BOOK_IMBALANCE_THRESHOLD:+.2f} for "
            f"{cfg.BOOK_PERSISTENCE_SECONDS:.0f}s -- a single lopsided snapshot is mostly noise. "
            f"{book_snapshot['samples']} samples over {book_snapshot['history_seconds']:.0f}s."
        )

        venue_rows = [
            {
                "Venue": venue,
                "Mid": f"${m['mid']:,.2f}",
                "Spread": f"{m['spread_bps']:.2f}bps" if m["spread_bps"] is not None else "--",
                "Imbalance": f"{m['imbalance']:+.2f}",
                "Microprice": f"{m['microprice_bps']:+.2f}bps" if m["microprice_bps"] is not None else "--",
                "Bid depth": f"${m['bid_notional']:,.0f}",
                "Ask depth": f"${m['ask_notional']:,.0f}",
            }
            for venue, m in agg["venues"].items()
        ]
        if venue_rows:
            st.dataframe(pd.DataFrame(venue_rows), width="stretch", hide_index=True)
        for venue, reason in (agg.get("excluded") or {}).items():
            st.caption(f"⚠️ {venue} excluded from the aggregate: {reason}")

        held_walls = [w for w in book_snapshot["walls"] if w["held"]]
        if held_walls:
            st.markdown("**Resting walls that have held**")
            wall_rows = [
                {
                    "Venue": w.get("venue", "--"),
                    "Side": "bid (below)" if w["side"] == "bid" else "ask (above)",
                    "Price": f"${w['price']:,.2f}",
                    "Distance": f"{w['distance_bps']:.0f}bps",
                    "Notional": f"${w['notional']:,.0f}",
                    "vs typical level": f"{w['multiple']:.0f}x",
                    "Held": f"{w['age_seconds']:.0f}s",
                }
                for w in held_walls[:8]
            ]
            st.dataframe(pd.DataFrame(wall_rows), width="stretch", hide_index=True)
        else:
            st.caption("No resting walls have held long enough to report.")

        st.caption(
            f"Only walls present for at least {cfg.BOOK_WALL_MIN_AGE_SECONDS:.0f}s are listed, and "
            "even those can still be pulled. Per-venue imbalances are averaged rather than the "
            "books being merged -- see the Notes tab for why, and for what these signals can and "
            "can't tell you."
        )

with tab_feeds, st.container(height=TAB_BODY_HEIGHT, border=False):
    if not cfg.ENABLE_EXCHANGE_FEEDS:
        st.caption("Multi-exchange feeds are disabled (ENABLE_EXCHANGE_FEEDS = False in config.py).")
    elif feed_snapshot is None:
        st.caption("Feeds not available for this pair.")
    else:
        _venue_status(feed_snapshot["status"])

        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.metric(f"Buy volume ({feed_snapshot['window_seconds']}s)",
                      f"${feed_snapshot['buy_notional']:,.0f}")
        with wc2:
            st.metric(f"Sell volume ({feed_snapshot['window_seconds']}s)",
                      f"${feed_snapshot['sell_notional']:,.0f}")
        with wc3:
            st.metric("Aggregated delta", f"{feed_snapshot['delta_ratio']:+.2f}")

        whale_col, liq_col = st.columns(2)
        with whale_col:
            st.markdown(_esc(f"**Recent whale trades** (≥ ${cfg.WHALE_TRADE_USD_THRESHOLD:,.0f})"))
            whales = feed_snapshot["whale_events"]
            if whales:
                whale_df = pd.DataFrame(whales[::-1][:15])
                whale_df["time"] = pd.to_datetime(whale_df["time"], unit="s").dt.strftime("%H:%M:%S")
                whale_df["notional"] = whale_df["notional"].map(lambda v: f"${v:,.0f}")
                st.dataframe(whale_df[["time", "exchange", "side", "notional", "price"]],
                             width="stretch", hide_index=True)
            else:
                st.caption("None yet this session.")
        with liq_col:
            st.markdown("**Recent liquidations**")
            liqs = feed_snapshot["liquidation_events"]
            if liqs:
                liq_df = pd.DataFrame(liqs[::-1][:15])
                liq_df["time"] = pd.to_datetime(liq_df["time"], unit="s").dt.strftime("%H:%M:%S")
                liq_df["notional"] = liq_df["notional"].map(lambda v: f"${v:,.0f}")
                st.dataframe(liq_df[["time", "exchange", "liquidated_side", "notional", "price"]],
                             width="stretch", hide_index=True)
            else:
                st.caption("None yet this session.")

        st.caption(
            "Trade data from Coinbase, Binance, and Bybit. Liquidations reflect Binance and Bybit "
            "USDT-perpetual futures (not spot) -- Coinbase spot has no liquidations."
        )

with tab_alerts, st.container(height=TAB_BODY_HEIGHT, border=False):
    if os.path.isfile(cfg.ALERT_LOG_PATH):
        try:
            log_df = pd.read_csv(cfg.ALERT_LOG_PATH)
        except pd.errors.ParserError as e:
            st.error(
                f"{cfg.ALERT_LOG_PATH} looks corrupted and can't be parsed ({e}). "
                "This can happen if the file was edited by hand or if an older, already-fixed "
                "bug wrote an inconsistent number of columns across rows. Deleting the file "
                "is the simplest fix -- a fresh one will be created on the next alert."
            )
        else:
            st.dataframe(log_df.tail(25).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("No alerts logged yet -- they'll show up here once enough indicators align.")

with tab_notes, st.container(height=TAB_BODY_HEIGHT, border=False):
    note_left, note_right = st.columns(2)

    with note_left:
        st.markdown("**Kalshi model assumptions**")
        st.caption(
            "- **Basis risk:** uses Coinbase's price as a proxy for Kalshi's actual settlement "
            "source (CF Benchmarks' multi-exchange Real-Time Index) -- these can diverge by a "
            "few dollars, especially during a volatile close.\n"
            "- **Settlement averaging:** Kalshi settles on a 60-second average of the reference "
            "price at close, not a single tick, and P(YES) prices the 60-second settlement "
            "average -- which puts it further from 50% than a point-in-time model, by 10+ "
            "points in the last couple of minutes. In the final minute, prints already taken "
            "are assumed equal to the current price (their real values aren't available "
            "here), so treat P(YES) there as rough.\n"
            "- **No drift:** assumes a pure random walk (driftless geometric Brownian motion). "
            "It deliberately ignores the bot's own directional signals.\n"
            "- **Not financial advice.** A calculator, not a guarantee."
        )

        if kalshi_quote is not None and kalshi_result["available"]:
            st.caption(
                "**On the model-vs-market gap:** " + _esc(MODEL_VS_MARKET_NOTE) + " A gap is far "
                "more often this model's own approximations above (Coinbase as a proxy for the "
                "index, minute-level data) than a mispricing."
            )

        path = _strike_path_liquidity(book_snapshot, kalshi_result, cfg)
        if path:
            st.markdown("**Resting liquidity between spot and the strike**")
            st.caption(_esc(
                f"${path['notional']:,.0f} of resting {path['side']}s sits between "
                f"${path['low']:,.2f} and ${path['high']:,.2f} on {path['venue']}. Price would "
                f"have to trade {path['direction_word']} that to settle {path['settles']}."
            ))
            for wall in path["walls"][:3]:
                st.caption(_esc(
                    f"• ${wall['notional']:,.0f} concentrated at ${wall['price']:,.2f} "
                    f"(held {wall['age_seconds']:.0f}s)"))
            st.caption(
                "**This does not adjust P(YES), and shouldn't be read as if it did.** It's one "
                "venue's resting size at one instant. Resting size can be pulled the moment price "
                "approaches, it says nothing about how much aggressive flow is coming the other "
                "way, and there's no historical order-book data to calibrate what any of it is "
                "worth. Context, not an edge."
            )

    with note_right:
        st.markdown("**Order-book signals have no backtest**")
        st.caption(
            "⚠️ Neither the order-book signal nor book flow can be backtested. No exchange "
            "publishes historical level-2 snapshots, so unlike the candle indicators there is no "
            "hit rate behind either of them -- they always carry weight 1.00 in weighted "
            "conviction, and nothing there is a measured edge. It is a better-constructed "
            "*description* of the book, which is not the same claim."
        )
        st.caption(
            "**Why venues are averaged, not merged:** Coinbase quotes BTC-USD while Binance.US "
            "and Bybit quote BTCUSDT. USDT is not a dollar and the basis drifts by tens of basis "
            "points, so pooling those price levels would build a book with a fictional mid. Each "
            "venue's imbalance is measured against its own mid, so the ratios stay comparable "
            "whatever the basis is doing."
        )

        st.markdown("**Per-indicator weights**")
        breakdown = state.weighted.get("breakdown", {})
        weight_details = weighting.compute_weight_details(state.backtest_results, cfg)

        def _why(name, vote):
            d = weight_details.get(name, {}).get(vote)
            if d is None:
                return "not backtestable"
            if d["reason"] == "no_data":
                return "no backtest data"
            edge = f"edge {d['edge'] * 100:+.1f} pts vs {d['base_rate'] * 100:.0f}% base"
            if d["reason"] == "small_sample":
                return f"{edge}; only ~{d['n_eff']:.0f} independent samples"
            if d["reason"] == "within_noise":
                return f"{edge}, within noise (z={d['z']:.1f})"
            return f"{edge}, z={d['z']:.1f}, ~{d['n_eff']:.0f} independent"

        if breakdown:
            rows = [
                {"Indicator": long_labels.get(name, name), "Vote": v["vote"].upper(),
                 "Weight": f"{v['weight']:.2f}", "Why": _why(name, v["vote"])}
                for name, v in breakdown.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("Nothing directional to weigh right now.")
        st.caption(
            "Weight 1.00 = counts the same as unweighted: no backtest data, too few independent "
            f"samples, or an edge over the base rate within {cfg.INDICATOR_WEIGHT_MIN_Z:.0f} "
            "standard errors of zero (noise). Above 1.00 = this indicator's calls have beaten "
            "the window's base rate for that direction; below 1.00 = done worse than it. Order book and book flow always show "
            "1.00 (no historical depth data exists). Aggregated exchange delta shows 1.00 until "
            "`python backtest_agg_delta.py` has been run, then a Binance.US-only estimate."
        )
