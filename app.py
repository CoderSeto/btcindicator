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

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import backtest
import config as cfg
import exchange_feeds
import kalshi
import kalshi_api
import kalshi_tracking
import orderbook
import orderbook_feeds
from charts import build_figure
from dashboard_data import load_state
from main import log_alert, log_signal_snapshot

st.set_page_config(page_title="BTC Indicator Bot", page_icon="\u20bf", layout="wide")

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
st.sidebar.button("Refresh now", use_container_width=True)
st.sidebar.caption("Changes here apply immediately. Edit config.py to change other defaults "
                    "(indicator periods, order book depth, etc).")

# These are shared module-level values, so this also updates what main.py's
# log_alert() and the indicator/signal functions see.
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
elif cached_results and "error" in cached_results:
    st.sidebar.caption(f"Last backtest attempt failed: {cached_results['error']}")
else:
    st.sidebar.caption(f"No backtest yet for {product_id} at this granularity.")

run_backtest_clicked = st.sidebar.button("Run backtest now", use_container_width=True)
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

st.title(f"\u20bf {product_id} Indicator Bot")

# --- Load data ---
state = load_state(cfg, feed_snapshot=feed_snapshot, book_snapshot=book_snapshot)

if state.error:
    st.error(f"Couldn't fetch live data: {state.error}")
    st.stop()

df, imbalance, snapshot = state.df, state.imbalance, state.snapshot
log_signal_snapshot(snapshot)
alert_fired = snapshot.strength >= cfg.COMPOSITE_ALERT_THRESHOLD
if alert_fired:
    log_alert(snapshot)

# --- Top row: headline metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(product_id, f"${snapshot.price:,.2f}")
    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')}")
with col2:
    emoji = {"bullish": "\U0001F7E2", "bearish": "\U0001F534", "neutral": "\u26aa"}[snapshot.composite]
    st.metric("Composite signal", f"{emoji} {snapshot.composite.upper()}",
              f"{snapshot.strength}/{len(snapshot.individual)} aligned")
book_read = snapshot.book or {}
with col3:
    if book_read.get("available"):
        held = book_read["persistence_seconds"]
        direction = (
            f"held {held:.0f}s \u00b7 {book_read['venue_count']} venues"
            if book_read["persistence_direction"]
            else f"{book_read['venue_count']} venues \u00b7 no held lean"
        )
        st.metric("Order book imbalance", f"{book_read['imbalance']:+.2f}", direction)
    elif imbalance is None:
        st.metric("Order book imbalance", "--", "waiting on depth feeds")
    else:
        direction = "buy pressure" if imbalance > 0.02 else "sell pressure" if imbalance < -0.02 else "balanced"
        st.metric("Order book imbalance (1 venue)", f"{imbalance:+.2f}", direction)

if alert_fired:
    st.success(
        f"\U0001F6A8 {snapshot.strength} indicators aligned {snapshot.composite.upper()} -- "
        f"logged to {cfg.ALERT_LOG_PATH}"
    )

# --- Kalshi 15m fair-value calculator (moved up top -- primary use case) ---
st.divider()
st.subheader("\u23F1\uFE0F Kalshi 15m Fair Value")
st.caption(
    "Enter the strike and time remaining from the actual Kalshi contract you're looking at. "
    "A genuinely different question from the Outlook below (probability for a specific "
    "strike/time, not a directional lean)."
)

kalshi_col1, kalshi_col2, kalshi_col3 = st.columns([1, 1, 1])
with kalshi_col3:
    auto_track = st.checkbox("Auto-track current window", value=True)

kalshi_live = None
if auto_track:
    kalshi_live = kalshi_api.current_btc15m_market()
    if kalshi_live["available"]:
        # Strike only resets when the window itself actually changes -- so
        # a manual edit mid-window survives instead of being fought every
        # refresh. Minutes-remaining is a live countdown, so it always updates.
        if st.session_state.get("kalshi_last_ticker") != kalshi_live["ticker"]:
            st.session_state["kalshi_strike"] = kalshi_live["strike"]
            st.session_state["kalshi_last_ticker"] = kalshi_live["ticker"]
        st.session_state["kalshi_minutes"] = round(kalshi_live["minutes_remaining"], 2)
    else:
        st.caption(f"\u26A0\uFE0F Auto-track couldn't reach Kalshi ({kalshi_live['reason']}) -- using manual values below.")

if "kalshi_strike" not in st.session_state:
    st.session_state["kalshi_strike"] = float(round(snapshot.price))
if "kalshi_minutes" not in st.session_state:
    st.session_state["kalshi_minutes"] = round(kalshi.minutes_until_next_kalshi_close(), 1)

with kalshi_col1:
    kalshi_strike = st.number_input("Strike price ($)", min_value=0.0, step=10.0, key="kalshi_strike")
with kalshi_col2:
    kalshi_minutes = st.number_input("Minutes remaining", min_value=0.0, step=0.5, key="kalshi_minutes")

if kalshi_live and kalshi_live["available"]:
    quote_bits = [f"Tracking: `{kalshi_live['ticker']}`"]
    if kalshi_live["yes_bid"] is not None and kalshi_live["yes_ask"] is not None:
        quote_bits.append(
            f"Kalshi YES quote: {kalshi_live['yes_bid'] * 100:.0f}\u00a2/{kalshi_live['yes_ask'] * 100:.0f}\u00a2"
        )
    st.caption(" \u00b7 ".join(quote_bits))

kalshi_result = kalshi.fair_value(df, kalshi_strike, kalshi_minutes, cfg.KALSHI_VOL_LOOKBACK_MINUTES)

if kalshi_result["available"] and kalshi_live and kalshi_live["available"]:
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
    )

if not kalshi_result["available"]:
    st.warning(f"Can't compute fair value: {kalshi_result['reason']}")
else:
    show_gap = bool(
        kalshi_live and kalshi_live["available"]
        and kalshi_live["yes_bid"] is not None and kalshi_live["yes_ask"] is not None
    )
    metric_cols = st.columns(4 if show_gap else 3)
    with metric_cols[0]:
        st.metric("P(YES / above strike)", f"{kalshi_result['p_yes'] * 100:.1f}%")
    with metric_cols[1]:
        st.metric("P(NO / below strike)", f"{kalshi_result['p_no'] * 100:.1f}%")
    with metric_cols[2]:
        st.metric("Annualized vol used", f"{kalshi_result['annual_vol'] * 100:.1f}%")
    if show_gap:
        kalshi_mid = (kalshi_live["yes_bid"] + kalshi_live["yes_ask"]) / 2
        gap_pts = (kalshi_result["p_yes"] - kalshi_mid) * 100
        with metric_cols[3]:
            st.metric("Kalshi's own YES quote", f"{kalshi_mid * 100:.0f}\u00a2", f"{gap_pts:+.1f} pts vs model")
        st.caption(
            "The gap is model-vs-market, not a signal to act on -- it can reflect a real "
            "mispricing, or just this model's own approximations (basis risk, settlement "
            "averaging -- see below) being wrong instead. Worth understanding why before "
            "trusting it."
        )

    st.caption(
        f"${kalshi_result['current_price']:,.2f} vs strike ${kalshi_result['strike']:,.2f} \u00b7 "
        f"{kalshi_result['minutes_remaining']:.1f} min left \u00b7 "
        f"std dev \u00b1${kalshi_result['price_std']:,.2f} "
        f"({kalshi_result['vol_samples_used']}/{kalshi_result['vol_samples_requested']} vol samples)"
    )

    # Resting liquidity between spot and the strike. Deliberately NOT
    # folded into P(YES) above: this project's rule is that independently
    # sourced numbers don't get blended without validated evidence for the
    # blending method, and there is none here -- no historical order-book
    # data exists to calibrate what a given amount of resting size does to
    # a crossing probability. So it's shown as what it is: a factual
    # description of the path, next to an unchanged model number.
    if book_snapshot and book_snapshot.get("available"):
        venue_books = book_snapshot.get("books", {})
        # Prefer Coinbase: it's the venue the price and the fair-value model
        # both come from, and it's quoted in actual dollars rather than USDT.
        path_venue = "coinbase" if "coinbase" in venue_books else next(iter(venue_books), None)
        path_book = venue_books.get(path_venue) if path_venue else None
        spot = kalshi_result["current_price"]
        strike = kalshi_result["strike"]

        if path_book and abs(strike - spot) / max(spot, 1) <= cfg.BOOK_RETAIN_BPS / 10_000.0:
            side = "ask" if strike > spot else "bid"
            path_notional = orderbook.notional_between(path_book, spot, strike, side)
            walls_in_path = [
                w for w in book_snapshot.get("walls", [])
                if w.get("held") and w.get("venue") == path_venue and w["side"] == side
                and min(spot, strike) <= w["price"] <= max(spot, strike)
            ]
            with st.expander("Resting liquidity between spot and the strike"):
                direction_word = "up through" if strike > spot else "down through"
                settles = "YES" if strike > spot else "NO"
                st.markdown(
                    f"**${path_notional:,.0f}** of resting {side}s sits between "
                    f"${min(spot, strike):,.2f} and ${max(spot, strike):,.2f} on "
                    f"{path_venue}. Price would have to trade {direction_word} that "
                    f"to settle {settles}."
                )
                for wall in walls_in_path[:3]:
                    st.caption(
                        f"\u2022 ${wall['notional']:,.0f} concentrated at ${wall['price']:,.2f} "
                        f"(held {wall['age_seconds']:.0f}s)"
                    )
                st.caption(
                    "**This does not adjust the probability above, and shouldn't be read as if "
                    "it did.** It's one venue's resting size at one instant. Resting size can be "
                    "pulled the moment price approaches, it says nothing about how much "
                    "aggressive flow is coming the other way, and there's no historical "
                    "order-book data to calibrate what any of it is worth. Context, not an edge."
                )

    with st.expander("Compare against the bot's backtested signals"):
        confidence_lines = state.outlook.get("confidence_lines")
        if confidence_lines:
            for line in confidence_lines:
                st.info(f"\U0001F4CA {line}")
            st.caption(
                "A SEPARATE empirical read, not blended into the probability above -- only "
                "genuinely comparable if BACKTEST_LOOKAHEAD_CANDLES is set to ~15 to match "
                "this window."
            )
        else:
            st.caption("No backtest yet -- run one from the sidebar first.")

with st.expander("Kalshi model assumptions and limitations"):
    st.markdown(
        "- **Basis risk:** uses Coinbase's price as a proxy for Kalshi's actual settlement "
        "source (CF Benchmarks' multi-exchange Real-Time Index) -- these can diverge by a "
        "few dollars, especially during a volatile close.\n"
        "- **Settlement averaging:** Kalshi settles on a 60-second average of the reference "
        "price at close, not a single tick. This model treats it as a single point-in-time "
        "draw, which slightly overstates how extreme the probability gets in the final "
        "minute or two.\n"
        "- **No drift:** assumes a pure random walk (driftless geometric Brownian motion). "
        "It deliberately ignores the bot's own directional signals below.\n"
        "- **Not financial advice.** A calculator, not a guarantee."
    )
st.divider()

# --- Outlook ---
st.subheader("Outlook")
outlook_result = state.outlook
lean_icon = {"bullish": "\U0001F4C8", "bearish": "\U0001F4C9", "neutral": "\u2696\ufe0f"}[snapshot.composite]
st.markdown(f"{lean_icon} **{outlook_result['headline']}**")
if outlook_result["detail"]:
    st.write(outlook_result["detail"])
if outlook_result["caution"]:
    st.warning(outlook_result["caution"])
if outlook_result.get("book_note"):
    st.markdown(f"\U0001F9F1 {outlook_result['book_note']}")
if outlook_result["weighted_line"]:
    st.markdown(f"\u2696\ufe0f {outlook_result['weighted_line']}")
    with st.expander("See per-indicator weights"):
        breakdown = state.weighted.get("breakdown", {})
        if breakdown:
            weight_labels = {
                "sma_cross": "SMA Cross", "ema_cross": "EMA Cross", "rsi": "RSI",
                "macd": "MACD", "bollinger": "Bollinger", "volume": "Volume Spike",
                "orderbook": "Order Book", "book_flow": "Book Flow",
                "agg_delta": "Agg. Exchange Delta",
            }
            rows = [
                {"Indicator": weight_labels.get(name, name), "Vote": v["vote"].upper(), "Weight": f"{v['weight']:.2f}"}
                for name, v in breakdown.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Weight 1.00 = counts the same as unweighted (no backtest data, or a coin-flip "
            "historical hit rate). Above 1.00 = this indicator's calls have historically been "
            "right more often; below 1.00 = less often. Order book, book flow and aggregated "
            "exchange delta always show 1.00 -- no historical order-book data exists to "
            "backtest the first two at all."
        )
with st.expander("Backtested confidence for the raw composite", expanded=bool(outlook_result["confidence_lines"])):
    if outlook_result["confidence_lines"]:
        for line in outlook_result["confidence_lines"]:
            st.info(f"\U0001F4CA {line}")
    else:
        st.caption("No backtest yet for this pair -- run one from the sidebar to see historical hit rates here.")
st.caption(
    "This reads the current pattern of indicators -- it's not a prediction and not financial advice. "
    "Always do your own research."
)

# --- Indicator badges ---
st.subheader("Indicators")
labels = {
    "sma_cross": "SMA Cross", "ema_cross": "EMA Cross", "rsi": "RSI",
    "macd": "MACD", "bollinger": "Bollinger", "volume": "Volume Spike",
    "orderbook": "Order Book", "book_flow": "Book Flow",
    "agg_delta": "Agg. Exchange Delta",
}
emoji_map = {"bullish": "\U0001F7E2", "bearish": "\U0001F534", "neutral": "\u26aa"}
badge_cols = st.columns(len(snapshot.individual))
for col, (name, val) in zip(badge_cols, snapshot.individual.items()):
    with col:
        st.markdown(f"**{labels.get(name, name)}**")
        st.markdown(f"{emoji_map[val]} {val.upper()}")

# --- Chart (collapsed by default -- takes real space) ---
with st.expander("Price Chart", expanded=False):
    st.plotly_chart(build_figure(df, cfg), use_container_width=True)

# --- Aggregated order book (collapsed by default) ---
with st.expander("Order Book Depth (Coinbase / Binance.US / Bybit)", expanded=False):
    if not cfg.ENABLE_BOOK_FEEDS:
        st.caption("Level-2 depth feeds are disabled (ENABLE_BOOK_FEEDS = False in config.py). "
                    "The order-book signal falls back to a single Coinbase REST snapshot.")
    elif not book_snapshot:
        st.caption("Depth feeds not available for this pair.")
    elif not book_snapshot.get("available"):
        st.caption(f"Waiting on depth feeds -- {book_snapshot.get('reason', 'no book yet')}.")
        status_cols = st.columns(max(len(book_snapshot.get("status", {})), 1))
        for col, (name, val) in zip(status_cols, book_snapshot.get("status", {}).items()):
            with col:
                icon = "\U0001F7E2" if val == "connected" else (
                    "\U0001F7E1" if val in ("connecting", "reconnecting", "syncing book") else "\U0001F534")
                st.markdown(f"{icon} **{name}**")
                st.caption(val)
    else:
        agg = book_snapshot["aggregate"]
        persistence = book_snapshot["persistence"]
        flow = book_snapshot["flow"]

        status_cols = st.columns(max(len(book_snapshot["status"]), 1))
        for col, (name, val) in zip(status_cols, book_snapshot["status"].items()):
            with col:
                icon = "\U0001F7E2" if val == "connected" else (
                    "\U0001F7E1" if val in ("connecting", "reconnecting", "syncing book") else "\U0001F534")
                st.markdown(f"{icon} **{name}**")
                st.caption(val)

        book_cols = st.columns(4)
        with book_cols[0]:
            st.metric(f"Imbalance ({agg['band_bps']:.0f}bps)", f"{agg['imbalance']:+.2f}",
                      f"near {agg['imbalance_near']:+.2f} / far {agg['imbalance_far']:+.2f}")
        with book_cols[1]:
            st.metric("Venue agreement", f"{agg['agreement'] * 100:.0f}%",
                      f"{agg['venue_count']} venue(s) in aggregate")
        with book_cols[2]:
            st.metric("Held for",
                      f"{persistence['seconds']:.0f}s",
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
            st.dataframe(pd.DataFrame(venue_rows), use_container_width=True, hide_index=True)
        for venue, reason in (agg.get("excluded") or {}).items():
            st.caption(f"\u26A0\uFE0F {venue} excluded from the aggregate: {reason}")
        st.caption(
            "Per-venue imbalances are averaged rather than the books being merged: Coinbase "
            "quotes BTC-USD while the others quote BTCUSDT, and pooling those price levels "
            "would build a book with a fictional mid. Each venue's imbalance is measured "
            "against its own mid, so the ratios stay comparable whatever the basis is doing."
        )

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
            st.dataframe(pd.DataFrame(wall_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Only walls present for at least {cfg.BOOK_WALL_MIN_AGE_SECONDS:.0f}s are listed. "
                "A large level that appears and vanishes as price approaches is "
                "indistinguishable from a spoof, and age is the only evidence available "
                "either way. Even a wall that has held can still be pulled."
            )
        else:
            st.caption("No resting walls have held long enough to report.")

        st.caption(
            "\u26A0\uFE0F Neither the order-book signal nor book flow can be backtested. No exchange "
            "publishes historical level-2 snapshots, so unlike the candle indicators there is no "
            "hit rate behind either of them -- they always carry weight 1.00 in weighted "
            "conviction, and nothing here is a measured edge."
        )

# --- Live multi-exchange feeds (collapsed by default) ---
with st.expander("Live Feeds (Coinbase / Binance.US / Bybit)", expanded=False):
    if not cfg.ENABLE_EXCHANGE_FEEDS:
        st.caption("Multi-exchange feeds are disabled (ENABLE_EXCHANGE_FEEDS = False in config.py).")
    elif feed_snapshot is None:
        st.caption("Feeds not available for this pair.")
    else:
        status = feed_snapshot["status"]
        status_icon = {"connected": "\U0001F7E2"}
        status_cols = st.columns(max(len(status), 1))
        if status:
            for col, (name, val) in zip(status_cols, status.items()):
                with col:
                    icon = status_icon.get(val, "\U0001F7E1" if val in ("connecting", "reconnecting") else "\U0001F534")
                    st.markdown(f"**{name}**")
                    st.markdown(f"{icon} {val}")
        else:
            st.caption("Connecting...")

        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.metric(f"Buy volume ({feed_snapshot['window_seconds']}s)", f"${feed_snapshot['buy_notional']:,.0f}")
        with wc2:
            st.metric(f"Sell volume ({feed_snapshot['window_seconds']}s)", f"${feed_snapshot['sell_notional']:,.0f}")
        with wc3:
            st.metric("Aggregated delta", f"{feed_snapshot['delta_ratio']:+.2f}")

        whale_col, liq_col = st.columns(2)
        with whale_col:
            st.markdown(f"**Recent whale trades** (\u2265 ${cfg.WHALE_TRADE_USD_THRESHOLD:,.0f})")
            whales = feed_snapshot["whale_events"]
            if whales:
                whale_df = pd.DataFrame(whales[::-1][:15])
                whale_df["time"] = pd.to_datetime(whale_df["time"], unit="s").dt.strftime("%H:%M:%S")
                whale_df["notional"] = whale_df["notional"].map(lambda v: f"${v:,.0f}")
                st.dataframe(
                    whale_df[["time", "exchange", "side", "notional", "price"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("None yet this session.")
        with liq_col:
            st.markdown("**Recent liquidations**")
            liqs = feed_snapshot["liquidation_events"]
            if liqs:
                liq_df = pd.DataFrame(liqs[::-1][:15])
                liq_df["time"] = pd.to_datetime(liq_df["time"], unit="s").dt.strftime("%H:%M:%S")
                liq_df["notional"] = liq_df["notional"].map(lambda v: f"${v:,.0f}")
                st.dataframe(
                    liq_df[["time", "exchange", "liquidated_side", "notional", "price"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("None yet this session.")

        st.caption(
            "Trade data from Coinbase, Binance, and Bybit. Liquidations reflect Binance and Bybit "
            "USDT-perpetual futures (not spot) -- Coinbase spot has no liquidations."
        )

# --- Alert log (collapsed by default) ---
with st.expander("Recent alerts", expanded=False):
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
            st.dataframe(log_df.tail(25).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts logged yet -- they'll show up here once enough indicators align.")
