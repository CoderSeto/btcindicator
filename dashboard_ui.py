"""
Rendering for the Streamlit dashboard (app.py). Every function takes the
per-run settings copy `cfg` explicitly and never writes to the shared config
module (CLAUDE.md gotcha 14). Every string carrying `$` that reaches
st.markdown / st.caption / st.warning goes through `_esc`.

Layout, decision first:
  1. the live strip (`live_strip`, run as an st.fragment on its own
     STRIP_REFRESH_SECONDS timer): price, Kalshi quote and its age, P(YES),
     time left, the decision line and its one-line caveat;
  2. composite / order book / alert, signal badges;
  3. the Kalshi inputs box and the outlook;
  4. tabs for everything heavy, Notes for the long explanations.
"""
import os
import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import data_fetch
import kalshi
import kalshi_api
import kalshi_decision
import orderbook
import paper_trading
import weighting
from charts import build_figure

EMOJI = {"bullish": "\U0001F7E2", "bearish": "\U0001F534", "neutral": "⚪"}
SHORT_LABELS = {
    "sma_cross": "SMA", "ema_cross": "EMA", "rsi": "RSI", "macd": "MACD",
    "bollinger": "BOLL", "volume": "VOL", "orderbook": "BOOK",
    "book_flow": "FLOW", "agg_delta": "DELTA",
}
LONG_LABELS = {
    "sma_cross": "SMA Cross", "ema_cross": "EMA Cross", "rsi": "RSI",
    "macd": "MACD", "bollinger": "Bollinger", "volume": "Volume Spike",
    "orderbook": "Order Book", "book_flow": "Book Flow",
    "agg_delta": "Agg. Exchange Delta",
}
KALSHI_PAIR = "BTC-USD"   # KXBTC15M is a BTC/USD contract

# The study's answer to "is the model-vs-market gap an edge?" -- shown once, in Notes.
MODEL_VS_MARKET_NOTE = (
    "Model minus market is not a trading signal. On 30,688 settled KXBTC15M contracts the model "
    "scored worse than Kalshi's own mid on held-out data, and trading toward it lost money after "
    "fees (REVIEW_FINDINGS.md, \"Is there an edge?\")."
)

# The page sizes: fixed-height scrollers keep the page height constant whatever wraps.
BOX_HEIGHT = 205
TAB_BODY_HEIGHT = 300

CSS = """
<style>
  /* Streamlit's defaults are generous with vertical space; these hook its test IDs, which
     change between releases -- a selector that stops matching just makes the page roomier. */
  .block-container {padding-top: 2.4rem; padding-bottom: 0.8rem;}
  [data-testid="stVerticalBlock"] {gap: 0.4rem;}
  [data-testid="stHorizontalBlock"] {gap: 0.5rem;}
  [data-testid="stVerticalBlockBorderWrapper"] {padding: 0.1rem 0.15rem;}
  [data-testid="stMetricValue"] {font-size: 1.35rem; line-height: 1.5rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.72rem;}
  [data-testid="stMetricDelta"] {font-size: 0.7rem;}
  [data-testid="stTabs"] button p {font-size: 0.8rem;}
  h1, h2, h3, h4 {margin-bottom: 0.3rem; padding-top: 0.1rem;}
  hr {margin: 0.35rem 0;}
</style>
"""


# --- Small helpers --------------------------------------------------------------

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


def _book_status(book_snapshot: dict) -> dict:
    """Depth-feed status per venue, with a connected venue whose book stopped
    changing shown as frozen: it's left out of the read (BOOK_STALE_SECONDS)."""
    frozen = set(book_snapshot.get("frozen_venues", []))
    return {v: ("frozen, not counted" if v in frozen else s)
            for v, s in book_snapshot.get("status", {}).items()}


def _venue_status(status: dict):
    """One compact row of connection dots, shared by the depth and trade feeds."""
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
    retained band). A plain factual readout, never fed back into P(YES)
    (design principle 1: no validated way to blend it).
    """
    if not (book_snapshot and book_snapshot.get("available") and kalshi_result.get("available")):
        return None
    venue_books = book_snapshot.get("books", {})
    # Prefer Coinbase: the venue the price and the model come from, quoted in dollars, not USDT.
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
        "venue": venue, "side": side, "low": low, "high": high,
        "notional": orderbook.notional_between(book, spot, strike, side),
        "direction_word": "up through" if strike > spot else "down through",
        "settles": "YES" if strike > spot else "NO",
        "walls": [w for w in book_snapshot.get("walls", [])
                  if w.get("held") and w.get("venue") == venue and w["side"] == side
                  and low <= w["price"] <= high],
    }


# --- The ticker, shared by every tab ----------------------------------------------
#
# data_fetch.latest_price() (Coinbase /ticker) is the only price source (gotcha 19). The strip
# reruns every STRIP_REFRESH_SECONDS in every open tab; this keeps that to one ticker request
# per interval for the whole process, and the page refresh seeds it with load_state's own read.

_price_cache = {}
_price_lock = threading.Lock()


def seed_price(product_id: str, result: dict):
    with _price_lock:
        _price_cache[product_id] = (time.monotonic(), result)


def shared_latest_price(product_id: str, max_age: float, candles=None) -> dict:
    with _price_lock:
        hit = _price_cache.get(product_id)
        if hit and time.monotonic() - hit[0] < max_age:
            return hit[1]
    result = data_fetch.latest_price(product_id, candles)
    seed_price(product_id, result)
    return result


# --- The live strip: computation (pure apart from the two shared polls) ------------

def live_view(cfg, ctx: dict, kalshi_live: dict, price: dict) -> dict:
    """
    Everything the strip shows, from the freshest price and quote. `ctx` is what the
    page refresh hands over: product_id, price_df, strike, manual minutes, auto_track,
    decision rule/result and the open paper positions.
    """
    live_price = price["price"]
    minutes = ctx["minutes"]
    live_ok = bool(kalshi_live and kalshi_live.get("available"))
    if live_ok:
        minutes = kalshi_live["minutes_remaining"]
    result = kalshi.fair_value(
        ctx["price_df"], ctx["strike"], minutes, cfg.KALSHI_VOL_LOOKBACK_MINUTES,
        candle_seconds=cfg.GRANULARITY, current_price=live_price,
        max_strike_distance=cfg.KALSHI_MAX_STRIKE_DISTANCE)
    quote = None
    if live_ok and kalshi_live.get("yes_bid") is not None and kalshi_live.get("yes_ask") is not None:
        quote = (kalshi_live["yes_bid"] + kalshi_live["yes_ask"]) / 2
    age = kalshi_live.get("quote_age_seconds") if live_ok else None
    decision = grade = timing = exit_ = None
    if result["available"] and live_ok and result["strike"] == kalshi_live["strike"]:
        grade = kalshi_decision.entry_quality(kalshi_live["yes_bid"], kalshi_live["yes_ask"],
                                              kalshi_live.get("last_price"), minutes, 1)
        decision = kalshi_decision.decide_action(
            ctx["rule"], result["p_yes"], kalshi_live["yes_bid"], kalshi_live["yes_ask"], minutes, 1,
            ctx["rule_result"], quote_age_seconds=age, max_quote_age_seconds=cfg.KALSHI_MAX_QUOTE_AGE_SECONDS)
        timing = kalshi_decision.entry_timing(grade, minutes)
        held = [t for t in ctx["open_trades"] if t["ticker"] == kalshi_live["ticker"]]
        if held:
            exit_ = kalshi_decision.decide_exit(held[-1]["side"], result["p_yes"], kalshi_live["yes_bid"],
                                                kalshi_live["yes_ask"], minutes, ctx["rule"], held[-1]["contracts"])
    return {"price": price, "kalshi": kalshi_live, "result": result, "quote": quote, "quote_age": age,
            "stale": age is not None and age > cfg.KALSHI_MAX_QUOTE_AGE_SECONDS,
            "decision": decision, "grade": grade, "timing": timing, "exit": exit_, "minutes": minutes}


def _quote_help(k: dict) -> str:
    last = k.get("last_429")
    return (f"Top of Kalshi's orderbook (public REST, polled once for all tabs). Poll interval now "
            f"{k.get('poll_interval', 0):.2f}s, adaptive: doubles on 429/5xx/timeout, shrinks after 30 "
            f"straight successes. Last 429: {last.strftime('%H:%M:%S') + ' UTC' if last else 'none'}. "
            f"{k.get('requests', 0)} requests this process.")


def render_strip(cfg, view: dict, product_id: str):
    """The decision-first top of the page."""
    price, k, res = view["price"], view["kalshi"], view["result"]
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 0.8])
    with c1:
        st.metric(product_id, f"${price['price']:,.2f}",
                  help="Last trade, from Coinbase's ticker, refreshed every "
                       f"{cfg.STRIP_REFRESH_SECONDS:g}s. Signals are scored on completed candles, "
                       "so they can lag this by up to one candle.")
    with c2:
        if k and k.get("available") and k.get("yes_bid") is not None and k.get("yes_ask") is not None:
            age = view["quote_age"]
            st.metric("Kalshi YES", f"{k['yes_bid'] * 100:.0f}¢ / {k['yes_ask'] * 100:.0f}¢",
                      (f"⚠️ {age:.0f}s old, stale" if view["stale"] else f"{age:.1f}s old"),
                      delta_color="off", help=_quote_help(k))
        elif k and k.get("available"):
            st.metric("Kalshi YES", "--", "one-sided book", delta_color="off", help=_quote_help(k))
        else:
            st.metric("Kalshi YES", "--", (k or {}).get("reason", "auto-track off")[:38], delta_color="off")
    with c3:
        if not res["available"]:
            st.metric("P(YES)", "--", res["reason"][:38])
        else:
            gap = (f"{(res['p_yes'] - view['quote']) * 100:+.0f} pts vs market" if view["quote"] is not None
                   else f"{res['minutes_remaining']:.1f} min left")
            st.metric("P(YES)", f"{res['p_yes'] * 100:.1f}%", gap,
                      help=f"Model fair value (P(NO) {res['p_no'] * 100:.1f}%). Never blended with the "
                           "bot's signals; model minus market is not a trading signal -- see Notes.")
    with c4:
        m = view["minutes"]
        st.metric("Time left", f"{int(m)}:{int(round((m % 1) * 60)) % 60:02d}" if m is not None else "--")
    d = view["decision"]
    if d is not None:
        st.markdown(_esc(kalshi_decision.decision_line(d, view["grade"], view["timing"], view["exit"])))
        st.caption(_esc(d["caveat"]))
        with st.expander("Why", expanded=False):
            ex = view["exit"]
            st.caption(_esc(" ".join(d["reasons"] + ([d["flip"] + "."] if d["flip"] else []))
                            + f" Timing: {view['timing']['reason']}."
                            + (" " + ex["reason"] + ". " + ex.get("evidence", "") if ex is not None else "")))
            g = view["grade"]
            bits = []
            if g["cost"] is not None:
                c = g["cost"]
                bits.append(f"Entry cost (half-spread + fee): YES {c['cost_yes'] * 100:.1f}c, "
                            f"NO {c['cost_no'] * 100:.1f}c; break-even YES {c['breakeven_yes'] * 100:.1f}%, "
                            f"NO {c['breakeven_no'] * 100:.1f}%.")
            bits += [r[0].upper() + r[1:] + "." for r in g["reasons"]]
            bits.append(kalshi_decision.calibration_line(g["calibration"]))
            st.caption(_esc(" ".join(b for b in bits if b)))
            if d["evidence"]:
                st.caption(_esc(d["evidence"]))
    if price["source"] != "ticker":
        st.caption(_esc(f"⚠️ Price from the newest candle, not the ticker ({price['reason']}). "
                        f"{data_fetch.CANDLE_CACHE_NOTE}."))
    if res["available"] and res["final_minute"]:
        st.caption("⚠️ Final minute: part of the settlement average is already fixed and "
                   "assumed equal to the current price. Rough.")


def live_strip(cfg, ctx: dict):
    """Fragment body: re-read the shared price and quote, recompute, render."""
    price = shared_latest_price(ctx["product_id"], cfg.STRIP_REFRESH_SECONDS / 2, ctx["candles"])
    if price["price"] is None:
        st.error("No price available.")
        return
    k = kalshi_api.live_market(cfg) if ctx["auto_track"] else None
    render_strip(cfg, live_view(cfg, ctx, k, price), ctx["product_id"])


# --- Second row: composite, book, alert, badges ------------------------------------

def render_signal_row(cfg, snapshot, imbalance, book_read: dict, alert_fired: bool):
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.metric("Composite", f"{EMOJI[snapshot.composite]} {snapshot.composite.upper()}",
                  f"{snapshot.strength}/{len(snapshot.individual)} aligned")
    with c2:
        if book_read.get("available"):
            held = book_read["persistence_seconds"]
            sub = (f"held {held:.0f}s · {book_read['venue_count']} venues" if book_read["persistence_direction"]
                   else f"{book_read['venue_count']} venues · no held lean")
            st.metric("Order book", f"{book_read['imbalance']:+.2f}", sub,
                      help="Notional depth within the configured band of mid, distance-weighted and averaged "
                           "across venues. Only votes once a lean has held.")
        elif imbalance is None:
            st.metric("Order book", "--", "waiting on depth feeds")
        else:
            sub = "buy pressure" if imbalance > 0.02 else "sell pressure" if imbalance < -0.02 else "balanced"
            st.metric("Order book (1 venue)", f"{imbalance:+.2f}", sub,
                      help="Fallback reading: one Coinbase REST snapshot, unweighted base volume over a fixed "
                           "level count. Not the same scale as the aggregated reading.")
    with c3:
        badge_cols = st.columns(max(len(snapshot.individual), 1))
        for col, (name, val) in zip(badge_cols, snapshot.individual.items()):
            with col:
                st.markdown(
                    f"<div style='text-align:center;line-height:1.25' title='{LONG_LABELS.get(name, name)}: {val}'>"
                    f"<span style='font-size:0.68rem;opacity:0.65'>{SHORT_LABELS.get(name, name)}</span><br>"
                    f"<span style='font-size:0.95rem'>{EMOJI[val]}</span></div>",
                    unsafe_allow_html=True)
        if alert_fired:
            st.markdown(f"<div style='font-size:0.8rem'>🚨 <b>{snapshot.strength} aligned "
                        f"{snapshot.composite.upper()}</b><span style='opacity:0.6'> · logged to "
                        f"{cfg.ALERT_LOG_PATH}</span></div>", unsafe_allow_html=True)
    missing = [n for n in LONG_LABELS if n not in snapshot.individual]
    if missing:
        st.caption("Warming up: " + ", ".join(LONG_LABELS[n] for n in missing)
                   + " -- absent rather than voting neutral until their feeds have enough history.")


# --- Kalshi inputs and outlook ----------------------------------------------------------

def render_kalshi_box(cfg, product_id, auto_track, kalshi_live, kalshi_result, book_snapshot):
    with st.container(border=True, height=BOX_HEIGHT):
        st.markdown("**⏱️ Kalshi 15m inputs**")
        k1, k2, k3, k4 = st.columns([1, 1, 0.8, 0.8])
        with k1:
            st.number_input("Strike ($)", min_value=0.0, step=10.0, key="kalshi_strike")
        with k2:
            st.number_input("Minutes left", min_value=0.0, step=0.5, key="kalshi_minutes")
        with k3:
            st.checkbox("Auto-track", key="kalshi_auto_track", value=True,
                        help="Pulls the open KXBTC15M contract's strike and close time from Kalshi's "
                             "public API, and its quote for the strip above.")
        with k4:
            if kalshi_result["available"]:
                st.metric("Ann. vol", f"{kalshi_result['annual_vol'] * 100:.1f}%")
        if auto_track and product_id != KALSHI_PAIR:
            st.caption(f"Auto-track only tracks KXBTC15M on {KALSHI_PAIR} -- on {product_id} "
                       "this is a plain calculator on the values above.")
        if auto_track and kalshi_live and not kalshi_live["available"]:
            st.caption(_esc(f"⚠️ Auto-track couldn't reach Kalshi ({kalshi_live['reason']}) "
                            "-- using the values above."))
        if not kalshi_result["available"]:
            st.warning(_esc(f"Can't compute fair value: {kalshi_result['reason']}"))
            return
        bits = [f"${kalshi_result['current_price']:,.2f} vs ${kalshi_result['strike']:,.2f}",
                f"±${kalshi_result['price_std']:,.2f} sd"]
        if kalshi_live and kalshi_live["available"]:
            bits.append(f"`{kalshi_live['ticker']}`")
        st.caption(_esc(" · ".join(bits)))
        path = _strike_path_liquidity(book_snapshot, kalshi_result, cfg)
        if path:
            st.caption(_esc(f"\U0001F9F1 **${path['notional']:,.0f}** of resting {path['side']}s between spot "
                            f"and strike on {path['venue']} -- context only, **not** folded into P(YES)."))


def render_outlook(snapshot, outlook_result):
    with st.container(border=True, height=BOX_HEIGHT):
        lean_icon = {"bullish": "\U0001F4C8", "bearish": "\U0001F4C9", "neutral": "⚖️"}[snapshot.composite]
        st.markdown(f"**{lean_icon} {outlook_result['headline']}**")
        if outlook_result["detail"]:
            st.caption(_esc(outlook_result["detail"]))
        if outlook_result["caution"]:
            st.warning(_esc(outlook_result["caution"]), icon="⚠️")
        if outlook_result.get("book_note"):
            st.caption(_esc(f"\U0001F9F1 {outlook_result['book_note']}"))
        if outlook_result["weighted_line"]:
            st.caption(_esc(f"⚖️ {outlook_result['weighted_line']}"))


# --- Tabs ---------------------------------------------------------------------------------

def render_chart_tab(cfg, df, product_id):
    # Rebuilt only when the data under it changes (a new candle or a moved last close).
    key = (product_id, len(df), str(df["time"].iloc[-1]), float(df["close"].iloc[-1]))
    if st.session_state.get("_chart_key") != key:
        st.session_state["_chart_fig"], st.session_state["_chart_key"] = build_figure(df, cfg), key
    st.plotly_chart(st.session_state["_chart_fig"], width="stretch")


def render_book_tab(cfg, book_snapshot):
    if not cfg.ENABLE_BOOK_FEEDS:
        st.caption("Level-2 depth feeds are disabled (ENABLE_BOOK_FEEDS = False in config.py). "
                   "The order-book signal falls back to a single Coinbase REST snapshot.")
        return
    if not book_snapshot:
        st.caption("Depth feeds not available for this pair.")
        return
    if not book_snapshot.get("available"):
        st.caption(f"Waiting on depth feeds -- {book_snapshot.get('reason', 'no book yet')}.")
        _venue_status(_book_status(book_snapshot))
        return
    agg, persistence, flow = book_snapshot["aggregate"], book_snapshot["persistence"], book_snapshot["flow"]
    _venue_status(_book_status(book_snapshot))
    cols = st.columns(4)
    with cols[0]:
        st.metric(f"Imbalance ({agg['band_bps']:.0f}bps)", f"{agg['imbalance']:+.2f}",
                  f"near {agg['imbalance_near']:+.2f} / far {agg['imbalance_far']:+.2f}")
    with cols[1]:
        st.metric("Venue agreement", f"{agg['agreement'] * 100:.0f}%", f"{agg['venue_count']} venue(s) in aggregate")
    with cols[2]:
        st.metric("Held for", f"{persistence['seconds']:.0f}s", persistence["direction"] or "no lean past threshold")
    with cols[3]:
        if flow.get("available"):
            st.metric("Resting flow", f"{flow['value']:+.2f}",
                      f"bids {flow['bid_change_pct']:+.0f}% / asks {flow['ask_change_pct']:+.0f}%")
        else:
            st.metric("Resting flow", "--", flow.get("reason", "unavailable"))
    st.caption(f"Votes once a lean has held past {cfg.BOOK_IMBALANCE_THRESHOLD:+.2f} for "
               f"{cfg.BOOK_PERSISTENCE_SECONDS:.0f}s. {book_snapshot['samples']} samples over "
               f"{book_snapshot['history_seconds']:.0f}s.")
    rows = [{"Venue": v, "Mid": f"${m['mid']:,.2f}",
             "Spread": f"{m['spread_bps']:.2f}bps" if m["spread_bps"] is not None else "--",
             "Imbalance": f"{m['imbalance']:+.2f}",
             "Microprice": f"{m['microprice_bps']:+.2f}bps" if m["microprice_bps"] is not None else "--",
             "Bid depth": f"${m['bid_notional']:,.0f}", "Ask depth": f"${m['ask_notional']:,.0f}"}
            for v, m in agg["venues"].items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for venue, reason in (agg.get("excluded") or {}).items():
        st.caption(f"⚠️ {venue} excluded from the aggregate: {reason}")
    held = [w for w in book_snapshot["walls"] if w["held"]]
    if held:
        st.markdown(f"**Resting walls that have held** ({cfg.BOOK_WALL_MIN_AGE_SECONDS:.0f}s+; can still be pulled)")
        st.dataframe(pd.DataFrame([
            {"Venue": w.get("venue", "--"), "Side": "bid (below)" if w["side"] == "bid" else "ask (above)",
             "Price": f"${w['price']:,.2f}", "Distance": f"{w['distance_bps']:.0f}bps",
             "Notional": f"${w['notional']:,.0f}", "vs typical level": f"{w['multiple']:.0f}x",
             "Held": f"{w['age_seconds']:.0f}s"} for w in held[:8]]), width="stretch", hide_index=True)
    else:
        st.caption("No resting walls have held long enough to report.")


def render_feeds_tab(cfg, feed_snapshot):
    if not cfg.ENABLE_EXCHANGE_FEEDS:
        st.caption("Multi-exchange feeds are disabled (ENABLE_EXCHANGE_FEEDS = False in config.py).")
        return
    if feed_snapshot is None:
        st.caption("Feeds not available for this pair.")
        return
    _venue_status(feed_snapshot["status"])
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.metric(f"Buy volume ({feed_snapshot['window_seconds']}s)", f"${feed_snapshot['buy_notional']:,.0f}")
    with wc2:
        st.metric(f"Sell volume ({feed_snapshot['window_seconds']}s)", f"${feed_snapshot['sell_notional']:,.0f}")
    with wc3:
        st.metric("Aggregated delta", f"{feed_snapshot['delta_ratio']:+.2f}")
    whale_col, liq_col = st.columns(2)
    for col, title, key, cols in (
            (whale_col, _esc(f"**Recent whale trades** (≥ ${cfg.WHALE_TRADE_USD_THRESHOLD:,.0f})"),
             "whale_events", ["time", "exchange", "side", "notional", "price"]),
            (liq_col, "**Recent liquidations** (Binance/Bybit USDT perps)", "liquidation_events",
             ["time", "exchange", "liquidated_side", "notional", "price"])):
        with col:
            st.markdown(title)
            events = feed_snapshot[key]
            if events:
                ev = pd.DataFrame(events[::-1][:15])
                ev["time"] = pd.to_datetime(ev["time"], unit="s").dt.strftime("%H:%M:%S")
                ev["notional"] = ev["notional"].map(lambda v: f"${v:,.0f}")
                st.dataframe(ev[cols], width="stretch", hide_index=True)
            else:
                st.caption("None yet this session.")


def render_alerts_tab(cfg):
    if not os.path.isfile(cfg.ALERT_LOG_PATH):
        st.caption("No alerts logged yet -- they'll show up here once enough indicators align.")
        return
    try:
        log_df = pd.read_csv(cfg.ALERT_LOG_PATH)
    except pd.errors.ParserError as e:
        st.error(f"{cfg.ALERT_LOG_PATH} looks corrupted and can't be parsed ({e}). This can happen if the "
                 "file was edited by hand or if an older, already-fixed bug wrote an inconsistent number "
                 "of columns across rows. Deleting the file is the simplest fix -- a fresh one will be "
                 "created on the next alert.")
    else:
        st.dataframe(log_df.tail(25).iloc[::-1], width="stretch", hide_index=True)


def render_paper_tab(cfg, kalshi_live, kalshi_result, snapshot, live_price):
    # Settle first, so a contract that resolved since the last refresh is already in the numbers.
    paper_trading.settle_due(cfg)
    quote_ok = bool(kalshi_live and kalshi_live["available"])
    yes_price = paper_trading.fill_price("yes", kalshi_live["yes_bid"], kalshi_live["yes_ask"]) if quote_ok else None
    no_price = paper_trading.fill_price("no", kalshi_live["yes_bid"], kalshi_live["yes_ask"]) if quote_ok else None
    fresh_market = lambda: kalshi_api.live_market(cfg, force=True)  # noqa: E731  (the shared poll, refreshed if not paused)

    oc1, oc2, oc3 = st.columns([1, 1, 1])
    with oc1:
        paper_contracts = st.number_input("Contracts", min_value=1, step=1, value=1, key="paper_contracts")
    with oc2:
        buy_yes = st.button(f"Buy YES @ {yes_price * 100:.1f}¢" if yes_price else "Buy YES (no quote)",
                            key="paper_buy_yes", disabled=yes_price is None, width="stretch")
    with oc3:
        buy_no = st.button(f"Buy NO @ {no_price * 100:.1f}¢" if no_price else "Buy NO (no quote)",
                           key="paper_buy_no", disabled=no_price is None, width="stretch")
    if not quote_ok:
        st.caption("Paper trading needs the live KXBTC15M quote: turn on Auto-track, on BTC-USD, "
                   "with Kalshi reachable.")
    elif buy_yes or buy_no:
        fresh = fresh_market()
        model_p = None
        if kalshi_result["available"] and fresh.get("available") and kalshi_result["strike"] == fresh["strike"]:
            model_p = kalshi_result["p_yes"]
        placed = paper_trading.place_order(cfg, fresh, "yes" if buy_yes else "no", int(paper_contracts),
                                           spot=live_price, model_p_yes=model_p, composite=snapshot.composite)
        if placed["ok"]:
            t = placed["trade"]
            st.success(_esc(f"Bought {t['contracts']} {t['side'].upper()} on `{t['ticker']}` at "
                            f"{t['price'] * 100:.1f}¢ + ${t['fee']:.2f} fee = ${t['cost']:.2f}. "
                            f"Pays ${t['contracts']:.2f} if {t['side'].upper()} wins."))
        else:
            st.error(_esc(placed["reason"]))

    ac1, ac2, ac3 = st.columns([1, 1, 1])
    with ac1:
        auto_on = st.checkbox(
            "Auto-trade", value=cfg.PAPER_AUTO_ENABLED, key="paper_auto",
            help=f"Enters one position per market, from {cfg.PAPER_AUTO_ENTRY_MINUTES:g} down to "
                 f"{cfg.PAPER_AUTO_LAST_MINUTES:g} minutes before close, on the first refresh where the "
                 "rule fires. Only trades while this page is open and refreshing; both rules are untested "
                 "ideas, not recommendations.")
    with ac2:
        auto_strategy = st.selectbox(
            "Auto rule", paper_trading.STRATEGIES,
            index=paper_trading.STRATEGIES.index(cfg.PAPER_AUTO_STRATEGY)
            if cfg.PAPER_AUTO_STRATEGY in paper_trading.STRATEGIES else 0,
            key="paper_auto_strategy", label_visibility="collapsed",
            help="composite: follow the bot's composite signal when it is strong enough to alert. "
                 "model: trade toward the Kalshi fair-value model when its edge beats the fee. "
                 "decision: the walk-forward-selected rule shown on the dashboard (no rule = no trades).")
    with ac3:
        auto_contracts = st.number_input("Auto contracts", min_value=1, step=1, value=cfg.PAPER_AUTO_CONTRACTS,
                                         key="paper_auto_contracts", label_visibility="collapsed")
    if auto_on and not quote_ok:
        st.caption("Auto-trade is waiting for the live KXBTC15M quote (Auto-track on, BTC-USD).")
    elif auto_on:
        auto_model_p = None
        if kalshi_result["available"] and kalshi_result["strike"] == kalshi_live["strike"]:
            auto_model_p = kalshi_result["p_yes"]
        step = paper_trading.auto_step(
            cfg, kalshi_live, fresh_market, auto_strategy, int(auto_contracts),
            composite=snapshot.composite, strength=snapshot.strength,
            alert_threshold=cfg.COMPOSITE_ALERT_THRESHOLD, model_p_yes=auto_model_p, spot=live_price)
        if step["action"] == "traded":
            t = step["trade"]
            st.success(_esc(f"Auto ({auto_strategy}): bought {t['contracts']} {t['side'].upper()} on "
                            f"`{t['ticker']}` at {t['price'] * 100:.1f}¢ + ${t['fee']:.2f} fee -- {step['reason']}."))
        else:
            st.caption(_esc(f"Auto ({auto_strategy}): {step['reason']}."))

    paper_trades = paper_trading.load_trades(cfg.PAPER_TRADE_LOG_PATH)
    s = paper_trading.summarize(paper_trades, cfg.PAPER_MIN_SAMPLE, cfg.PAPER_NOISE_Z)
    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1:
        st.metric("Net P&L", f"${s['net_pnl']:+,.2f}",
                  f"{s['roi'] * 100:+.1f}% on ${s['staked']:,.2f}" if s["roi"] is not None else None)
    with pm2:
        st.metric("Won / lost", f"{s['wins']} / {s['losses']}", f"{s['n_open']} open" if s["n_open"] else None,
                  delta_color="off")
    with pm3:
        st.metric("Win rate", f"{s['win_rate'] * 100:.1f}%" if s["win_rate"] is not None else "--",
                  f"break-even {s['breakeven_rate'] * 100:.1f}%" if s["breakeven_rate"] is not None else None,
                  delta_color="off")
    with pm4:
        st.metric("Fees paid", f"${s['fees']:,.2f}")
    if s["n_settled"]:
        st.caption(_esc(f"{paper_trading.VERDICT_TEXT[s['verdict']]} ({s['n_markets']} settled market(s); win rate "
                        f"±{s['win_rate_se'] * 100:.1f} pts, 1 SE.)"))
    by_source = paper_trading.summarize_by_source(paper_trades, cfg.PAPER_MIN_SAMPLE, cfg.PAPER_NOISE_Z)
    if len(by_source) > 1 or any(src != paper_trading.MANUAL for src in by_source):
        st.dataframe(pd.DataFrame([
            {"placed by": src, "settled": ss["n_settled"], "open": ss["n_open"], "net P&L": f"${ss['net_pnl']:+,.2f}",
             "win rate": f"{ss['win_rate'] * 100:.1f}%" if ss["win_rate"] is not None else "",
             "break-even": f"{ss['breakeven_rate'] * 100:.1f}%" if ss["breakeven_rate"] is not None else "",
             "read": paper_trading.VERDICT_TEXT[ss["verdict"]]}
            for src, ss in by_source.items()]), width="stretch", hide_index=True)
    if paper_trades:
        now_utc = paper_trading.utcnow()

        def _row(t):
            closes = paper_trading.parse_time(t["close_time"])
            if t["status"] == "settled":
                state_text = "WON" if t["side"] == t["outcome"] else "LOST"
            else:
                state_text = "awaiting result" if closes and closes <= now_utc else "open"
            return {"opened (UTC)": (paper_trading.parse_time(t["opened_at"]) or now_utc).strftime("%m-%d %H:%M:%S"),
                    "contract": t["ticker"], "by": t["source"], "side": t["side"].upper(), "qty": t["contracts"],
                    "paid": f"{t['price'] * 100:.1f}¢", "fee": f"${t['fee']:.2f}", "state": state_text,
                    "P&L": f"${t['pnl']:+,.2f}" if t["pnl"] is not None else ""}
        st.dataframe(pd.DataFrame([_row(t) for t in reversed(paper_trades[-50:])]), width="stretch", hide_index=True)
    else:
        st.caption("No paper trades yet -- pick a size and buy a side above.")
    st.caption(_esc(paper_trading.LIMITS_NOTE))


def render_notes_tab(cfg, state, kalshi_result, book_snapshot):
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
        st.caption("**On the model-vs-market gap:** " + _esc(MODEL_VS_MARKET_NOTE) + " A gap is far more often "
                   "this model's own approximations above (Coinbase as a proxy for the index, minute-level "
                   "data) than a mispricing.")
        st.caption("**Kalshi quote:** read from the public orderbook endpoint, one poll shared by every tab at an "
                   "adaptive interval. A quote older than "
                   f"{cfg.KALSHI_MAX_QUOTE_AGE_SECONDS:g}s forces NO TRADE. The market record's own quote lags "
                   "the book (DASHBOARD_NOTES.md), so it is not used for prices.")
        st.caption("**Outlook** is commentary on the current indicator pattern, not a prediction and not "
                   "financial advice.")
        path = _strike_path_liquidity(book_snapshot, kalshi_result, cfg)
        if path:
            st.markdown("**Resting liquidity between spot and the strike**")
            st.caption(_esc(f"${path['notional']:,.0f} of resting {path['side']}s sits between ${path['low']:,.2f} "
                            f"and ${path['high']:,.2f} on {path['venue']}. Price would have to trade "
                            f"{path['direction_word']} that to settle {path['settles']}."))
            for wall in path["walls"][:3]:
                st.caption(_esc(f"• ${wall['notional']:,.0f} concentrated at ${wall['price']:,.2f} "
                                f"(held {wall['age_seconds']:.0f}s)"))
            st.caption("**This does not adjust P(YES), and shouldn't be read as if it did.** It's one venue's "
                       "resting size at one instant. Resting size can be pulled the moment price approaches, it "
                       "says nothing about how much aggressive flow is coming the other way, and there's no "
                       "historical order-book data to calibrate what any of it is worth. Context, not an edge.")
    with note_right:
        st.markdown("**Order-book signals have no backtest**")
        st.caption(
            "⚠️ Neither the order-book signal nor book flow can be backtested. No exchange publishes "
            "historical level-2 snapshots, so unlike the candle indicators there is no hit rate behind either "
            "of them -- they always carry weight 1.00 in weighted conviction, and nothing there is a measured "
            "edge. It is a better-constructed *description* of the book, which is not the same claim.")
        st.caption(
            "**Why venues are averaged, not merged:** Coinbase quotes BTC-USD while Binance.US and Bybit quote "
            "BTCUSDT. USDT is not a dollar and the basis drifts by tens of basis points, so pooling those price "
            "levels would build a book with a fictional mid. Each venue's imbalance is measured against its own "
            "mid, so the ratios stay comparable whatever the basis is doing. Only walls that have held are "
            "listed in the Order Book tab, and even those can be pulled.")
        st.caption("**Feeds:** trades from Coinbase, Binance and Bybit. Liquidations are Binance and Bybit "
                   "USDT-perpetual futures (not spot) -- Coinbase spot has no liquidations.")
        st.markdown("**Per-indicator weights**")
        breakdown = state.weighted.get("breakdown", {})
        details = weighting.compute_weight_details(state.backtest_results, cfg)

        def _why(name, vote):
            d = details.get(name, {}).get(vote)
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
            st.dataframe(pd.DataFrame([
                {"Indicator": LONG_LABELS.get(n, n), "Vote": v["vote"].upper(), "Weight": f"{v['weight']:.2f}",
                 "Why": _why(n, v["vote"])} for n, v in breakdown.items()]), width="stretch", hide_index=True)
        else:
            st.caption("Nothing directional to weigh right now.")
        st.caption(
            "Weight 1.00 = counts the same as unweighted: no backtest data, too few independent "
            f"samples, or an edge over the base rate within {cfg.INDICATOR_WEIGHT_MIN_Z:.0f} "
            "standard errors of zero (noise). Above 1.00 = this indicator's calls have beaten "
            "the window's base rate for that direction; below 1.00 = done worse than it. Order book and book flow "
            "always show 1.00 (no historical depth data exists). Aggregated exchange delta shows 1.00 until "
            "`python backtest_agg_delta.py` has been run, then a Binance.US-only estimate.")


def header_line(product_id, refresh_seconds):
    st.markdown(f"#### ₿ {product_id}<span style='font-weight:400;font-size:0.75rem;opacity:0.6'>"
                f" &nbsp;·&nbsp; {datetime.now().strftime('%H:%M:%S')} &nbsp;·&nbsp; page refresh "
                f"{refresh_seconds}s</span>", unsafe_allow_html=True)
