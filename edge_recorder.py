"""
Forward recorder for the KXBTC15M edge hunt (EDGE_HYPOTHESES.md).

Once a second, writes one row joining everything the pre-registered
hypotheses need, as it stood at that moment:

  Kalshi   the open market's ticker, strike and close; the YES top of book
           from the public orderbook endpoint (YES ask = 1 - best NO bid,
           kalshi_api.parse_orderbook); the last trade; the age of the quote
           and the round trip the poll took (k_rtt_ms), so quote staleness
           can be judged with the recorder's own latency included.
  Coinbase spot from data_fetch.latest_price() (the ticker, never a candle:
           CLAUDE.md gotcha 19), with its source flagged.
  Depth    the aggregated level-2 read from orderbook_feeds.BookAggregator:
           imbalance, near imbalance, microprice, persistence, flow.
  Flow     exchange_feeds.TradeAggregator: rolling buy/sell notional and the
           delta ratio; liquidation notional by side over EDGE_LIQ_WINDOW_SECONDS.
  Model    kalshi.settlement_probability and crossing_probability, and every
           raw input they were given (spot, strike, seconds to close, annual
           vol, how many candles the vol used and when that window ended),
           tagged with MODEL_VERSION so a later change can't be mistaken for
           the one the row was scored with.

Rules (CLAUDE.md principles 4 and 5, gotchas 1 and 16): fixed schema, every
write through csv_utils.append_row, blank cells (never zero) where a source
was down, and an explicit `*_up` flag per source. A tick-to-tick interval
longer than EDGE_GAP_SECONDS writes a `gap` row, so a hole in the record is
visible in the record and never mistaken for a quiet market.

Kalshi prints go to a second log (EDGE_TRADES_PATH), one row per trade,
deduplicated by trade_id. They exist so a maker fill model could be judged
against what actually traded; EDGE_HYPOTHESES.md says why that is not enough.

`--capture` saves verbatim Kalshi payloads to tests/fixtures/ and exits. The
orderbook and trades parsers are untrusted until those files exist.

Key-free. Runs standalone (`python edge_recorder.py`) or inside the dashboard (`start_background`, sharing its quote poll,
ticker and feeds; config EDGE_RECORD_IN_DASHBOARD). Not wired into the alerts or the console bot.
"""
import argparse
import json
import pathlib
import threading
import time
from datetime import datetime, timezone

import config
import csv_utils
import data_fetch
import kalshi
import kalshi_api

MODEL_VERSION = "kalshi.settlement_probability/2026-11-08"

RECORD_FIELDS = [
    "row_kind", "ts_utc", "gap_seconds",
    "ticker", "strike", "close_time", "secs_to_close",
    "k_yes_bid", "k_yes_ask", "k_yes_bid_sz", "k_yes_ask_sz", "k_last",
    "k_quote_age_s", "k_rtt_ms", "k_up",
    "cb_price", "cb_source", "cb_up",
    "book_imb", "book_imb_near", "book_microprice_bps", "book_flow",
    "book_persist_s", "book_persist_dir", "book_venues_up", "book_up",
    "agg_delta", "agg_buy_notional", "agg_sell_notional", "agg_venues_up", "agg_up",
    "liq_long_notional", "liq_short_notional", "liq_up",
    "vol_ann", "vol_samples", "vol_candle_end",
    "p_settle", "p_point", "model_version",
]

TRADE_FIELDS = ["ts_utc", "ticker", "trade_id", "created_time", "yes_price", "count", "taker_side"]

FIXTURE_DIR = pathlib.Path(__file__).parent / "tests" / "fixtures"
ROTATE_CHECK_ROWS = 600            # check the file size about every ten minutes of ticks


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _fmt(x, digits=6):
    """Blank for None/NaN (principle 4); a plain number otherwise."""
    if x is None:
        return ""
    if isinstance(x, float):
        if x != x:
            return ""
        return f"{x:.{digits}f}".rstrip("0").rstrip(".") if digits else repr(x)
    return x


def _status_up(status: dict) -> int:
    return sum(1 for s in status.values() if s == "connected")


class Recorder:
    """
    One tick = one row. Every external source is an injected callable so the
    whole thing runs on fakes in tests; `run()` wires the real ones.

      fetch_market()              -> kalshi_api.current_btc15m_market() dict
      fetch_orderbook(ticker)     -> raw orderbook payload
      fetch_trades(ticker, min_ts)-> raw trades payload
      fetch_spot()                -> data_fetch.latest_price() dict
      fetch_candles()             -> completed 1-minute candle DataFrame (vol input)
      book_snapshot()             -> BookAggregator.snapshot() or None
      trade_snapshot()            -> TradeAggregator.snapshot() or None
      clock()                     -> unix seconds
    """

    def __init__(self, cfg, fetch_market, fetch_orderbook, fetch_trades, fetch_spot, fetch_candles,
                 book_snapshot=None, trade_snapshot=None, clock=time.time, log=print, fetch_quote=None):
        self.cfg = cfg
        self.fetch_market = fetch_market
        self.fetch_orderbook = fetch_orderbook
        self.fetch_trades = fetch_trades
        self.fetch_spot = fetch_spot
        self.fetch_candles = fetch_candles
        self.book_snapshot = book_snapshot or (lambda: None)
        self.trade_snapshot = trade_snapshot or (lambda: None)
        self.clock = clock
        self.log = log
        # Dashboard mode: a callable returning kalshi_api.live_market()'s dict (market + the shared poll's
        # top of book and its age). The recorder then reads the quote the dashboard already polls instead
        # of sending its own orderbook request.
        self.fetch_quote = fetch_quote
        self.market = None
        self._market_checked = None
        self._quote = None            # last good parsed book
        self._quote_ts = None
        self._vol = None              # {"annual_vol", "samples_used", "candle_end"}
        self._vol_checked = None
        self._trades_checked = None
        self._seen_trade_ids = set()
        self._last_tick = None
        self.rows_written = 0
        self.gaps_written = 0

    # -- sources ----------------------------------------------------------

    def _refresh_market(self, now):
        due = self._market_checked is None or now - self._market_checked >= self.cfg.EDGE_MARKET_REFRESH_SECONDS
        closed = self.market and self.market["close_time"].timestamp() <= now
        if not (due or closed):
            return
        self._market_checked = now
        try:
            m = self.fetch_market()
        except Exception as e:
            self.log(f"[edge] market fetch failed: {e}")
            return
        if not m or not m.get("available"):
            if self.market and self.market["close_time"].timestamp() <= now:
                self.market = None
            return
        if self.market is None or m["ticker"] != self.market["ticker"]:
            self._quote, self._quote_ts = None, None
            self._seen_trade_ids = set()
            self.log(f"[edge] market {m['ticker']} strike {m['strike']} closes {m['close_time'].isoformat()}")
        self.market = m

    def _poll_book(self, now):
        """Returns (quote_or_None, rtt_ms_or_None, up)."""
        if not self.market:
            return None, None, False
        if self.fetch_quote is not None:
            return self._shared_quote(now)
        t0 = self.clock()
        try:
            payload = self.fetch_orderbook(self.market["ticker"])
            quote = kalshi_api.parse_orderbook(payload)
        except kalshi_api.KalshiSchemaError:
            raise                                    # unrecognised schema: stop, don't record garbage
        except Exception as e:
            self.log(f"[edge] orderbook poll failed: {e}")
            return self._quote, None, False
        rtt_ms = (self.clock() - t0) * 1000.0
        if quote["yes_bid"] is None and quote["yes_ask"] is None:
            return quote, rtt_ms, True               # up, but an empty book
        self._quote, self._quote_ts = quote, now
        return quote, rtt_ms, True

    def _shared_quote(self, now):
        """(quote, None, up) from the dashboard's shared Kalshi poll; the quote's own age is what is recorded."""
        try:
            m = self.fetch_quote()
        except Exception as e:
            self.log(f"[edge] shared quote failed: {e}")
            return self._quote, None, False
        if (not m or not m.get("available") or m.get("ticker") != self.market["ticker"]
                or m.get("poll_error") or m.get("quote_age_seconds") is None):
            return self._quote, None, False
        quote = {"yes_bid": m.get("yes_bid"), "yes_ask": m.get("yes_ask"),
                 "yes_bid_size": m.get("yes_bid_size"), "yes_ask_size": m.get("yes_ask_size")}
        if quote["yes_bid"] is None and quote["yes_ask"] is None:
            return quote, None, True
        self._quote, self._quote_ts = quote, now - float(m["quote_age_seconds"])
        return quote, None, True

    def _poll_trades(self, now):
        if not self.market or self.fetch_trades is None:
            return
        if self._trades_checked is not None and now - self._trades_checked < self.cfg.EDGE_TRADES_REFRESH_SECONDS:
            return
        self._trades_checked = now
        try:
            payload = self.fetch_trades(self.market["ticker"], None)
            trades = kalshi_api.parse_trades(payload)
        except kalshi_api.KalshiSchemaError:
            raise
        except Exception as e:
            self.log(f"[edge] trades poll failed: {e}")
            return
        for t in trades:
            key = t["trade_id"] or (t["created_time"].isoformat(), t["yes_price"], t["count"], t["taker_side"])
            if key in self._seen_trade_ids:
                continue
            self._seen_trade_ids.add(key)
            csv_utils.append_row(self.cfg.EDGE_TRADES_PATH, TRADE_FIELDS, [
                _iso(now), t["ticker"] or self.market["ticker"], t["trade_id"] or "",
                t["created_time"].isoformat(timespec="milliseconds"),
                _fmt(t["yes_price"], 4), _fmt(t["count"], 4), t["taker_side"],
            ])

    def _refresh_vol(self, now):
        if self._vol_checked is not None and now - self._vol_checked < self.cfg.EDGE_VOL_REFRESH_SECONDS:
            return
        self._vol_checked = now
        try:
            df = self.fetch_candles()
            v = kalshi.realized_volatility(df, self.cfg.KALSHI_VOL_LOOKBACK_MINUTES, 60)
        except Exception as e:
            self.log(f"[edge] vol refresh failed: {e}")
            self._vol = None
            return
        if v.get("annual_vol") is None:
            self._vol = None
            return
        end = df["time"].iloc[-1]
        end_ts = end.timestamp() if hasattr(end, "timestamp") else float(end)
        self._vol = {"annual_vol": v["annual_vol"], "samples_used": v["samples_used"],
                     "candle_end": end_ts + 60.0}

    # -- rows -------------------------------------------------------------

    def _gap_row(self, now, gap):
        row = {k: "" for k in RECORD_FIELDS}
        row.update({"row_kind": "gap", "ts_utc": _iso(now), "gap_seconds": _fmt(gap, 3)})
        return row

    def build_row(self, now) -> dict:
        row = {k: "" for k in RECORD_FIELDS}
        row["row_kind"], row["ts_utc"] = "tick", _iso(now)
        row["model_version"] = MODEL_VERSION

        self._refresh_market(now)
        quote, rtt_ms, k_up = self._poll_book(now)
        row["k_up"] = int(k_up)
        m = self.market
        if m:
            secs = m["close_time"].timestamp() - now
            row.update({"ticker": m["ticker"], "strike": _fmt(m["strike"], 2),
                        "close_time": m["close_time"].isoformat(), "secs_to_close": _fmt(secs, 3),
                        "k_last": _fmt(m.get("last_price"), 4)})
            if quote is not None:
                row.update({"k_yes_bid": _fmt(quote["yes_bid"], 4), "k_yes_ask": _fmt(quote["yes_ask"], 4),
                            "k_yes_bid_sz": _fmt(quote["yes_bid_size"], 4),
                            "k_yes_ask_sz": _fmt(quote["yes_ask_size"], 4)})
            if self._quote_ts is not None:
                row["k_quote_age_s"] = _fmt(now - self._quote_ts, 3)
            row["k_rtt_ms"] = _fmt(rtt_ms, 1)
            self._poll_trades(now)

        try:
            spot = self.fetch_spot()
        except Exception as e:
            self.log(f"[edge] spot failed: {e}")
            spot = {"price": None, "source": None}
        price = spot.get("price")
        row["cb_up"] = int(price is not None)
        if price is not None:
            row["cb_price"], row["cb_source"] = _fmt(float(price), 2), spot.get("source") or ""

        book = self.book_snapshot()
        if book and book.get("available"):
            agg, pers, flow = book["aggregate"], book.get("persistence") or {}, book.get("flow") or {}
            row.update({"book_up": 1, "book_imb": _fmt(agg.get("imbalance")),
                        "book_imb_near": _fmt(agg.get("imbalance_near")),
                        "book_microprice_bps": _fmt(agg.get("microprice_bps"), 3),
                        "book_venues_up": _fmt(agg.get("venue_count"), 0),
                        "book_persist_s": _fmt(pers.get("seconds"), 1),
                        "book_persist_dir": pers.get("direction") or ""})
            if flow.get("available"):
                row["book_flow"] = _fmt(flow.get("value"))
        else:
            row["book_up"] = 0

        trades = self.trade_snapshot()
        if trades:
            up = _status_up(trades.get("status") or {})
            row["agg_venues_up"] = up
            if up and trades.get("total_notional", 0) > 0:
                row.update({"agg_up": 1, "agg_delta": _fmt(trades["delta_ratio"]),
                            "agg_buy_notional": _fmt(trades["buy_notional"], 2),
                            "agg_sell_notional": _fmt(trades["sell_notional"], 2)})
            else:
                row["agg_up"] = 0
            liq_up = any(s == "connected" for n, s in (trades.get("status") or {}).items() if "liq" in n)
            row["liq_up"] = int(liq_up)
            if liq_up:
                cutoff = now - self.cfg.EDGE_LIQ_WINDOW_SECONDS
                long_n = sum(e["notional"] for e in trades.get("liquidation_events", [])
                             if e["time"] >= cutoff and e["liquidated_side"] == "long")
                short_n = sum(e["notional"] for e in trades.get("liquidation_events", [])
                              if e["time"] >= cutoff and e["liquidated_side"] == "short")
                row["liq_long_notional"], row["liq_short_notional"] = _fmt(long_n, 2), _fmt(short_n, 2)
        else:
            row["agg_up"], row["liq_up"] = 0, 0

        self._refresh_vol(now)
        if self._vol:
            row.update({"vol_ann": _fmt(self._vol["annual_vol"]), "vol_samples": self._vol["samples_used"],
                        "vol_candle_end": _iso(self._vol["candle_end"])})
            if m and price is not None and secs > 0:
                mins = secs / 60.0
                row["p_settle"] = _fmt(kalshi.settlement_probability(float(price), m["strike"], mins,
                                                                     self._vol["annual_vol"]))
                row["p_point"] = _fmt(kalshi.crossing_probability(float(price), m["strike"], mins,
                                                                  self._vol["annual_vol"]))
        return row

    def tick(self, now=None) -> dict:
        now = self.clock() if now is None else now
        if self._last_tick is not None:
            gap = now - self._last_tick
            if gap > self.cfg.EDGE_GAP_SECONDS:
                g = self._gap_row(now, gap)
                csv_utils.append_row(self.cfg.EDGE_RECORD_PATH, RECORD_FIELDS, [g[k] for k in RECORD_FIELDS])
                self.gaps_written += 1
        self._last_tick = now
        row = self.build_row(now)
        if csv_utils.append_row(self.cfg.EDGE_RECORD_PATH, RECORD_FIELDS, [row[k] for k in RECORD_FIELDS]):
            self.rows_written += 1
        if self.rows_written % ROTATE_CHECK_ROWS == 0:
            max_mb = getattr(self.cfg, "EDGE_RECORD_MAX_MB", None)
            if max_mb and csv_utils.rotate_if_larger(self.cfg.EDGE_RECORD_PATH, int(max_mb * 1_000_000)):
                self.log(f"[edge] {self.cfg.EDGE_RECORD_PATH} passed {max_mb} MB: renamed aside, starting a new file")
        return row


# --- live wiring -----------------------------------------------------------

def _fetch_candles(cfg):
    def fetch():
        df = data_fetch.get_candles(cfg.PRODUCT_ID, 60, limit=cfg.KALSHI_VOL_LOOKBACK_MINUTES + 5)
        completed, _ = data_fetch.drop_forming_candle(df, 60)
        return completed
    return fetch


def build_live(cfg, with_feeds: bool = True):
    """The Recorder on the real sources, plus the feed handles to stop on exit."""
    handles = []
    book_snapshot = trade_snapshot = None
    if with_feeds:
        import exchange_feeds
        import orderbook_feeds
        trade_handle = exchange_feeds.start_feeds(cfg)
        book_handle = orderbook_feeds.start_book_feeds(cfg)
        handles += [trade_handle, book_handle]
        book_snapshot, trade_snapshot = book_handle.snapshot, trade_handle.snapshot
    rec = Recorder(
        cfg,
        fetch_market=kalshi_api.current_btc15m_market,
        fetch_orderbook=kalshi_api.get_orderbook,
        fetch_trades=lambda ticker, min_ts: kalshi_api.get_trades(ticker, min_ts=min_ts),
        fetch_spot=lambda: data_fetch.latest_price(cfg.PRODUCT_ID),
        fetch_candles=_fetch_candles(cfg),
        book_snapshot=book_snapshot,
        trade_snapshot=trade_snapshot,
    )
    return rec, handles


def start_background(cfg, fetch_quote, fetch_spot, book_snapshot=None, trade_snapshot=None, clock=time.time,
                     log=print):
    """
    Records from inside the dashboard process, on a daemon thread, until the process ends. The Kalshi
    quote, the ticker and the depth/flow snapshots all come from the dashboard's own shared sources
    (no extra feeds, no second orderbook poll). Kalshi prints and the 1-minute vol window are still
    fetched by the recorder itself, at their slow EDGE_* intervals. A tick that fails is logged and the
    next one goes on; an unrecognised trades payload switches only the trades log off.
    Returns (recorder, stop_event, thread).
    """
    rec = Recorder(cfg, fetch_market=fetch_quote, fetch_orderbook=None,
                   fetch_trades=lambda ticker, min_ts: kalshi_api.get_trades(ticker, min_ts=min_ts),
                   fetch_spot=fetch_spot, fetch_candles=_fetch_candles(cfg), book_snapshot=book_snapshot,
                   trade_snapshot=trade_snapshot, clock=clock, log=log, fetch_quote=fetch_quote)
    stop = threading.Event()

    def loop():
        log(f"[edge] recording to {cfg.EDGE_RECORD_PATH} every {cfg.EDGE_TICK_SECONDS}s while the dashboard runs")
        while not stop.is_set():
            started = time.time()
            try:
                rec.tick(started)
            except kalshi_api.KalshiSchemaError as e:
                log(f"[edge] Kalshi trades payload not recognised ({e}); trades log off, quotes continue")
                rec.fetch_trades = None
            except Exception as e:
                log(f"[edge] tick failed: {e}")
            stop.wait(max(0.0, cfg.EDGE_TICK_SECONDS - (time.time() - started)))

    thread = threading.Thread(target=loop, name="edge-recorder", daemon=True)
    thread.start()
    return rec, stop, thread


def run(cfg, stop_event: threading.Event = None):
    rec, handles = build_live(cfg)
    stop_event = stop_event or threading.Event()
    print(f"[edge] recording to {cfg.EDGE_RECORD_PATH} every {cfg.EDGE_TICK_SECONDS}s; Ctrl-C to stop")
    try:
        while not stop_event.is_set():
            started = time.time()
            try:
                rec.tick(started)
            except kalshi_api.KalshiSchemaError as e:
                print(f"[edge] STOPPING: Kalshi payload not recognised: {e}\n"
                      f"[edge] run `python edge_recorder.py --capture` and update kalshi_api's parsers.")
                raise
            except Exception as e:
                print(f"[edge] tick failed: {e}")
            stop_event.wait(max(0.0, cfg.EDGE_TICK_SECONDS - (time.time() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        for h in handles:
            h.stop()
        print(f"[edge] wrote {rec.rows_written} rows, {rec.gaps_written} gap rows")


def capture(out_dir: pathlib.Path = FIXTURE_DIR) -> list:
    """Saves verbatim Kalshi orderbook and trades payloads for the open market as test fixtures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    markets = kalshi_api.get_open_markets()
    if not markets:
        raise SystemExit("no open KXBTC15M market right now; try again in a minute")
    ticker = markets[0]["ticker"]
    written = []
    for name, payload in (("kalshi_orderbook.json", kalshi_api.get_orderbook(ticker)),
                          ("kalshi_trades.json", kalshi_api.get_trades(ticker))):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    stamp = out_dir / "kalshi_orderbook_capture.txt"
    stamp.write_text(f"{ticker} captured {datetime.now(timezone.utc).isoformat()} from {kalshi_api.BASE_URL}\n",
                     encoding="utf-8")
    written.append(stamp)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--capture", action="store_true",
                    help="save verbatim Kalshi orderbook/trades payloads to tests/fixtures/ and exit")
    args = ap.parse_args()
    if args.capture:
        for p in capture():
            print(f"wrote {p}")
    else:
        run(config)
