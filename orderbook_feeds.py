"""
Live level-2 order-book feeds across Coinbase, Binance.US and Bybit,
maintained as local books and sampled into a rolling history.

Why this exists
---------------
The bot's original order-book signal was one Coinbase REST snapshot per
polling cycle. That's a still photograph taken every 30 seconds of
something that changes many times a second, which limits it to "the book
looks lopsided right now" -- and a book looking lopsided in a single
frame is very close to noise. Resting size is placed and pulled
constantly, and a large chunk of what shows up in any one snapshot is
gone moments later.

Keeping a live local book instead makes three things possible that a
snapshot cannot answer at all:

  - **Persistence.** Has this lean actually held for the last 30 seconds,
    or did it appear between one blink and the next? This is the single
    biggest noise filter available.
  - **Flow.** Is resting size being *added* to the bid, or *pulled* from
    the ask? Both look identical in a snapshot -- you need two points in
    time to tell them apart -- and both tend to precede movement.
  - **Wall age.** A large level that has sat there for two minutes is
    evidence of something. One that appears when price approaches and
    vanishes before it arrives is a spoof. Age is the only thing that
    separates them, and a snapshot has no age.

A note on aggr.trade
--------------------
aggr.trade aggregates *trades*, not order books -- it has no order-book
view (the request for one, Tucsky/aggr issue #346, was closed without
implementation) and no public API for external consumers either way. So
there is no aggr.trade order book to read from. What it *does* do is
connect straight to each exchange's own public WebSocket from the
browser and merge the streams client-side, and that is exactly the
approach taken here, extended to depth: same public feeds, same
merge-locally idea, no third-party dependency and no API keys.

Sources for every message schema below, so none of it is guessed:
  - Coinbase Exchange `level2_batch`: unauthenticated (the plain `level2`
    channel has required authentication since 2023-08-01), batched every
    50ms, identical JSON schema to `level2`.
  - Binance `<symbol>@depth@100ms` diff stream plus a REST snapshot, per
    Binance's own "How to manage a local order book correctly" procedure.
    Binance.US mirrors binance.com's market-data API shape exactly (see
    exchange_feeds for why this project uses Binance.US at all).
  - Bybit v5 `orderbook.{depth}.{symbol}` on the spot stream: snapshot
    followed by deltas, size 0 = remove, `u == 1` = service restarted,
    re-seed from that message.

Run `python orderbook_feeds.py` standalone to sanity-check connectivity.
"""
import json
import math
import threading
import time
from collections import deque
from typing import Optional

import requests
import websocket

import orderbook as ob
from exchange_feeds import run_with_reconnect, to_binance_symbol, to_bybit_symbol

websocket.setdefaulttimeout(15)

_BINANCE_US_REST_BASE = "https://api.binance.us"
_BINANCE_SNAPSHOT_LIMIT = 5000
# After a failed book sync, wait this long before the stream may trigger another.
_BINANCE_RESYNC_COOLDOWN_SECONDS = 10.0


# --- Message parsers ------------------------------------------------------
# Pure functions over already-decoded messages, so they can be tested
# against the exchanges' own documented example payloads without a
# network connection -- same convention as exchange_feeds' trade parsers.

def parse_coinbase_l2(msg: dict) -> Optional[dict]:
    """
    Coinbase Exchange level2 / level2_batch.

    Snapshot:
        {"type": "snapshot", "product_id": "BTC-USD",
         "bids": [["10101.10", "0.45054140"]],
         "asks": [["10102.55", "0.57753524"]]}

    Update:
        {"type": "l2update", "product_id": "BTC-USD",
         "time": "2019-08-14T20:42:27.265Z",
         "changes": [["buy", "10101.80000000", "0.162567"]]}

    Returns {"kind": "snapshot", "bids": [...], "asks": [...]} or
    {"kind": "update", "changes": [(side, price, size), ...]} with side
    normalized to "bid"/"ask" ("buy"/"sell" in the raw payload refer to
    the resting side, not a taker direction). A size of 0 means the level
    was removed. Returns None for anything else on the socket.
    """
    kind = msg.get("type")
    if kind == "snapshot":
        return {"kind": "snapshot", "bids": msg.get("bids") or [], "asks": msg.get("asks") or []}
    if kind != "l2update":
        return None

    changes = []
    for change in msg.get("changes") or []:
        try:
            side = "bid" if change[0] == "buy" else "ask"
            changes.append((side, float(change[1]), float(change[2])))
        except (IndexError, TypeError, ValueError):
            continue
    return {"kind": "update", "changes": changes}


def parse_binance_depth_update(msg: dict) -> Optional[dict]:
    """
    Binance diff depth stream (`<symbol>@depth@100ms`):

        {"e": "depthUpdate", "E": 1672515782136, "s": "BNBBTC",
         "U": 157, "u": 160,
         "b": [["0.0024", "10"]], "a": [["0.0026", "100"]]}

    `U`/`u` are the first/last update IDs covered by the event; they're
    what makes gap detection possible, and are returned here as
    first_id/last_id.
    """
    if msg.get("e") != "depthUpdate":
        return None
    try:
        return {
            "first_id": int(msg["U"]),
            "last_id": int(msg["u"]),
            "bids": msg.get("b") or [],
            "asks": msg.get("a") or [],
        }
    except (KeyError, TypeError, ValueError):
        return None


def parse_bybit_orderbook(msg: dict) -> Optional[dict]:
    """
    Bybit v5 `orderbook.{depth}.{symbol}`:

        {"topic": "orderbook.50.BTCUSDT", "type": "snapshot",
         "ts": 1672304484978,
         "data": {"s": "BTCUSDT",
                  "b": [["16493.50", "0.006"]],
                  "a": [["16611.00", "0.029"]],
                  "u": 18521288, "seq": 7961638724},
         "cts": 1672304484976}

    `type` is "snapshot" or "delta". Per Bybit's docs a delta size of 0
    means the level was filled or cancelled, and `u == 1` means their
    orderbook service restarted and the message must be treated as a
    fresh snapshot -- both handled by the caller via the returned "kind".
    """
    if not str(msg.get("topic", "")).startswith("orderbook."):
        return None
    data = msg.get("data") or {}
    try:
        update_id = int(data.get("u", 0))
    except (TypeError, ValueError):
        update_id = 0

    # u == 1 is a service restart; Bybit sends it as a snapshot, but treat
    # it as one regardless of how `type` is labelled rather than trusting
    # the two fields to always agree.
    kind = "snapshot" if (msg.get("type") == "snapshot" or update_id == 1) else "update"
    return {
        "kind": kind,
        "bids": data.get("b") or [],
        "asks": data.get("a") or [],
        "update_id": update_id,
    }


# --- Local book -----------------------------------------------------------

class LocalBook:
    """
    A price -> size map per side, rebuilt from a snapshot and kept current
    by incremental updates. Not thread-safe on its own -- BookAggregator
    owns the lock that guards every instance.
    """

    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.update_id = None
        self.ready = False
        self.last_update_ts = 0.0

    def reset(self):
        self.bids = {}
        self.asks = {}
        self.update_id = None
        self.ready = False

    def _apply_levels(self, side_map: dict, raw_levels):
        for level in raw_levels or []:
            try:
                price = float(level[0])
                size = float(level[1])
            except (IndexError, TypeError, ValueError):
                continue
            if size <= 0:
                side_map.pop(price, None)
            else:
                side_map[price] = size

    def apply_snapshot(self, bids, asks, update_id=None, ts: Optional[float] = None):
        self.bids = {}
        self.asks = {}
        self._apply_levels(self.bids, bids)
        self._apply_levels(self.asks, asks)
        self.update_id = update_id
        self.ready = True
        self.last_update_ts = ts or time.time()

    def apply_side_updates(self, bids, asks, update_id=None, ts: Optional[float] = None):
        self._apply_levels(self.bids, bids)
        self._apply_levels(self.asks, asks)
        if update_id is not None:
            self.update_id = update_id
        self.last_update_ts = ts or time.time()

    def apply_changes(self, changes, ts: Optional[float] = None):
        """changes: iterable of (side, price, size) with side in {"bid","ask"}."""
        for side, price, size in changes:
            side_map = self.bids if side == "bid" else self.asks
            if size <= 0:
                side_map.pop(price, None)
            else:
                side_map[price] = size
        self.last_update_ts = ts or time.time()

    def materialize(self, retain_bps: Optional[float] = None) -> dict:
        """
        Turns the maps into the sorted (price, size) book format the
        `orderbook` module works on, keeping only levels within
        `retain_bps` of mid.

        The filter happens before the sort on purpose. Coinbase's level2
        feed carries the entire book -- tens of thousands of levels --
        and sorting all of it once a second would be wasteful, when
        everything past a few percent from mid is irrelevant here.
        Finding best bid/ask and filtering are both linear scans with
        small constants; only the few hundred surviving levels get
        sorted.
        """
        if not self.bids or not self.asks:
            return {"bids": [], "asks": []}

        best_bid = max(self.bids)
        best_ask = min(self.asks)
        mid = (best_bid + best_ask) / 2.0

        if retain_bps is None or mid <= 0:
            bid_items = self.bids.items()
            ask_items = self.asks.items()
        else:
            span = mid * retain_bps / 10_000.0
            low, high = mid - span, mid + span
            bid_items = [(p, s) for p, s in self.bids.items() if p >= low]
            ask_items = [(p, s) for p, s in self.asks.items() if p <= high]

        return {
            "bids": sorted(((p, s) for p, s in bid_items if s > 0), key=lambda l: -l[0]),
            "asks": sorted(((p, s) for p, s in ask_items if s > 0), key=lambda l: l[0]),
        }


# --- Aggregator -----------------------------------------------------------

class BookAggregator:
    """
    Holds one LocalBook per venue, samples the aggregated metrics on a
    fixed interval, and derives the time-dependent reads (persistence,
    flow, wall age) that a single snapshot can't provide.

    The WebSocket callback threads write into the books; the sampler
    thread reads them; the main thread (console loop or Streamlit rerun)
    calls snapshot(). One lock guards all shared state.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._books = {}
        self._status = {}
        self._history = deque()
        self._walls = {}
        self._latest_aggregate = None
        self._latest_books = {}
        self._unavailable_reason = "no venue order book has connected yet"
        self._last_sample_ts = 0.0
        self._frozen_venues = set()
        self._started_at = time.time()

    # -- writer side (called from feed threads) --
    #
    # Every mutation goes through one of these, so the lock discipline
    # lives in one place instead of being re-implemented (and eventually
    # got wrong) in each venue's callbacks. The compound Binance
    # operations in particular are methods rather than inline callback
    # code specifically so their sequence-gap handling can be unit tested
    # without a socket.

    def _book(self, venue: str) -> LocalBook:
        """Lock must already be held."""
        if venue not in self._books:
            self._books[venue] = LocalBook()
        return self._books[venue]

    def set_status(self, venue: str, status: str):
        with self._lock:
            self._status[venue] = status

    def reset_book(self, venue: str):
        with self._lock:
            self._book(venue).reset()

    def is_ready(self, venue: str) -> bool:
        with self._lock:
            return self._book(venue).ready

    def apply_snapshot(self, venue: str, bids, asks, update_id=None):
        with self._lock:
            self._book(venue).apply_snapshot(bids, asks, update_id=update_id)

    def apply_updates(self, venue: str, bids, asks, update_id=None):
        """Applies a side-keyed update, but only to an already-seeded book."""
        with self._lock:
            book = self._book(venue)
            if not book.ready:
                return False
            book.apply_side_updates(bids, asks, update_id=update_id)
            return True

    def apply_changes(self, venue: str, changes):
        with self._lock:
            book = self._book(venue)
            if not book.ready:
                return False
            book.apply_changes(changes)
            return True

    def apply_sequenced_update(self, venue: str, event: dict) -> str:
        """
        Binance's documented update procedure for an already-seeded book,
        as one atomic step. Returns:

          "applied"     -- the event was in sequence and has been applied
          "stale"       -- entirely covered by the snapshot already, ignore
          "gap"         -- events were missed; the book has been discarded
                           and the caller must re-seed from a new snapshot
          "not_seeded"  -- nothing to apply to yet

        Per Binance: `U` of the next event equals `u + 1` of the previous
        one, so `U > local_update_id + 1` means something was dropped, and
        the only correct response is to throw the book away rather than
        keep applying updates on top of a hole.
        """
        with self._lock:
            book = self._book(venue)
            if not book.ready:
                return "not_seeded"
            if book.update_id is not None:
                if event["last_id"] <= book.update_id:
                    return "stale"
                if event["first_id"] > book.update_id + 1:
                    book.reset()
                    return "gap"
            book.apply_side_updates(event["bids"], event["asks"], update_id=event["last_id"])
            return "applied"

    def seed_sequenced_book(self, venue: str, bids, asks, last_update_id: int, buffered: list) -> bool:
        """
        Steps 5-7 of Binance's local-order-book procedure: install the REST
        snapshot, drop buffered events it already covers, then replay what's
        left. Returns False (leaving the book unseeded) if the replay hits a
        sequence gap, which means the snapshot and the buffer don't meet and
        the whole sync has to be retried.
        """
        with self._lock:
            book = self._book(venue)
            book.apply_snapshot(bids, asks, update_id=last_update_id)
            for event in buffered:
                if event["last_id"] <= last_update_id:
                    continue
                if event["first_id"] > book.update_id + 1:
                    book.reset()
                    return False
                book.apply_side_updates(event["bids"], event["asks"], update_id=event["last_id"])
            return True

    # -- sampling --

    def sample(self) -> Optional[dict]:
        """
        Takes one reading of every connected venue's book, appends it to
        the rolling history, and updates wall ages. Called on a timer by
        the sampler thread; returns the aggregate it recorded (or None if
        no venue had a usable book yet).
        """
        cfg = self.cfg
        now = time.time()

        stale_after = getattr(cfg, "BOOK_STALE_SECONDS", None)
        with self._lock:
            # A ready book that has stopped changing is a frozen feed (D12): it
            # would keep voting with an old picture of the book until the socket
            # timed out. Leave it out, and say so in the snapshot.
            frozen = {venue for venue, lb in self._books.items()
                      if lb.ready and stale_after is not None and now - lb.last_update_ts > stale_after}
            self._frozen_venues = frozen
            books = {
                venue: lb.materialize(cfg.BOOK_RETAIN_BPS)
                for venue, lb in self._books.items()
                if lb.ready and venue not in frozen
            }

        per_venue = {venue: ob.book_metrics(b, cfg) for venue, b in books.items()}
        aggregate = ob.aggregate_metrics(per_venue, cfg)
        if not aggregate.get("available"):
            # Clear the cached read rather than leaving it in place. Every
            # venue dropping at once is exactly when a stale-but-plausible
            # book would be most misleading, and the old value would
            # otherwise sit there indefinitely with nothing marking it as
            # no longer live.
            with self._lock:
                self._latest_aggregate = None
                self._latest_books = {}
                self._unavailable_reason = aggregate.get("reason", "no usable venue book")
            return None

        sample = {
            "ts": now,
            "imbalance": aggregate["imbalance"],
            "imbalance_near": aggregate["imbalance_near"],
            "microprice_bps": aggregate["microprice_bps"],
            "bid_notional": aggregate["bid_notional"],
            "ask_notional": aggregate["ask_notional"],
            "agreement": aggregate["agreement"],
            "venue_count": aggregate["venue_count"],
            # Per-venue in-band notional, so flow can compare the same venues at
            # both ends of its window (see _flow).
            "venue_notional": {
                venue: (m["bid_notional"], m["ask_notional"])
                for venue, m in aggregate["venues"].items()
            },
        }

        with self._lock:
            self._history.append(sample)
            cutoff = now - cfg.BOOK_HISTORY_SECONDS
            while self._history and self._history[0]["ts"] < cutoff:
                self._history.popleft()
            self._latest_books = books
            self._latest_aggregate = aggregate
            self._last_sample_ts = now
            self._track_walls(aggregate.get("walls", []), now)

        return aggregate

    def _track_walls(self, walls, now: float):
        """
        Ages walls across samples. Called with the lock held.

        Matching is by price bucket rather than exact price so that a wall
        repricing by a tick reads as the same wall rather than a brand new
        one with its age reset to zero -- otherwise nothing would ever
        accumulate enough age to be reported as held.
        """
        cfg = self.cfg
        grace = self._max_sample_gap()
        # Expire first, against the time this sample was taken. Pruning only after
        # updating let a wall seen on both sides of an unsampled outage carry its
        # age straight across it.
        for key in [k for k, e in self._walls.items() if now - e["last_seen"] > grace]:
            del self._walls[key]

        seen = set()
        for wall in walls:
            # Buckets of a fixed width in basis points, i.e. fixed in log price.
            # (Sizing the bucket from the wall's own price, price / (price * bps),
            # gave every price the same key, so all walls on a side merged into
            # one entry and a brand-new wall inherited an old one's age.)
            bucket_bps = max(cfg.BOOK_WALL_MATCH_BPS, 1e-9)
            key = (wall.get("venue"), wall["side"],
                   round(math.log(wall["price"]) * 10_000.0 / bucket_bps))
            seen.add(key)
            entry = self._walls.get(key)
            if entry is None:
                self._walls[key] = {"first_seen": now, "last_seen": now, "wall": wall}
            else:
                entry["last_seen"] = now
                entry["wall"] = wall

        # A wall briefly absent from one sample (partially consumed, then
        # replenished) shouldn't lose its accumulated age, so entries are
        # only dropped after a few samples' grace -- checked above, before
        # this sample's sightings refresh anything.

    # -- derived reads --

    def _max_sample_gap(self) -> float:
        """
        The longest silence between two samples that still counts as continuous
        observation. Samples are only taken while some venue has a usable book, so
        a longer hole means nobody was watching -- an outage, or the machine asleep
        -- and whatever held before it can't be credited with holding through it.
        Same allowance the wall tracker uses for a briefly missing wall.
        """
        return max(3.0 * self.cfg.BOOK_SAMPLE_INTERVAL_SECONDS, 3.0)

    def _persistence(self, history) -> dict:
        """
        How long the aggregated imbalance has continuously held one side
        past BOOK_IMBALANCE_THRESHOLD, ending at the most recent sample.

        Returns {"seconds": float, "direction": "bullish"/"bearish"/None,
                 "mean_imbalance": float}. Direction is None whenever the
        current reading is inside the threshold -- a lean that isn't
        currently there has no persistence, however long it lasted before.
        """
        threshold = self.cfg.BOOK_IMBALANCE_THRESHOLD
        if not history:
            return {"seconds": 0.0, "direction": None, "mean_imbalance": 0.0}

        current = history[-1]["imbalance"]
        if abs(current) < threshold:
            return {"seconds": 0.0, "direction": None, "mean_imbalance": current}

        sign = 1 if current > 0 else -1
        end_ts = history[-1]["ts"]
        start_ts = end_ts
        run = []
        max_gap = self._max_sample_gap()
        for s in reversed(history):
            v = s["imbalance"]
            if run and start_ts - s["ts"] > max_gap:
                break  # a hole in the history: the run can't be traced across it
            if abs(v) >= threshold and (1 if v > 0 else -1) == sign:
                start_ts = s["ts"]
                run.append(v)
            else:
                break

        return {
            "seconds": end_ts - start_ts,
            "direction": "bullish" if sign > 0 else "bearish",
            "mean_imbalance": sum(run) / len(run) if run else current,
        }

    def _flow(self, history) -> dict:
        """
        Change in resting notional on each side over
        BOOK_FLOW_WINDOW_SECONDS, as

            value = log(bid_now / bid_then) - log(ask_now / ask_then)

        Positive = resting bids grew relative to resting asks (bids
        stacking and/or asks being pulled); negative = the reverse. Logs
        so that a doubling on one side and a halving on the other are
        equal and opposite rather than lopsided, and a ratio-of-ratios so
        that book-wide liquidity swings -- everyone widening at once --
        largely cancel instead of reading as direction.

        Honest caveat, and it's not a small one: the band is anchored to
        mid, so when price moves, the set of levels inside the band
        changes for mechanical reasons that have nothing to do with anyone
        adding or pulling size. Both sides shift together, which cancels
        much of it, but not all. Treat this as suggestive, never as a
        clean measurement of intent.
        """
        window = self.cfg.BOOK_FLOW_WINDOW_SECONDS
        if len(history) < 2:
            return {"available": False, "reason": "not enough history yet"}

        now = history[-1]
        target_ts = now["ts"] - window
        past = None
        for s in history:
            if s["ts"] <= target_ts:
                past = s
            else:
                break

        if past is None:
            elapsed = now["ts"] - history[0]["ts"]
            return {"available": False,
                    "reason": f"only {elapsed:.0f}s of book history, need {window:.0f}s"}

        # Only venues in the aggregate at BOTH ends count. Summing whatever was
        # usable at each end turns a venue dropping out (a disconnect, or the
        # spread gate during volatility) into "liquidity pulled" on whichever
        # side that venue was heavier on.
        past_bid, past_ask = past["bid_notional"], past["ask_notional"]
        now_bid, now_ask = now["bid_notional"], now["ask_notional"]
        if "venue_notional" in past and "venue_notional" in now:
            common = sorted(set(past["venue_notional"]) & set(now["venue_notional"]))
            if not common:
                return {"available": False,
                        "reason": "no venue was in the aggregate at both ends of the window"}
            past_bid = sum(past["venue_notional"][v][0] for v in common)
            past_ask = sum(past["venue_notional"][v][1] for v in common)
            now_bid = sum(now["venue_notional"][v][0] for v in common)
            now_ask = sum(now["venue_notional"][v][1] for v in common)

        if min(past_bid, past_ask, now_bid, now_ask) <= 0:
            return {"available": False, "reason": "a side of the book was empty in the window"}

        bid_ratio = now_bid / past_bid
        ask_ratio = now_ask / past_ask
        value = math.log(bid_ratio) - math.log(ask_ratio)

        return {
            "available": True,
            "value": value,
            "bid_change_pct": (bid_ratio - 1.0) * 100.0,
            "ask_change_pct": (ask_ratio - 1.0) * 100.0,
            "window_seconds": now["ts"] - past["ts"],
        }

    def _aged_walls(self, now: float) -> list:
        """Walls with their observed age attached, longest-held first."""
        cfg = self.cfg
        out = []
        for entry in self._walls.values():
            age = entry["last_seen"] - entry["first_seen"]
            out.append({
                **entry["wall"],
                "age_seconds": age,
                "held": age >= cfg.BOOK_WALL_MIN_AGE_SECONDS,
            })
        out.sort(key=lambda w: (-w["age_seconds"], -w["notional"]))
        return out

    # -- reader side --

    def snapshot(self) -> dict:
        """
        The current aggregated book read, for the console loop or a
        dashboard rerun. Everything time-dependent is derived here from
        the sampled history rather than recomputed from the live books,
        so a caller always sees a self-consistent set of numbers.
        """
        now = time.time()
        with self._lock:
            status = dict(self._status)
            history = list(self._history)
            aggregate = self._latest_aggregate
            books = dict(self._latest_books)
            walls = self._aged_walls(now)
            uptime = now - self._started_at
            last_sample_ts = self._last_sample_ts
            reason = self._unavailable_reason
            frozen = sorted(self._frozen_venues)

        # A sampler thread that has died would otherwise leave the last
        # good reading in place forever, looking live. Anything older than
        # a few sampling intervals is reported as no data rather than
        # handed on as though it were current.
        max_age = max(5.0 * self.cfg.BOOK_SAMPLE_INTERVAL_SECONDS, 10.0)
        if aggregate and (now - last_sample_ts) > max_age:
            aggregate = None
            reason = f"book read is {now - last_sample_ts:.0f}s stale -- sampler may have stopped"

        if not aggregate or not aggregate.get("available"):
            return {
                "available": False,
                "reason": reason,
                "status": status,
                "uptime_seconds": uptime,
                "samples": len(history),
                "frozen_venues": frozen,
            }

        persistence = self._persistence(history)
        flow = self._flow(history)
        age = (history[-1]["ts"] - history[0]["ts"]) if len(history) > 1 else 0.0

        return {
            "available": True,
            "aggregate": aggregate,
            "books": books,
            "status": status,
            "persistence": persistence,
            "flow": flow,
            "walls": walls,
            "samples": len(history),
            "history_seconds": age,
            "uptime_seconds": uptime,
            "sampled_at": history[-1]["ts"],
            # Venues connected but left out of the read because their book
            # stopped changing (config.BOOK_STALE_SECONDS).
            "frozen_venues": frozen,
        }


# --- WebSocket runners ----------------------------------------------------

def _coinbase_book_runner(agg: BookAggregator, product_id: str, stop_event: threading.Event, holder: dict):
    """
    Coinbase Exchange level2_batch: a full-book snapshot followed by
    incremental updates, batched every 50ms.

    level2_batch rather than level2 because the plain level2 channel has
    required authentication since 2023-08-01, and this project deliberately
    uses no API keys anywhere. The two channels carry the same JSON schema;
    level2_batch just coalesces updates, which is if anything preferable
    here -- the sampler only reads once a second.
    """
    name = "coinbase"
    url = "wss://ws-feed.exchange.coinbase.com"

    def on_open(ws):
        agg.set_status(name, "connecting")
        agg.reset_book(name)
        ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": [product_id],
            "channels": ["level2_batch"],
        }))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "error":
            agg.set_status(name, f"error: {msg.get('message', 'unknown')}")
            return
        parsed = parse_coinbase_l2(msg)
        if not parsed:
            return
        if parsed["kind"] == "snapshot":
            agg.apply_snapshot(name, parsed["bids"], parsed["asks"])
            agg.set_status(name, "connected")
        else:
            agg.apply_changes(name, parsed["changes"])

    def on_error(ws, error):
        agg.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        agg.reset_book(name)
        agg.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    run_with_reconnect(name, create_app, stop_event, agg, holder)


def _binance_book_runner(agg: BookAggregator, symbol: str, stop_event: threading.Event, holder: dict):
    """
    Binance.US diff depth stream plus a REST snapshot, following Binance's
    documented "How to manage a local order book correctly" procedure:
    buffer stream events first, then fetch the snapshot, discard buffered
    events already covered by it, and apply the rest -- checking
    first/last update IDs for gaps on every event afterwards.

    The partial-book stream (`@depth20`) would be far less code, since each
    message is self-contained. It isn't used because 20 levels of BTCUSDT
    span a fraction of a basis point, which is nowhere near the 25bp band
    the signal is read from -- it would look like a connected venue while
    contributing almost nothing to the measurement.
    """
    name = "binance_us"
    url = f"wss://stream.binance.us:9443/ws/{symbol.lower()}@depth@100ms"

    def create_app():
        # `syncing` is True from the moment a resync starts until the buffer it
        # collected has been fully replayed onto the new book. While it's True,
        # every stream event goes into the buffer -- never straight onto the book
        # -- so events reach the book in stream order even while the snapshot is
        # being installed. `retry_after` spaces out attempts after a failed sync.
        state = {"buffer": [], "syncing": False, "retry_after": 0.0}
        state_lock = threading.Lock()

        def drain(consumed: int) -> bool:
            """
            Replays buffered events past `consumed` until the buffer is empty,
            then hands the stream back to on_message by clearing `syncing` --
            under the same lock on_message checks, so no event can fall between
            the last replayed one and the first applied live. Returns False if
            the replay hit a gap (the book has been discarded).
            """
            while True:
                with state_lock:
                    extra = state["buffer"][consumed:]
                    if not extra:
                        state["buffer"] = []
                        state["syncing"] = False
                        return True
                    consumed += len(extra)
                for event in extra:
                    if agg.apply_sequenced_update(name, event) == "gap":
                        return False

        def resync():
            """Fetch a REST snapshot and splice the buffered events onto it."""
            with state_lock:
                if state["syncing"]:
                    return
                state["syncing"] = True
                state["buffer"] = []
            try:
                agg.set_status(name, "syncing book")
                agg.reset_book(name)

                # Let the stream buffer at least one event first, so step 4
                # of the procedure -- "is the snapshot newer than the first
                # buffered event?" -- has something to compare against.
                time.sleep(1.0)

                for _ in range(5):
                    if stop_event.is_set():
                        return
                    resp = requests.get(
                        f"{_BINANCE_US_REST_BASE}/api/v3/depth",
                        params={"symbol": symbol, "limit": _BINANCE_SNAPSHOT_LIMIT},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    snap = resp.json()
                    last_update_id = int(snap["lastUpdateId"])

                    with state_lock:
                        buffered = list(state["buffer"])
                        consumed = len(buffered)

                    # Step 4: a snapshot older than the first event we
                    # buffered leaves a hole between them -- refetch.
                    if buffered and last_update_id < buffered[0]["first_id"]:
                        time.sleep(1.0)
                        continue

                    if (agg.seed_sequenced_book(name, snap.get("bids"), snap.get("asks"),
                                                last_update_id, buffered)
                            and drain(consumed)):
                        agg.set_status(name, "connected")
                        return
                    time.sleep(1.0)

                agg.set_status(name, "error: could not sync book")
            except Exception as e:
                agg.set_status(name, f"error: {e}")
            finally:
                with state_lock:
                    if state["syncing"]:
                        # Didn't finish: leave nothing half-built, and hold off
                        # before the next attempt rather than hammering the
                        # REST endpoint (a 5000-level depth request is heavy).
                        agg.reset_book(name)
                        state["buffer"] = []
                        state["syncing"] = False
                        state["retry_after"] = time.time() + _BINANCE_RESYNC_COOLDOWN_SECONDS

        def start_resync():
            threading.Thread(target=resync, daemon=True).start()

        def on_open(ws):
            agg.set_status(name, "connecting")
            start_resync()

        def on_message(ws, message):
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                return
            event = parse_binance_depth_update(msg)
            if not event:
                return

            with state_lock:
                if state["syncing"]:
                    # A sync is in flight: hold the event so it's replayed in
                    # order once the snapshot lands.
                    state["buffer"].append(event)
                    return
                retry_after = state["retry_after"]

            result = agg.apply_sequenced_update(name, event)
            if result in ("applied", "stale"):
                return
            # "gap": the book was discarded. "not_seeded": an earlier sync failed
            # and nothing else would ever retry it -- without this the venue
            # stayed dark until the socket happened to reconnect.
            if result == "gap" or time.time() >= retry_after:
                start_resync()

        def on_error(ws, error):
            agg.set_status(name, f"error: {error}")

        def on_close(ws, code, reason):
            agg.reset_book(name)
            agg.set_status(name, "disconnected")

        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    run_with_reconnect(name, create_app, stop_event, agg, holder)


def _bybit_book_runner(agg: BookAggregator, symbol: str, depth: int,
                        stop_event: threading.Event, holder: dict):
    """
    Bybit v5 spot orderbook.{depth}.{symbol}. Self-seeding: the first
    message after subscribing is a snapshot, and Bybit re-sends one
    whenever its own state resets, so there's no REST bootstrap to do.
    """
    name = "bybit"
    url = "wss://stream.bybit.com/v5/public/spot"
    topic = f"orderbook.{depth}.{symbol}"

    def on_open(ws):
        agg.set_status(name, "connecting")
        agg.reset_book(name)
        ws.send(json.dumps({"op": "subscribe", "args": [topic]}))

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        parsed = parse_bybit_orderbook(msg)
        if not parsed:
            return
        if parsed["kind"] == "snapshot":
            agg.apply_snapshot(name, parsed["bids"], parsed["asks"], update_id=parsed["update_id"])
            agg.set_status(name, "connected")
        else:
            agg.apply_updates(name, parsed["bids"], parsed["asks"], update_id=parsed["update_id"])

    def on_error(ws, error):
        agg.set_status(name, f"error: {error}")

    def on_close(ws, code, reason):
        agg.reset_book(name)
        agg.set_status(name, "disconnected")

    def create_app():
        return websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                       on_error=on_error, on_close=on_close)

    run_with_reconnect(name, create_app, stop_event, agg, holder)


def _sampler_runner(agg: BookAggregator, stop_event: threading.Event):
    interval = agg.cfg.BOOK_SAMPLE_INTERVAL_SECONDS
    while not stop_event.is_set():
        try:
            agg.sample()
        except Exception as e:
            # A sampling bug must never kill the thread and silently
            # freeze every time-dependent reading at its last value.
            print(f"[orderbook_feeds] sampler error: {e}")
        stop_event.wait(interval)


class BookFeedHandle:
    """Handle to a running set of order-book feed threads."""

    def __init__(self, aggregator: BookAggregator, stop_event: threading.Event, holders: list):
        self.aggregator = aggregator
        self._stop_event = stop_event
        self._holders = holders

    def snapshot(self) -> dict:
        return self.aggregator.snapshot()

    def stop(self):
        self._stop_event.set()
        for holder in self._holders:
            ws = holder.get("ws")
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


def start_book_feeds(cfg) -> BookFeedHandle:
    """
    Starts a depth feed for every exchange that supports cfg.PRODUCT_ID,
    plus the sampler thread, and returns a handle to read from. Pairs with
    no Binance/Bybit equivalent (e.g. BTC-EUR) simply run Coinbase-only --
    the aggregation handles a single venue fine.
    """
    aggregator = BookAggregator(cfg)
    stop_event = threading.Event()
    holders = []

    def spawn(target, *args):
        holder = {}
        holders.append(holder)
        threading.Thread(target=target, args=(*args, holder), daemon=True).start()

    spawn(_coinbase_book_runner, aggregator, cfg.PRODUCT_ID, stop_event)

    binance_symbol = to_binance_symbol(cfg.PRODUCT_ID)
    if binance_symbol:
        spawn(_binance_book_runner, aggregator, binance_symbol, stop_event)

    bybit_symbol = to_bybit_symbol(cfg.PRODUCT_ID)
    if bybit_symbol:
        spawn(_bybit_book_runner, aggregator, bybit_symbol, cfg.BYBIT_BOOK_DEPTH, stop_event)

    threading.Thread(target=_sampler_runner, args=(aggregator, stop_event), daemon=True).start()

    return BookFeedHandle(aggregator, stop_event, holders)


if __name__ == "__main__":
    import config as _cfg

    print(f"Connecting to level-2 book feeds for {_cfg.PRODUCT_ID} -- watching for 45 seconds...\n")
    handle = start_book_feeds(_cfg)
    try:
        for _ in range(45):
            time.sleep(1)
            snap = handle.snapshot()
            if not snap["available"]:
                print(f"status={snap['status']}  ({snap['reason']})")
                continue
            a = snap["aggregate"]
            p = snap["persistence"]
            f = snap["flow"]
            flow_txt = f"{f['value']:+.2f}" if f.get("available") else "--"
            print(
                f"venues={a['venue_count']}  imb={a['imbalance']:+.2f} "
                f"(near {a['imbalance_near']:+.2f}, agree {a['agreement'] * 100:.0f}%)  "
                f"micro={a['microprice_bps']:+.2f}bps  "
                f"bid=${a['bid_notional']:,.0f} ask=${a['ask_notional']:,.0f}  "
                f"held={p['seconds']:.0f}s {p['direction'] or '--'}  "
                f"flow={flow_txt}  walls={sum(1 for w in snap['walls'] if w['held'])}"
            )
    finally:
        handle.stop()
    print("\nDone. A venue stuck on 'connecting'/'syncing book' or showing an error "
          "is most likely blocked by your network -- see the same note in exchange_feeds.py.")
