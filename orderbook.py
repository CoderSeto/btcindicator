"""
Order-book metrics.

The original version of this module summed raw base-asset volume over a
fixed number of top-of-book levels and took the ratio. That has three
problems that matter once you're aggregating more than one exchange:

  1. **A "level" is not a unit of anything.** Coinbase quotes BTC-USD in
     $0.01 increments, so 25 levels can span well under a dollar. Bybit's
     spot book is coarser. "Top 25 levels" therefore covers a completely
     different slice of the book on each venue, which makes the numbers
     non-comparable and impossible to aggregate honestly.
  2. **Distance was ignored.** Size resting 1 basis point from mid and
     size resting 80 basis points away counted identically, even though
     only the first is going to interact with price on the timescale
     this bot cares about.
  3. **Base volume, not notional.** Summing `size` alone weights a level
     by coin count; two levels at meaningfully different prices are not
     the same amount of money. Notional (`price * size`) is what actually
     has to be absorbed.

Everything here is instead expressed in **basis points from mid** and in
**notional**, which are venue-independent, so per-venue readings can be
compared and combined (see `aggregate_metrics`). All functions are pure --
they take a normalized book dict and return numbers, with no network or
threading involved -- so they can be tested directly.

Book format used throughout:

    {"bids": [(price, size), ...],   # descending by price
     "asks": [(price, size), ...]}   # ascending by price

Use `normalize_book()` to get that shape from any of the raw exchange
payloads (Coinbase REST/WebSocket, Binance.US, Bybit).
"""
import math
from statistics import median

# Levels with a size at or below this are treated as absent. Exchanges use
# a literal "0" size to mean "this price level was removed"; floating point
# residue slightly above zero shows up occasionally too.
_MIN_SIZE = 1e-12


# --- Normalization ---------------------------------------------------------

def normalize_levels(raw_levels) -> list:
    """
    Turns a raw [[price, size, ...], ...] array (all three exchanges use
    string-encoded price/size, and Coinbase's REST book appends an order
    count as a third element) into a list of (float, float) tuples with
    zero/garbage levels dropped. Order is preserved, not enforced --
    `normalize_book` handles sorting.
    """
    out = []
    for level in raw_levels or []:
        try:
            price = float(level[0])
            size = float(level[1])
        except (TypeError, ValueError, IndexError):
            continue
        if price > 0 and size > _MIN_SIZE:
            out.append((price, size))
    return out


def normalize_book(raw_book: dict) -> dict:
    """
    Normalizes and sorts a raw book payload with "bids"/"asks" keys.
    Sorting is enforced here rather than trusted: Coinbase's REST book is
    already sorted, Bybit's snapshot is documented as sorted, but a book
    reassembled from incremental deltas (see orderbook_feeds) is not.
    """
    bids = sorted(normalize_levels((raw_book or {}).get("bids")), key=lambda l: -l[0])
    asks = sorted(normalize_levels((raw_book or {}).get("asks")), key=lambda l: l[0])
    return {"bids": bids, "asks": asks}


# --- Basic top-of-book quantities ------------------------------------------

def best_bid_ask(book: dict):
    """Returns ((bid_price, bid_size), (ask_price, ask_size)) or (None, None)."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return None, None
    return bids[0], asks[0]


def mid_price(book: dict):
    """Arithmetic mid, or None if either side of the book is empty."""
    bid, ask = best_bid_ask(book)
    if bid is None:
        return None
    return (bid[0] + ask[0]) / 2.0


def spread_bps(book: dict):
    """
    Bid/ask spread in basis points of mid, or None if the book is
    one-sided. Used as a data-quality gate: a book whose spread has blown
    out is usually mid-dislocation or a stale feed, and its imbalance
    reading shouldn't be trusted at face value.
    """
    bid, ask = best_bid_ask(book)
    if bid is None:
        return None
    mid = (bid[0] + ask[0]) / 2.0
    if mid <= 0:
        return None
    return (ask[0] - bid[0]) / mid * 10_000.0


def microprice(book: dict):
    """
    Size-weighted mid ("microprice"): the best quote is pulled toward the
    side with LESS resting size, because that's the side that gets cleared
    first. With 10 BTC bid and 1 BTC offered, the true clearing price sits
    much closer to the ask than the arithmetic mid suggests.

        microprice = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)

    Note the cross-multiplication -- the bid price is weighted by the ASK
    size. This is the standard definition and the sign is easy to get
    backwards. Returns None for a one-sided book.
    """
    bid, ask = best_bid_ask(book)
    if bid is None:
        return None
    bid_px, bid_sz = bid
    ask_px, ask_sz = ask
    total = bid_sz + ask_sz
    if total <= 0:
        return None
    return (bid_px * ask_sz + ask_px * bid_sz) / total


def microprice_bps(book: dict):
    """
    Microprice expressed as a signed offset from mid, in basis points.
    Positive = leaning toward the ask (upward pressure at the touch),
    negative = leaning toward the bid. Returns None if unavailable.
    """
    micro = microprice(book)
    mid = mid_price(book)
    if micro is None or not mid:
        return None
    return (micro - mid) / mid * 10_000.0


# --- Banded, distance-weighted depth ---------------------------------------

def _level_weight(distance_bps: float, decay_bps: float) -> float:
    """
    Exponential decay by distance from mid: weight = exp(-d / decay_bps).
    At d = decay_bps a level counts for ~37% of a level sitting at the
    touch; at 3x decay_bps, ~5%. `decay_bps <= 0` disables decay and
    weights everything inside the band equally (flat notional sum), which
    is what the tests use as the simple reference case.
    """
    if decay_bps <= 0:
        return 1.0
    return math.exp(-distance_bps / decay_bps)


def side_depth(levels, mid: float, band_bps: float, decay_bps: float, is_bid: bool) -> dict:
    """
    Sums one side of the book within `band_bps` of `mid`.

    Returns {"notional": raw notional inside the band,
             "weighted": distance-weighted notional,
             "levels": how many levels were inside the band}.

    Both sums are reported because they answer different questions:
    `notional` is "how much money is actually sitting there" (the right
    number for 'is there a wall between spot and my strike'), while
    `weighted` is "how much of it is close enough to matter right now"
    (the right number for a directional read).
    """
    notional = 0.0
    weighted = 0.0
    count = 0
    if not mid or mid <= 0:
        return {"notional": 0.0, "weighted": 0.0, "levels": 0}

    for price, size in levels:
        distance_bps = ((mid - price) if is_bid else (price - mid)) / mid * 10_000.0
        if distance_bps < 0:
            # A crossed or locked book (bid above mid on the ask side, etc).
            # Clamp rather than discard -- the size is real, it's just at
            # or through the touch.
            distance_bps = 0.0
        if distance_bps > band_bps:
            # Levels arrive sorted outward from the touch, so the first
            # level past the band means every later one is too.
            break
        value = price * size
        notional += value
        weighted += value * _level_weight(distance_bps, decay_bps)
        count += 1

    return {"notional": notional, "weighted": weighted, "levels": count}


def band_imbalance(book: dict, band_bps: float, decay_bps: float = 0.0) -> dict:
    """
    Distance-weighted bid/ask imbalance within `band_bps` of mid.

    Returns {"imbalance": float in [-1, 1] (+1 = all weight on the bid),
             "bid_notional": ..., "ask_notional": ...,
             "bid_weighted": ..., "ask_weighted": ...,
             "mid": float | None, "band_bps": float}

    `imbalance` is computed from the WEIGHTED sums; the raw notionals are
    returned alongside for display and for the flow calculation, which
    wants unweighted money.
    """
    mid = mid_price(book)
    bid = side_depth(book.get("bids") or [], mid, band_bps, decay_bps, is_bid=True)
    ask = side_depth(book.get("asks") or [], mid, band_bps, decay_bps, is_bid=False)

    total = bid["weighted"] + ask["weighted"]
    imbalance = (bid["weighted"] - ask["weighted"]) / total if total > 0 else 0.0

    return {
        "imbalance": imbalance,
        "bid_notional": bid["notional"],
        "ask_notional": ask["notional"],
        "bid_weighted": bid["weighted"],
        "ask_weighted": ask["weighted"],
        "bid_levels": bid["levels"],
        "ask_levels": ask["levels"],
        "mid": mid,
        "band_bps": band_bps,
    }


# --- Liquidity walls -------------------------------------------------------

def detect_walls(book: dict, band_bps: float, multiple: float, min_notional: float) -> list:
    """
    Finds outlier levels inside the band -- a single price holding far
    more money than its neighbours. These are the levels price visibly
    stalls at, and (on the ask side, above spot) the ones that matter most
    for a "will it reach the strike" question.

    A level qualifies when its notional is at least `multiple` times the
    MEDIAN level notional inside the band on the same side, and is at
    least `min_notional` in absolute terms. The median (not the mean) is
    the comparison point deliberately: the mean is dragged upward by the
    very outlier being tested for, so a big enough wall would suppress its
    own detection.

    Returns a list of {"side": "bid"/"ask", "price", "size", "notional",
    "distance_bps", "multiple"}, largest notional first. Note that this
    sees a single snapshot -- it cannot tell a real wall from a spoof that
    will vanish when approached. `orderbook_feeds.BookAggregator` tracks
    how long each wall has persisted, which is the only evidence available
    on that question.
    """
    mid = mid_price(book)
    if not mid or mid <= 0:
        return []

    walls = []
    for side, levels, is_bid in (("bid", book.get("bids") or [], True),
                                 ("ask", book.get("asks") or [], False)):
        in_band = []
        for price, size in levels:
            distance_bps = ((mid - price) if is_bid else (price - mid)) / mid * 10_000.0
            if distance_bps > band_bps:
                break
            in_band.append((price, size, max(distance_bps, 0.0), price * size))

        if len(in_band) < 3:
            # Too few levels for "unusually large compared to its
            # neighbours" to mean anything.
            continue

        typical = median(n for (_, _, _, n) in in_band)
        if typical <= 0:
            continue

        for price, size, distance_bps, notional in in_band:
            if notional >= min_notional and notional >= multiple * typical:
                walls.append({
                    "side": side,
                    "price": price,
                    "size": size,
                    "notional": notional,
                    "distance_bps": distance_bps,
                    "multiple": notional / typical,
                })

    walls.sort(key=lambda w: -w["notional"])
    return walls


def notional_between(book: dict, low: float, high: float, side: str) -> float:
    """
    Total resting notional on `side` ("bid" or "ask") between the prices
    `low` and `high` (inclusive). Answers the literal question "how much
    money is sitting in the path between here and that price" -- used for
    the Kalshi strike readout, where the strike is often outside any
    sensible basis-point band.

    This is a description of what's resting there right now, nothing more.
    Resting size is not a barrier: it can be pulled before price arrives,
    and it says nothing about how much aggressive flow is coming.
    """
    if low > high:
        low, high = high, low
    levels = book.get("bids" if side == "bid" else "asks") or []
    return sum(p * s for (p, s) in levels if low <= p <= high)


# --- Whole-book metrics ----------------------------------------------------

def book_metrics(book: dict, cfg) -> dict:
    """
    The full single-venue read, bundling everything above at the three
    configured bands.

    `near` (a few bps) is the executable book -- what an order hitting the
    market right now would actually run into. `mid_band` is the working
    band the directional signal is taken from. `far` is context: a strong
    near reading that the far band contradicts is a thinner, more easily
    faked setup than one where both agree.

    Returns {"available": False, "reason": ...} for a one-sided or empty
    book rather than silently reporting zeros, so callers can tell
    "balanced" apart from "no data".
    """
    mid = mid_price(book)
    if mid is None:
        return {"available": False, "reason": "one-sided or empty book"}

    near = band_imbalance(book, cfg.BOOK_BAND_BPS_NEAR, cfg.BOOK_DECAY_BPS)
    working = band_imbalance(book, cfg.BOOK_BAND_BPS, cfg.BOOK_DECAY_BPS)
    far = band_imbalance(book, cfg.BOOK_BAND_BPS_FAR, cfg.BOOK_DECAY_BPS)

    bid, ask = best_bid_ask(book)

    return {
        "available": True,
        "mid": mid,
        "best_bid": bid[0],
        "best_ask": ask[0],
        "spread_bps": spread_bps(book),
        "microprice_bps": microprice_bps(book),
        "imbalance": working["imbalance"],
        "imbalance_near": near["imbalance"],
        "imbalance_far": far["imbalance"],
        "bid_notional": working["bid_notional"],
        "ask_notional": working["ask_notional"],
        "bid_weighted": working["bid_weighted"],
        "ask_weighted": working["ask_weighted"],
        "levels": working["bid_levels"] + working["ask_levels"],
        "walls": detect_walls(book, cfg.BOOK_BAND_BPS_FAR,
                              cfg.BOOK_WALL_MULTIPLE, cfg.BOOK_WALL_MIN_NOTIONAL),
        "band_bps": cfg.BOOK_BAND_BPS,
    }


# --- Cross-venue aggregation -----------------------------------------------

def aggregate_metrics(per_venue: dict, cfg) -> dict:
    """
    Combines per-venue `book_metrics` results into one aggregated read.

    **Why this averages ratios instead of merging the books.** The obvious
    approach -- concatenate every venue's levels into one big book and run
    the same math over it -- is wrong here, and quietly so. Coinbase
    quotes BTC-USD; Binance.US and Bybit quote BTCUSDT. USDT is not a
    dollar, and the basis between those pairs drifts by tens of basis
    points. Pooling their price levels builds a book whose "mid" is a
    fiction and whose near band mixes levels that are 2bps from mid on one
    venue with levels 30bps from mid on another. Every venue's imbalance,
    though, is a dimensionless ratio measured against ITS OWN mid, so
    those are directly comparable no matter what the basis is doing.

    Venues are weighted by their in-band notional, so a deep venue counts
    for more than a thin one -- but a venue with a blown-out spread
    (> cfg.BOOK_MAX_SPREAD_BPS) is dropped entirely, since that usually
    means a stale or dislocated feed rather than real information.

    `agreement` is the notional-weighted share of participating venues
    whose imbalance points the same way as the aggregate. It is reported
    separately, not folded into the imbalance number: three venues leaning
    the same way at 0.3 is a materially different picture from one venue
    at 0.9 dragging two flat ones along with it, and collapsing both into
    a single figure would hide exactly that difference.
    """
    usable = {}
    excluded = {}
    for venue, m in (per_venue or {}).items():
        if not m or not m.get("available"):
            excluded[venue] = (m or {}).get("reason", "no data")
            continue
        sp = m.get("spread_bps")
        if sp is not None and sp > cfg.BOOK_MAX_SPREAD_BPS:
            excluded[venue] = f"spread {sp:.1f}bps > {cfg.BOOK_MAX_SPREAD_BPS}bps"
            continue
        if (m.get("bid_notional", 0.0) + m.get("ask_notional", 0.0)) <= 0:
            excluded[venue] = "no depth inside band"
            continue
        usable[venue] = m

    if not usable:
        return {"available": False, "reason": "no usable venue books", "excluded": excluded,
                "venues": {}, "venue_count": 0}

    def weighted_mean(key):
        num = 0.0
        den = 0.0
        for m in usable.values():
            w = m["bid_notional"] + m["ask_notional"]
            v = m.get(key)
            if v is None:
                continue
            num += v * w
            den += w
        return (num / den) if den > 0 else 0.0

    imbalance = weighted_mean("imbalance")

    agree_w = 0.0
    total_w = 0.0
    for m in usable.values():
        w = m["bid_notional"] + m["ask_notional"]
        total_w += w
        if imbalance > 0 and m["imbalance"] > 0:
            agree_w += w
        elif imbalance < 0 and m["imbalance"] < 0:
            agree_w += w
    agreement = (agree_w / total_w) if total_w > 0 else 0.0

    walls = []
    for venue, m in usable.items():
        for wall in m.get("walls", []):
            walls.append({**wall, "venue": venue})
    walls.sort(key=lambda w: -w["notional"])

    return {
        "available": True,
        "imbalance": imbalance,
        "imbalance_near": weighted_mean("imbalance_near"),
        "imbalance_far": weighted_mean("imbalance_far"),
        "microprice_bps": weighted_mean("microprice_bps"),
        "agreement": agreement,
        "bid_notional": sum(m["bid_notional"] for m in usable.values()),
        "ask_notional": sum(m["ask_notional"] for m in usable.values()),
        "venues": {v: m for v, m in usable.items()},
        "venue_count": len(usable),
        "excluded": excluded,
        "walls": walls,
        "band_bps": cfg.BOOK_BAND_BPS,
    }


# --- Legacy single-snapshot entry point ------------------------------------

def compute_imbalance(order_book: dict, depth: int) -> float:
    """
    The original flat top-N-levels base-volume imbalance, kept because the
    bot still falls back to a single Coinbase REST book snapshot when the
    live depth feeds are disabled or haven't connected yet, and because
    `alerts_log.csv` has historical rows computed this way.

    Prefer `band_imbalance()` / `book_metrics()` for anything new -- see
    this module's docstring for why level counts and base volume are the
    wrong units.
    """
    bids = (order_book or {}).get("bids", [])[:depth]
    asks = (order_book or {}).get("asks", [])[:depth]

    bid_volume = sum(float(level[1]) for level in bids)
    ask_volume = sum(float(level[1]) for level in asks)

    total = bid_volume + ask_volume
    if total == 0:
        return 0.0

    return (bid_volume - ask_volume) / total
