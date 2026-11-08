"""
Paper trading for Kalshi's KXBTC15M contracts: simulated orders against the
real market, settled by the real outcome. No money, no account, no auth -- it
reads the same public quote and result endpoints the rest of the bot does.

Why it is built the way it is (REVIEW_FINDINGS.md, "Is there an edge?"): the
study found no edge in these contracts after costs, and the costs are most of
the story. A paper account that fills at the mid and charges no fee would show
profits that a real account never sees. So a simulated order here:

  - buys YES at the ASK, or NO at 1 - the YES BID (crossing the spread, as a
    taker does -- no resting-order fills are assumed);
  - pays Kalshi's taker fee, from the same function the edge study uses
    (kalshi_study.order_fee: 0.07 * C * P * (1 - P), rounded up to the cent
    per order);
  - settles on Kalshi's own recorded result for the contract, not on
    Coinbase's price, so basis risk against the CF Benchmarks index is in the
    result, as it would be for a real position.

What it does NOT model, and the output says so: the size resting at the touch
(an order is assumed to fill in full at the quoted price, so keep sizes small),
latency between the quote and a real fill, and any price movement while you
decide. Results are also small-sample: summarize() reports the noise on the
win rate next to the break-even rate, so a lucky week isn't read as skill.

The ledger is an append-only CSV of EVENTS, written only through
csv_utils.append_row (gotcha #16): an "open" row when an order is placed and a
"settle" row when its contract resolves. A position's state is the join of the
two, so nothing is ever rewritten in place and two tabs can't corrupt it. A
fixed schema with blank cells for fields an event doesn't carry (gotcha #1).

Run standalone to settle what's due and print the report:

    python paper_trading.py
"""
import csv
import math
import os
import uuid
from datetime import datetime, timezone

import kalshi_api
import kalshi_study
import kalshi_tracking
from csv_utils import append_row

LEDGER_HEADER = [
    "event", "trade_id", "timestamp", "ticker", "side", "contracts", "price", "fee",
    "yes_bid", "yes_ask", "strike", "close_time", "spot", "model_p_yes", "composite",
    "outcome", "payout", "pnl",
]

SIDES = ("yes", "no")

# A contract that has closed but isn't resolved yet (Kalshi settles a few
# seconds to a few minutes after close) is re-checked at most this often.
SETTLE_RETRY_SECONDS = 20.0
_last_settle_attempt = {}   # ticker -> datetime of the last query; process-wide, like kalshi_api's poll cache


def utcnow() -> datetime:
    """The module's clock (a function so tests and the dashboard tests can move it)."""
    return datetime.now(timezone.utc)


# --- pricing -----------------------------------------------------------------

def fill_price(side: str, yes_bid, yes_ask):
    """
    What a taker pays for ONE contract: YES at the ask, NO at 1 - the YES bid
    (buying NO is selling YES at the bid). None if that side has no quote or
    the price isn't strictly between $0 and $1.
    """
    if side == "yes":
        price = yes_ask
    elif side == "no":
        price = None if yes_bid is None else 1.0 - yes_bid
    else:
        raise ValueError(f"side must be 'yes' or 'no', not {side!r}")
    if price is None:
        return None
    price = round(float(price), 4)
    return price if 0.0 < price < 1.0 else None


def order_fee(price: float, contracts: int) -> float:
    """Kalshi's taker fee for one order, in dollars (see kalshi_study.order_fee)."""
    return round(float(kalshi_study.order_fee(price, contracts)), 4)


def order_cost(price: float, contracts: int) -> float:
    """Everything paid up front: the contracts plus the fee."""
    return round(price * contracts + order_fee(price, contracts), 4)


def settlement(side: str, contracts: int, price: float, fee: float, result: str) -> dict:
    """
    {"won", "payout", "pnl"} for a position once its contract resolved to
    `result` ("yes" or "no"). A winning contract pays $1; the cost and the fee
    are gone either way.
    """
    won = side == result
    payout = float(contracts) if won else 0.0
    return {"won": won, "payout": payout, "pnl": round(payout - price * contracts - fee, 4)}


# --- the ledger --------------------------------------------------------------

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _read_events(path: str) -> list:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _settled_ids(path: str) -> set:
    return {e["trade_id"] for e in _read_events(path) if e.get("event") == "settle"}


def load_trades(path: str) -> list:
    """
    Every position, oldest first, as dicts: the "open" event's fields plus
    `status` ("open" or "settled") and, once settled, `outcome`, `payout`,
    `pnl` and `settled_at`. A settle row with no matching open row (a hand-
    edited file) is ignored.
    """
    trades = {}
    order = []
    for e in _read_events(path):
        tid = e.get("trade_id")
        if e.get("event") == "open" and tid and tid not in trades:
            trades[tid] = {
                "trade_id": tid,
                "opened_at": e["timestamp"],
                "ticker": e["ticker"],
                "side": e["side"],
                "contracts": int(_num(e["contracts"]) or 0),
                "price": _num(e["price"]),
                "fee": _num(e["fee"]) or 0.0,
                "yes_bid": _num(e["yes_bid"]),
                "yes_ask": _num(e["yes_ask"]),
                "strike": _num(e["strike"]),
                "close_time": e["close_time"],
                "spot": _num(e["spot"]),
                "model_p_yes": _num(e["model_p_yes"]),
                "composite": e["composite"] or None,
                "status": "open",
                "outcome": None, "payout": None, "pnl": None, "settled_at": None,
            }
            order.append(tid)
        elif e.get("event") == "settle" and tid in trades and trades[tid]["status"] == "open":
            trades[tid].update(status="settled", outcome=e["outcome"], payout=_num(e["payout"]),
                               pnl=_num(e["pnl"]), settled_at=e["timestamp"])
    return [trades[t] for t in order]


def parse_time(s):
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def place_order(cfg, market: dict, side: str, contracts: int, spot=None, model_p_yes=None,
                composite=None, now=None) -> dict:
    """
    Simulates one taker order against `market` (a kalshi_api.parse_market
    result, ideally fetched just now rather than from the poll cache) and
    appends it to the ledger. Returns {"ok": True, "trade": {...}} or
    {"ok": False, "reason": ...}; nothing is written on a refusal.

    `spot`, `model_p_yes` and `composite` are context recorded with the trade
    so it can be studied later (did the signals agree with the trades? did the
    model's gap to the market mean anything?). They never affect the fill.
    """
    now = now or utcnow()
    if side not in SIDES:
        return {"ok": False, "reason": f"Unknown side {side!r}."}
    if not isinstance(contracts, int) or isinstance(contracts, bool) or contracts < 1:
        return {"ok": False, "reason": "Order at least 1 whole contract."}
    if not market.get("available"):
        return {"ok": False, "reason": market.get("reason", "No open KXBTC15M market to trade.")}
    if market["close_time"] <= now:
        return {"ok": False, "reason": "This contract has already closed."}
    price = fill_price(side, market.get("yes_bid"), market.get("yes_ask"))
    if price is None:
        return {"ok": False, "reason": f"No usable {side.upper()} quote right now "
                                       f"(bid {market.get('yes_bid')}, ask {market.get('yes_ask')})."}

    fee = order_fee(price, contracts)
    trade_id = uuid.uuid4().hex[:12]
    row = [
        "open", trade_id, now.isoformat(), market["ticker"], side, contracts, price, fee,
        market["yes_bid"] if market.get("yes_bid") is not None else "",
        market["yes_ask"] if market.get("yes_ask") is not None else "",
        market["strike"], market["close_time"].isoformat(),
        spot if spot is not None else "",
        model_p_yes if model_p_yes is not None else "",
        composite or "",
        "", "", "",
    ]
    if not append_row(cfg.PAPER_TRADE_LOG_PATH, LEDGER_HEADER, row):
        return {"ok": False, "reason": f"Couldn't write {cfg.PAPER_TRADE_LOG_PATH} (see the console)."}
    return {"ok": True, "trade": {
        "trade_id": trade_id, "ticker": market["ticker"], "side": side, "contracts": contracts,
        "price": price, "fee": fee, "cost": round(price * contracts + fee, 4),
        "strike": market["strike"], "close_time": market["close_time"],
    }}


def settle_due(cfg, now=None, retry_seconds: float = SETTLE_RETRY_SECONDS) -> int:
    """
    Settles every open position whose contract has closed and been resolved by
    Kalshi, by appending a "settle" row for each. Returns how many it settled.
    A contract that has closed but isn't resolved yet is left open and asked
    about again after `retry_seconds`. Never raises: the dashboard calls this
    on every refresh.
    """
    now = now or utcnow()
    path = cfg.PAPER_TRADE_LOG_PATH
    due = []
    for t in load_trades(path):
        closes = parse_time(t["close_time"])
        if t["status"] == "open" and closes is not None and closes <= now:
            due.append(t)
    if not due:
        return 0

    askable = []
    for ticker in dict.fromkeys(t["ticker"] for t in due):
        last = _last_settle_attempt.get(ticker)
        if last is None or (now - last).total_seconds() >= retry_seconds:
            _last_settle_attempt[ticker] = now
            askable.append(ticker)
    # kalshi_tracking.resolve_outcomes returns {ticker: "yes"/"no"} for those
    # that have resolved, and skips (to retry later) any it can't reach.
    results = kalshi_tracking.resolve_outcomes(askable, {})

    settled = 0
    for t in due:
        result = results.get(t["ticker"])
        if result not in SIDES:
            continue
        s = settlement(t["side"], t["contracts"], t["price"], t["fee"], result)
        row = ["settle", t["trade_id"], now.isoformat()] + [""] * 12 + [result, s["payout"], s["pnl"]]
        # skip_if runs under the file lock, so a second tab settling the same
        # trade at the same moment can't write it twice.
        if append_row(path, LEDGER_HEADER, row, skip_if=lambda tid=t["trade_id"]: tid in _settled_ids(path)):
            settled += 1
    return settled


# --- reporting ---------------------------------------------------------------

def summarize(trades: list, min_sample: int = 30, noise_z: float = 2.0) -> dict:
    """
    The account in numbers. Rates are by contracts, so they weight a 5-lot as
    five. `breakeven_rate` is the win rate that would have made the settled
    trades exactly break even after fees (cost per contract, since a winner
    pays $1): the bar `win_rate` has to clear. `win_rate_se` is one standard
    error on the win rate, counting each contract (market) once however many
    orders were placed in it -- orders in one market share one outcome, so
    they are not independent evidence. `verdict` is "none" with nothing
    settled, "small_sample" below `min_sample` markets, "within_noise" while
    the gap to break-even is inside `noise_z` standard errors, else "ahead" or
    "behind".
    """
    settled = [t for t in trades if t["status"] == "settled"]
    n_open = sum(1 for t in trades if t["status"] == "open")
    out = {"n_open": n_open, "n_settled": len(settled), "n_markets": 0, "wins": 0, "losses": 0,
           "net_pnl": 0.0, "fees": 0.0, "staked": 0.0, "win_rate": None, "breakeven_rate": None,
           "win_rate_se": None, "roi": None, "verdict": "none"}
    if not settled:
        return out

    contracts = sum(t["contracts"] for t in settled)
    won_contracts = sum(t["contracts"] for t in settled if t["side"] == t["outcome"])
    out["wins"] = sum(1 for t in settled if t["side"] == t["outcome"])
    out["losses"] = len(settled) - out["wins"]
    out["net_pnl"] = round(sum(t["pnl"] for t in settled), 4)
    out["fees"] = round(sum(t["fee"] for t in settled), 4)
    out["staked"] = round(sum(t["price"] * t["contracts"] + t["fee"] for t in settled), 4)
    out["n_markets"] = len({t["ticker"] for t in settled})
    out["win_rate"] = won_contracts / contracts
    out["breakeven_rate"] = out["staked"] / contracts
    out["win_rate_se"] = math.sqrt(out["win_rate"] * (1 - out["win_rate"]) / out["n_markets"])
    out["roi"] = out["net_pnl"] / out["staked"] if out["staked"] else None

    gap = out["win_rate"] - out["breakeven_rate"]
    if out["n_markets"] < min_sample:
        out["verdict"] = "small_sample"
    elif abs(gap) < noise_z * max(out["win_rate_se"], 1e-9):
        out["verdict"] = "within_noise"
    else:
        out["verdict"] = "ahead" if gap > 0 else "behind"
    return out


VERDICT_TEXT = {
    "none": "Nothing has settled yet.",
    "small_sample": "Too few settled markets to tell skill from luck.",
    "within_noise": "The gap to break-even is within noise -- not distinguishable from luck.",
    "ahead": "Ahead of break-even by more than the noise. Keep going before believing it.",
    "behind": "Behind break-even by more than the noise.",
}

LIMITS_NOTE = (
    "Fills are at the ask (YES) or 1 - bid (NO) plus Kalshi's taker fee, and settle on Kalshi's own "
    "result. Not modelled: the size available at the quote (keep orders small), latency, and price "
    "moving while you decide -- real fills would be a little worse. The edge study found no strategy "
    "that beat these costs on 30,688 past contracts, so a profitable streak here is more likely luck "
    "than skill until the sample is large."
)


def report(cfg) -> str:
    trades = load_trades(cfg.PAPER_TRADE_LOG_PATH)
    s = summarize(trades, cfg.PAPER_MIN_SAMPLE, cfg.PAPER_NOISE_Z)
    lines = [f"Paper trades: {len(trades)} placed, {s['n_open']} open, {s['n_settled']} settled "
             f"across {s['n_markets']} market(s)."]
    if s["n_settled"]:
        lines += [
            f"Net P&L: ${s['net_pnl']:+,.2f}  (fees paid ${s['fees']:,.2f}; staked ${s['staked']:,.2f}; "
            f"ROI {s['roi'] * 100:+.1f}%)",
            f"Won {s['wins']}, lost {s['losses']}.  Win rate {s['win_rate'] * 100:.1f}% "
            f"(+/-{s['win_rate_se'] * 100:.1f} pts, 1 SE) vs break-even {s['breakeven_rate'] * 100:.1f}%.",
            VERDICT_TEXT[s["verdict"]],
        ]
    lines += ["", LIMITS_NOTE]
    return "\n".join(lines)


if __name__ == "__main__":
    import config as cfg
    n = settle_due(cfg)
    if n:
        print(f"Settled {n} position(s).\n")
    print(report(cfg))
